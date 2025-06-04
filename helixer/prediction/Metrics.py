import sys
import torch
import os
import csv

from terminaltables import AsciiTable
from torchmetrics.classification import MulticlassConfusionMatrix

from helixer.prediction.Callback import Callback


# callback (separate out into training and prediction and test callbacks?
# or completely separate like Tony?, i.e. callbacks module
# with each class being a different file)
class ConfusionMatrixCallback(Callback):
    def __init__(self):
        self.genic_col_names = ['ig', 'utr', 'exon', 'intron']
        self.phase_col_names = ['no_phase', 'phase_0', 'phase_1', 'phase_2']
        self.genic_extra_cols = ['sub_genic', 'genic']
        self.genic_classes = len(self.genic_col_names)
        self.phase_classes = len(self.phase_col_names)
        self.genic_cm = MulticlassConfusionMatrix(num_classes=self.genic_classes)
        self.phase_cm = MulticlassConfusionMatrix(num_classes=self.phase_classes)

    @staticmethod
    def _argmax_y(tensor):
        arr = torch.argmax(tensor, dim=-1).ravel().type(torch.int8)
        return arr

    @staticmethod
    def _remove_masked_bases(y_true, y_pred, sw):
        """Remove bases marked as errors, should also remove zero padding"""
        sw = sw.type(bool)
        y_true = y_true[sw]
        y_pred = y_pred[sw]
        return y_true, y_pred

    # todo: this is definitely not working yet
    # todo: is this necessary if we overlap at the start, torch feeds directly in here;
    #  overlap predictions then before feeding it to different callbacks, why overlap those anyway, just feed in direct
    #  overlap, prediction per chunk also overlapped ones
    # def _overlap_all_data(self, batch_idx, y_true, y_pred, sw):
    #     assert len(y_pred.shape) == 4, "this reshape assumes shape is " \
    #                                    "(batch_size, chunk_size // pool, pool, label dim)" \
    #                                    "and apparently it is time to fix that, shape is {}".format(y_pred.shape)
    #     bs, cspool, pool, ydim = y_pred.shape
    #     y_pred = y_pred.reshape([bs, cspool * pool, ydim])
    #     y_pred = self.generator.ol_helper.overlap_predictions(batch_idx, y_pred)
    #     y_pred = y_pred.reshape([-1, cspool, pool, ydim])
    #     # edge handle sw & y_true (as done with y_pred and to restore 1:1 input output
    #     sw = self.generator.ol_helper.subset_input(batch_idx, sw)
    #     y_true = self.generator.ol_helper.subset_input(batch_idx, y_true)
    #     return y_true, y_pred, sw

    def count_and_calculate_one_batch(self, cm, y_true, y_pred, sw, overlap=False):
        # preprocess
        # if overlap:
        #    y_true, y_pred, sw = self._overlap_all_data(batch_idx, y_true, y_pred,
        #                                                sw)  # need a copy of sw here instead??
        y_true, y_pred = self._remove_masked_bases(y_true, y_pred, sw)
        y_true = self._argmax_y(y_true)
        y_pred = self._argmax_y(y_pred)
        # calculate matrix
        cm.update(preds=y_pred, target=y_true)

    def _get_precision_recall_f1(self, runner, cm, genic):
        tp = cm.diagonal()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        # Avoid division by zero
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        # if it's the genic confusion matrix, we compute two extra "classes"
        # subgenic = exon + intron and genic = exon + intron + utr
        if genic:
            summary_stats = []
            for i in [2, 1]:  # first subgenic then genic statistics calculation
                pr = tp[i:] / (tp[i:] + fp[i:] + 1e-8)
                r = tp[i:] / (tp[i:] + fn[i:] + 1e-8)
                f1_ = 2 * (pr * r) / (pr + r + 1e-8)
                summary_stats.append([pr, r, f1_])
            precision = torch.cat([precision, summary_stats[0][0], summary_stats[1][0]])
            recall = torch.cat([recall, summary_stats[0][1], summary_stats[1][1]])
            f1 = torch.cat([f1, summary_stats[0][2], summary_stats[1][2]])
            runner.current_genic_f1 = summary_stats[1][2].item()
        return precision, recall, f1

    def print_results(self, runner, cm, genic=False):
        for table, table_name in self.prep_tables(runner, cm, genic):
            print('\n', AsciiTable(table, table_name).table, sep='')

    def prep_tables(self, runner, cm, genic):
        # list of tables: [(table, table_name), (table, table_name), ...]
        tables = []
        col_names = self.genic_col_names if genic else self.phase_col_names

        # normalize per row
        cm_normalized = cm.float() / cm.sum(dim=1, keepdim=True)

        conf = [[''] + [name + '_pred' for name in col_names]]
        for name, row in zip(col_names, cm.tolist()):
            conf.append([name + '_ref'] + row)
        tables.append((conf, 'confusion_matrix'))

        # normalized
        norm_conf = [[''] + [name + '_pred' for name in col_names]]
        for name, row in zip(col_names, cm_normalized.tolist()):
            norm_conf.append([name + '_ref'] + [round(x, ndigits=4) for x in row])
        tables.append((norm_conf, 'normalized_confusion_matrix'))

        # F1
        stats = [['', 'Precision', 'Recall', 'F1-Score']]
        metrics = ''
        precision, recall, f1 = self._get_precision_recall_f1(runner, cm, genic)
        if genic:
            col_names = col_names + self.genic_extra_cols
        for i in range(len(col_names)):
            metrics += ['{:.4f}'.format(s) for s in [precision[i], recall[i], f1[i]]]
            stats.append([col_names[i]] + metrics)
            if genic and i == (len(col_names) - len(self.genic_extra_cols)):
                stats.append([''] * 4)
        tables.append((stats, 'F1_summary'))

        return tables

    # todo: put in later as option for test (for train maybe after each epoch,
    #  full run, call write-statics?, ask gpt)
    # def export_to_csvs(self, pathout):
    #     if pathout is not None:
    #         if not os.path.exists(pathout):
    #             os.makedirs(pathout)
    #         scores = self._get_scores()
    #         for table, table_name in self.prep_tables(scores):
    #             with open('{}/{}.csv'.format(pathout, table_name), 'w') as f:
    #                 writer = csv.writer(f)
    #                 for row in table:
    #                     writer.writerow(row)

    def end_calculation_and_logging(self, runner):
        # todo: log here as well when logging is enabled, print for now
        final_genic_cm = self.genic_cm.compute()
        final_phase_cm = self.phase_cm.compute()
        # print/log nicely
        self.print_results(runner, final_genic_cm, genic=True)
        self.print_results(runner, final_phase_cm)

    def on_validation_batch_end(self, runner):
        self.count_and_calculate_one_batch(self.genic_cm, runner.current_y_true[0], runner.current_y_pred[0],
                                           runner.current_sample_weights)
        self.count_and_calculate_one_batch(self.phase_cm, runner.current_y_true[1], runner.current_y_pred[1],
                                           runner.current_sample_weights)

    def on_validation_epoch_end(self, runner):
        self.end_calculation_and_logging(runner)

    def on_test_batch_end(self, runner):
        # HINT: overlap is excluded for now until its clear if the calculation needs to overlap itself or
        # if the dataset handles that
        self.count_and_calculate_one_batch(self.genic_cm, runner.current_y_true[0], runner.current_y_pred[0],
                                           runner.current_sample_weights)
        self.count_and_calculate_one_batch(self.phase_cm, runner.current_y_true[1], runner.current_y_pred[1],
                                           runner.current_sample_weights)

    def on_test_epoch_end(self, runner):
        self.end_calculation_and_logging(runner)  # todo: check out if it's easier to return just the genic f1 and report from here
