import numpy as np
import os
import csv
from collections import defaultdict
from terminaltables import AsciiTable
from scipy.sparse import coo_matrix
from ignite.metrics.confusion_matrix import ConfusionMatrix


class HelixerConfusionMatrix:

    def __init__(self, col_names=None):
        if col_names is None:
            col_names = ['ig', 'utr', 'exon', 'intron']
        self.col_names = {i: name for i, name in enumerate(col_names)}
        self.n_classes = len(self.col_names)
        self.cm_obj = ConfusionMatrix(self.n_classes, device='cuda')
        self.cm = self.cm_obj.confusion_matrix  # shortcut

    def add_to_cm(self, y_true, y_pred):
        """Put in extra function to be testable"""
        # add to confusion matrix as long as _some_ bases were not masked
        if y_pred.numel() > 0:
            self.cm_obj.update((y_pred, y_true))

    def _get_normalized_cm(self):
        """Put in extra function to be testable"""
        class_sums = self.cm.sum(dim=1)
        normalized_cm = self.cm / class_sums[:, None]  # expand by one dim so broadcast work properly
        return normalized_cm

    @staticmethod
    def _precision_recall_f1(tp, fp, fn):
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0  # avoid an error due to division by 0
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    def _total_accuracy(self):
        return self.cm.trace() / self.cm.sum()

    @staticmethod
    def _add_to_scores(d):
        metrics = HelixerConfusionMatrix._precision_recall_f1(d['TP'], d['FP'], d['FN'])
        d['precision'], d['recall'], d['f1'] = metrics

    def _get_scores(self):
        scores = defaultdict(dict)
        # single column metrics
        for col in range(self.n_classes):
            name = self.col_names[col]
            d = scores[name]
            not_col = np.arange(self.n_classes) != col
            d['TP'] = self.cm[col, col]
            d['FP'] = self.cm[not_col, col].sum()
            d['FN'] = self.cm[col, not_col].sum()

            HelixerConfusionMatrix._add_to_scores(d)
        return scores

    def _print_results(self, scores):
        for table, table_name in self.prep_tables(scores):
            print('\n', AsciiTable(table, table_name).table, sep='')
        print('Total acc: {:.4f}'.format(self._total_accuracy()))

    def print_cm(self):
        scores = self._get_scores()
        self._print_results(scores)

    def prep_tables(self, scores):
        out = []
        names = list(self.col_names.values())

        # confusion matrix
        cm = [[''] + [x + '_pred' for x in names]]
        for i, row in enumerate(self.cm.tolist()):
            cm.append([names[i] + '_ref'] + row)
        out.append((cm, 'confusion_matrix'))

        # normalized
        normalized_cm = [cm[0]]
        for i, row in enumerate(self._get_normalized_cm().tolist()):
            normalized_cm.append([names[i] + '_ref'] + [round(x, ndigits=4) for x in row])
        out.append((normalized_cm, 'normalized_confusion_matrix'))

        # F1
        table = [['', 'Precision', 'Recall', 'F1-Score']]
        for i, (name, values) in enumerate(scores.items()):
            # check if there is an entropy value comming (only for single type classes)
            metrics = ['']
            metrics += ['{:.4f}'.format(s) for s in list(values.values())[3:]]  # [3:] to skip TP, FP, FN
            table.append([name] + metrics)
            if i == (len(names) - 1) and len(scores) > len(names):
                table.append([''] * 4)
        out.append((table, 'F1_summary'))

        return out

    def export_to_csvs(self, pathout):
        if pathout is not None:
            if not os.path.exists(pathout):
                os.mkdir(pathout)

            for table, table_name in self.prep_tables():
                with open('{}/{}.csv'.format(pathout, table_name), 'w') as f:
                    writer = csv.writer(f)
                    for row in table:
                        writer.writerow(row)


class HelixerConfusionMatrixGenic(HelixerConfusionMatrix):
    """Extension of HelixerConfusionMatrix that just adds the calculation of the composite scores"""

    def _get_scores(self):
        scores = super()._get_scores()

        # legacy cds score that works the same as the cds_f1 with the 3 column multi class encoding
        # essentiall merging the predictions as if error between exon and intron did not matter
        d = scores['legacy_cds']
        cm = self.cm
        d['TP'] = cm[2, 2] + cm[3, 3] + cm[2, 3] + cm[3, 2]
        d['FP'] = cm[0, 2] + cm[0, 3] + cm[1, 2] + cm[1, 3]
        d['FN'] = cm[2, 0] + cm[3, 0] + cm[2, 1] + cm[3, 1]
        HelixerConfusionMatrix._add_to_scores(d)

        # subgenic metric is essentially the same as the genic one
        # pretty redundant code to below, but done for minimizing the risk to mess up (for now)
        d = scores['sub_genic']
        for base_metric in ['TP', 'FP', 'FN']:
            d[base_metric] = sum([scores[m][base_metric] for m in ['exon', 'intron']])
        HelixerConfusionMatrix._add_to_scores(d)

        # genic metrics are calculated by summing up TP, FP, FN, essentially calculating a weighted
        # sum for the individual metrics. TP of the intergenic class are not taken into account
        d = scores['genic']
        for base_metric in ['TP', 'FP', 'FN']:
            d[base_metric] = sum([scores[m][base_metric] for m in ['utr', 'exon', 'intron']])
        HelixerConfusionMatrix._add_to_scores(d)

        return scores
