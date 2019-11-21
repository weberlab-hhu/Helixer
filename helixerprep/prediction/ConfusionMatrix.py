import numpy as np
import os
import csv
from collections import defaultdict
from terminaltables import AsciiTable
from sklearn.metrics import confusion_matrix


class ConfusionMatrix():
    def __init__(self, generator):
        np.set_printoptions(suppress=True)  # do not use scientific notation for the print out
        self.generator = generator
        self.cm = np.zeros((4, 4))
        self.col_names = {0: 'ig', 1: 'utr', 2: 'exon', 3: 'intron'}

    @staticmethod
    def _reshape_data(arr):
        arr = np.argmax(arr, axis=-1).astype(np.int8)
        arr = arr.reshape((arr.shape[0], -1)).ravel()
        return arr

    @staticmethod
    def _remove_masked_bases(y_true, y_pred, sw):
        """Remove bases marked as errors, should also remove zero padding"""
        sw = sw.astype(np.bool)
        y_pred = y_pred[sw]
        y_true = y_true[sw]
        return y_pred, y_true

    def _add_to_cm(self, y_true, y_pred, sw):
        """Put in extra function to be testable"""
        y_pred, y_true = ConfusionMatrix._remove_masked_bases(y_true, y_pred, sw)
        y_pred = ConfusionMatrix._reshape_data(y_pred)
        y_true = ConfusionMatrix._reshape_data(y_true)
        self.cm += confusion_matrix(y_true, y_pred, labels=range(4))

    def count_and_calculate_one_batch(self, y_true, y_pred, sw):
        self._add_to_cm(y_true, y_pred, sw)

    def _get_normalized_cm(self):
        """Put in extra function to be testable"""
        class_sums = np.sum(self.cm, axis=1)
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
        return np.trace(self.cm) / np.sum(self.cm)

    def _get_composite_scores(self):
        def add_to_scores(d):
            metrics = ConfusionMatrix._precision_recall_f1(d['TP'], d['FP'], d['FN'])
            d['precision'], d['recall'], d['f1'] = metrics

        scores = defaultdict(dict)
        # single column metrics
        for col in range(4):
            d = scores[self.col_names[col]]
            not_col = np.arange(4) != col
            d['TP'] = self.cm[col, col]
            d['FP'] = np.sum(self.cm[not_col, col])
            d['FN'] = np.sum(self.cm[col, not_col])
            add_to_scores(d)

        # legacy cds score that works the same as the cds_f1 with the 3 column multi class encoding
        # essentiall merging the predictions as if error between exon and intron did not matter
        d = scores['legacy_cds']
        d['TP'] = self.cm[2, 2] + self.cm[3, 3] + self.cm[2, 3] + self.cm[3, 2]
        d['FP'] = self.cm[0, 2] + self.cm[0, 3] + self.cm[1, 2] + self.cm[1, 3]
        d['FN'] = self.cm[2, 0] + self.cm[3, 0] + self.cm[2, 1] + self.cm[3, 1]
        add_to_scores(d)

        # genic metrics are calculated by summing up TP, FP, FN, essentially calculating a weighted
        # sum for the individual metrics. TP of the intergenic class are not taken into account
        d = scores['genic']
        for base_metric in ['TP', 'FP', 'FN']:
            d[base_metric] = sum([scores[m][base_metric] for m in ['utr', 'exon', 'intron']])
        add_to_scores(d)

        return scores

    def calculate_cm(self, model):
        for i in range(len(self.generator)):
            print(i, '/', len(self.generator) - 1, end="\r")

            X, y_true, sw = self.generator[i]
            y_pred = model.predict_on_batch(X)

            # throw away additional outputs
            if type(y_true) is list:
                y_pred, meta_pred = y_pred
                y_true, meta_true = y_true
            if type(sw) is list:
                sw = sw[0]

            self._add_to_cm(y_true, y_pred, sw)
        return self._print_results()

    def _print_results(self):
        scores = self._get_composite_scores()
        for table, table_name in self.prep_tables():
            print('\n', AsciiTable(table, table_name).table, sep='')
        print('Total acc: {:.4f}'.format(self._total_accuracy()))

        # return genic f1 for model saving in custom callback or other uses
        return scores['genic']['f1']

    def print_cm(self):
        self._print_results()

    def prep_tables(self):
        out = []
        names = ['ig', 'utr', 'exon', 'intron']

        # confusion matrix
        cm = [[''] + [x + '_pred' for x in names]]
        for i, row in enumerate(self.cm.astype(int).tolist()):
            cm.append([names[i] + '_ref'] + row)
        out.append((cm, 'confusion_matrix'))

        # normalized
        normalized_cm = [cm[0]]
        for i, row in enumerate(self._get_normalized_cm().tolist()):
            normalized_cm.append([names[i] + '_ref'] + [round(x, ndigits=4) for x in row])
        out.append((normalized_cm, 'normalized_confusion_matrix'))

        # F1
        scores = self._get_composite_scores()
        table = [['', 'Precision', 'Recall', 'F1-Score']]
        for i, (name, values) in enumerate(scores.items()):
            # 3: below to skip TP, FP, FN
            metrics = ['{:.4f}'.format(s) for s in list(values.values())[3:]]
            table.append([name] + metrics)
            if i == 3:
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



