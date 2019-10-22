import numpy as np
from pprint import pprint
from collections import defaultdict
from terminaltables import AsciiTable
from sklearn.metrics import confusion_matrix


class ConfusionMatrix():
    def __init__(self, generator, label_dim):
        np.set_printoptions(suppress=True)  # do not use scientific notation for the print out
        self.generator = generator
        self.label_dim = label_dim
        self.cm = np.zeros((self.label_dim, self.label_dim))
        self.col_names = {0: 'ig', 1: 'utr', 2: 'exon', 3: 'intron'}

    def _reshape_data(self, arr):
        arr = np.argmax(arr, axis=-1).astype(np.int8)
        arr = arr.reshape((arr.shape[0], -1)).ravel()
        return arr

    def _add_to_cm(self, y_true, y_pred):
        """Put in extra function to be testable"""
        # remove possible zero padding
        non_padded_idx = np.any(y_true, axis=-1)
        y_pred = y_pred[non_padded_idx]
        y_true = y_true[non_padded_idx]

        y_pred = self._reshape_data(y_pred)
        y_true = self._reshape_data(y_true)
        self.cm += confusion_matrix(y_true, y_pred, labels=range(self.label_dim))

    def _get_normalized_cm(self):
        """Put in extra function to be testable"""
        class_sums = np.sum(self.cm, axis=1)
        normalized_cm = self.cm / class_sums[:, None]  # expand by one dim so broadcast work properly
        return normalized_cm

    @staticmethod
    def _precision_recall_f1(tp, fp, fn):
        precision  = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def _get_composite_scores(self):
        def add_to_scores(d):
            metrics = ConfusionMatrix._precision_recall_f1(d['TP'], d['FP'], d['FN'])
            d['precision'], d['recall'], d['f1'] = metrics

        scores = defaultdict(dict)
        # single column metrics
        for col in range(self.label_dim):
            d = scores[self.col_names[col]]
            d['TP'] = self.cm[col, col]
            d['FP'] = np.sum(self.cm[np.arange(self.label_dim) != col, col])
            d['FN'] = np.sum(self.cm[col, np.arange(self.label_dim) != col])
            add_to_scores(d)

        composite_metrics = {
            'cds': ['exon', 'intron'],
            'genic': ['utr', 'exon', 'intron'],
        }
        for c_metric, parts in composite_metrics.items():
            d = scores[c_metric]
            for base_metric in ['TP', 'FP', 'FN']:
                d[base_metric] = sum([scores[m][base_metric] for m in parts])
            add_to_scores(d)

        return scores

    def _print_results(self):
        normalized_cm = self._get_normalized_cm()
        print('\n')
        pprint(self.cm)
        print()
        pprint(normalized_cm)
        print()

        scores = self._get_composite_scores()
        table = [['', 'Precision', 'Recall', 'F1-Score']]
        for i, (name, values) in enumerate(scores.items()):
            metrics = ['{:.4f}'.format(s) for s in list(values.values())[3:]]
            table.append([name] + metrics)
            if i == 3:
                table.append([''] * 4)
        print('\n', AsciiTable(table, '').table, sep='')

        # return genic f1 for model saving in custom callback
        return scores['genic']['f1']

    def calculate_cm(self, model):
        for i in range(len(self.generator)):
            print(i, '/', len(self.generator), end="\r")
            X, y_true = self.generator[i][:2]  # throw away sample weights if there are any
            y_pred = model.predict_on_batch(X)

            # throw away additional outputs
            if type(y_true) is list:
                y_pred, meta_pred = y_pred
                y_true, meta_true = y_true

            self._add_to_cm(y_true, y_pred)
        return self._print_results()
