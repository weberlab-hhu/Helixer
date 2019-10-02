import numpy as np
from pprint import pprint
from collections import defaultdict
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
        scores = defaultdict(dict)
        for col, name in self.col_names.items():
            scores[col]['TP'] = self.cm[col, col]
            scores[col]['FP'] = np.sum(self.cm[np.arange(self.label_dim) != col, col])
            scores[col]['FN'] = np.sum(self.cm[col, np.arange(self.label_dim) != col])
            metrics = ConfusionMatrix._precision_recall_f1(scores[col]['TP'],
                                                           scores[col]['FP'],
                                                           scores[col]['FN'])
            scores[col]['precision'], scores[col]['recall'], scores[col]['f1'] = metrics

        tp_cds = scores[2]['TP'] + scores[3]['TP']
        fp_cds = scores[2]['FP'] + scores[3]['FP']
        fn_cds = scores[2]['FN'] + scores[3]['FN']
        _, _, f1_cds = ConfusionMatrix._precision_recall_f1(tp_cds, fp_cds, fn_cds)
        genic_acc = (self.cm[1, 1] + self.cm[2, 2] + self.cm[3, 3]) / np.sum(self.cm[1:])
        return scores, f1_cds, genic_acc

    def _print_results(self):
        normalized_cm = self._get_normalized_cm()
        print('\n')
        pprint(self.cm)
        print()
        pprint(normalized_cm)
        print()

        scores, f1_cds, genic_acc = self._get_composite_scores()
        for col, values in scores.items():
            print(self.col_names[col], 'f1: {:.4f}'.format(values['f1']))
        print('\ngenic_acc: {:.4f}'.format(genic_acc))
        print('f1_cds: {:.4f}\n'.format(f1_cds))

    def calculate_cm(self, model):
        for i in range(len(self.generator)):
            print(i, '/', len(self.generator), end="\r")
            X, y_true = self.generator[i][:2]
            y_pred = model.predict_on_batch(X)

            # throw away additional outputs
            if type(y_true) is list:
                y_pred, meta_pred = y_pred
                y_true, meta_true = y_true

            self._add_to_cm(y_true, y_pred)
        self._print_results()
