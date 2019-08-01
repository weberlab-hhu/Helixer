import sys
import numpy as np
from terminaltables import AsciiTable


class F1Counter():
    def __init__(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def add(self, tn, fp, fn, tp):
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.tp += tp

    def get_values(self):
        if self.tp == 0:
            # print('Warning: Number of TP is 0, returning 0 for all metrics')
            return 0, 0, 0, 0
        else:
            return self.tn, self.fp, self.fn, self.tp

    def __repr__(self):
        return '[F1Counter, tn: {}, fp: {}, fn: {}, tp: {}]'.format(self.tn, self.fp, self.fn, self.tp)


class F1Calculator():
    def __init__(self, generator, n_steps):
        self.generator = generator
        self.n_steps = n_steps
        self.counters = {
            'Genic': {
                'cds': (1, F1Counter(), F1Counter()),
                'intron': (2, F1Counter(), F1Counter()),
                'total': (None, F1Counter(), F1Counter())  # None for simpler implementation
            },
            'Intergenic': {
                'cds': (1, F1Counter(), F1Counter()),
                'intron': (2, F1Counter(), F1Counter()),
                'total': (None, F1Counter(), F1Counter())
            },
            'Global': {
                'tr': (0, F1Counter(), F1Counter()),
                'cds': (1, F1Counter(), F1Counter()),
                'intron': (2, F1Counter(), F1Counter()),
                'total': (None, F1Counter(), F1Counter())
            }
        }

    def print_f1_scores(self):
        for region_name, counters in self.counters.items():
            table = [['', 'Precision', 'Recall', 'F1-Score']]
            for col_name, (_, counter0, counter1) in counters.items():
                if col_name == 'total':
                    table.append(['',''])
                for cls, counter in enumerate([counter0, counter1]):
                    precision, recall, f1 = F1Calculator._calculate_f1(*counter.get_values())
                    name = [col_name + ' ' + str(cls)]
                    table.append(name + ['{:.4f}'.format(s) for s in [precision, recall, f1]])
            print('\n', AsciiTable(table, region_name).table, sep='')

    @staticmethod
    def _calculate_base_metrics(y_true, y_pred, cls):
        if cls == 0:
            # invert everything so as to calculate now for the 0 case
            y_true = np.logical_not(np.copy(y_true))
            y_pred = np.logical_not(np.copy(y_pred))
        tn = np.count_nonzero(np.logical_or(y_true, y_pred) == 0)
        fp = np.count_nonzero(np.logical_and(np.logical_not(y_true), y_pred))
        fn = np.count_nonzero(np.logical_and(y_true, np.logical_not(y_pred)))
        tp = np.count_nonzero(np.logical_and(y_true, y_pred))
        return tn, fp, fn, tp

    @staticmethod
    def _calculate_f1(tn, fp, fn, tp):
        if tp == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def progress(count, total):
        """Expects the lowest count to be 1"""
        bar_len = 40
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        if count < total:
            bar = '=' * (filled_len - 1) + '>' + '-' * (bar_len - filled_len)
        else:
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
        if count == 1:
            print()
        print('F1 Score: [{}] {}%'.format(bar, percents), end="\r")

    def count_and_calculate(self, model):
        for i in range(self.n_steps):
            F1Calculator.progress(i + 1, self.n_steps)
            batch_data = next(self.generator)
            if len(batch_data) == 3:
                # todo mask with sample_weights
                X, y_true, sample_weights = batch_data
                pass
            else:
                X, y_true = batch_data
            y_pred = np.round(model.predict_on_batch(X)).astype(bool)

            # reshape back to [:, :, 3] if we predicted more than one point per timestep
            if y_pred.shape[-1] > 3:
                n_per_step = y_pred.shape[-1] // 3
                original_shape = (y_pred.shape[0], y_pred.shape[1] * n_per_step, 3)
                y_true = y_true.reshape(original_shape)
                y_pred = y_pred.reshape(original_shape)

            self.count_and_calculate_one_batch(y_true, y_pred)
        self.print_f1_scores()

    def count_and_calculate_one_batch(self, y_true, y_pred):
        for region_name, counters in self.counters.items():
            if region_name == 'Genic':
                mask = y_true[:, :, 0].astype(bool)
            elif region_name == 'Intergenic':
                mask = np.logical_not(y_true[:, :, 0].astype(bool))
            else:
                mask = np.ones(y_true.shape[:2]).astype(bool)
            if np.all(mask == False):
                # don't count if there is nothing to count
                continue
            for col_name, (col_id, _, _) in counters.items():
                if col_name != 'total':
                    # actually use the col_id
                    y_true_masked = y_true[:, :, col_id][mask].ravel().astype(bool)
                    y_pred_masked = y_pred[:, :, col_id][mask].ravel().astype(bool)
                else:
                    y_true_masked = y_true[mask].ravel().astype(bool)
                    y_pred_masked = y_pred[mask].ravel().astype(bool)

                base_metrics_0 = F1Calculator._calculate_base_metrics(y_true_masked, y_pred_masked,
                                                                      cls=0)
                self.counters[region_name][col_name][1].add(*base_metrics_0)
                base_metrics_1 = F1Calculator._calculate_base_metrics(y_true_masked, y_pred_masked,
                                                                      cls=1)
                self.counters[region_name][col_name][2].add(*base_metrics_1)
