import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix


class ConfusionMatrix():
    def __init__(self, generator, label_dim):
        np.set_printoptions(suppress=True)  # do not use scientific notation for the print out
        self.generator = generator
        self.label_dim = label_dim

    def _reshape_data(self, arr, n_steps):
        arr = np.argmax(arr, axis=-1).astype(np.int8)
        arr = arr.reshape((arr.shape[0], -1))
        arr = arr[:, :n_steps].ravel()  # remove overhang
        return arr

    def _add_to_cm(self, cm, y_true, y_pred, X):
        """Put in extra function to be testable"""
        # remove possible zero padding
        non_padded_idx = np.any(y_true, axis=-1)
        y_pred = y_pred[non_padded_idx]
        y_true = y_true[non_padded_idx]

        y_pred = self._reshape_data(y_pred, X.shape[1])
        y_true = self._reshape_data(y_true, X.shape[1])
        cm += confusion_matrix(y_true, y_pred, labels=range(self.label_dim))
        return cm

    def _print_cm(self, cm):
        print()
        pprint(cm)
        print()

    def calculate_cm(self, model):
        cm = np.zeros((self.label_dim, self.label_dim))
        for i in range(len(self.generator)):
            print(i, '/', len(self.generator))
            X, y_true = self.generator[i][:2]
            y_pred = model.predict_on_batch(X)

            # throw away additional inputs and outputs
            if type(X) is list:
                X, additional_input = X
            if type(y_true) is list:
                y_pred, meta_pred = y_pred
                y_true, meta_true = y_true

            cm = self._add_to_cm(cm, y_true, y_pred, X)

        self._print_cm(cm)
        # divide columns in confusion matrix by # of predictions made per class
        class_sums = np.sum(cm, axis=1)
        cm /= class_sums[:, None]  # expand by one dim so broadcast work properly
        self._print_cm(cm)

