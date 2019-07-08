#! /usr/bin/env python3
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Reshape
from HelixerModel import HelixerModel, get_col_accuracy_fn
import random
import numpy as np


class CNNModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--kernel_size', type=int, default=7)
        self.parser.add_argument('--final_kernel_size', type=int, default=128)
        self.parser.add_argument('--filter_depth', type=int, default=64)
        self.parser.add_argument('--n_layers', type=int, default=4)
        self.parse_args()

    def model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filter_depth,
                         kernel_size=self.kernel_size,
                         input_shape=(self.train_shape[1], 4),
                         padding="same",
                         activation="relu"))
        # -2 because first and last have different dimensions
        for _ in range(self.n_layers - 2):
            model.add(Conv1D(filters=self.filter_depth,
                             kernel_size=self.kernel_size,
                             padding="same",
                             activation="relu"))

        model.add(Conv1D(filters=3,
                         kernel_size=128,
                         activation="sigmoid",
                         padding="same"
                         ))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      #sample_weight_mode='temporal',
                      metrics=[
                          'accuracy',
                          get_col_accuracy_fn(0),
                          get_col_accuracy_fn(1),
                          get_col_accuracy_fn(2),
                      ])

    def _gen_data(self, h5_file, shuffle, exclude_erroneous_seqs=False, sample_intergenic=False):
        n_seq = h5_file['/data/X'].shape[0]
        X, y = [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for i in seq_indexes:
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                # apply intergenic sample weight value
                if len(X) == self.batch_size:
                    yield (
                        np.stack(X, axis=0),
                        np.stack(y, axis=0)
                    )
                    X, y = [], []


if __name__ == '__main__':
    model = CNNModel()
    model.run()
