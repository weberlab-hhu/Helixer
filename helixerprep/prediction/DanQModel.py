#! /usr/bin/env python3
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional, MaxPooling1D, Dropout
from HelixerModel import HelixerModel, get_col_accuracy_fn


class DanQModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=8)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr1', '--dropout1', type=float, default=0.0)
        self.parser.add_argument('-dr2', '--dropout2', type=float, default=0.0)
        self.parse_args()

    def model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filter_depth,
                         kernel_size=self.kernel_size,
                         input_shape=(self.shape_train[1], 4),
                         padding="same",
                         activation="relu"))

        if self.pool_size > 1:
            model.add(MaxPooling1D(pool_size=self.pool_size, padding='same'))

        model.add(Dropout(self.dropout1))
        model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))

        model.add(Dropout(self.dropout2))
        model.add(Dense(self.pool_size * 3, activation='sigmoid'))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      sample_weight_mode='temporal',
                      metrics=[
                          'accuracy',
                          get_col_accuracy_fn(0),
                          get_col_accuracy_fn(1),
                          get_col_accuracy_fn(2),
                      ])

    def _gen_data(self, h5_file, shuffle, exclude_err_seqs=False, sample_intergenic=False):
        assert exclude_err_seqs, 'DanQ can only be run atm without any errors in the sequences'
        assert self.intergenic_chance == 1.0, 'Intergenic sampling not supported atm with DanQ'
        assert self.intergenic_sample_weight == 1.0, 'Intergenic sw are not applied currently'

        n_seq = h5_file['/data/X'].shape[0]
        err_samples = np.array(h5_file['/data/err_samples'])
        X, y = [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for n, i in enumerate(seq_indexes):
                if err_samples[i]:
                    continue
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                # apply intergenic sample weight value
                if n == len(seq_indexes) - 1 or len(X) == self.batch_size:
                    X = np.stack(X, axis=0)
                    y = np.stack(y, axis=0)
                    sw = np.ones((y.shape[:2]), dtype=np.int8)
                    if self.pool_size > 1:
                        if y.shape[1] % self.pool_size != 0:
                            overhang = self.pool_size - (y.shape[1] % self.pool_size)
                            y = np.pad(y, ((0, 0), (0, overhang), (0, 0)), 'constant',
                                       constant_values=(0, 0))
                            sw = np.ones((y.shape[0], y.shape[1] // self.pool_size), dtype=np.int8)
                            sw[:, -1] = 0
                        y = y.reshape((
                            y.shape[0],
                            y.shape[1] // self.pool_size,
                            self.pool_size * 3
                        ))
                    yield (X, y, sw)
                    X, y = [], []


if __name__ == '__main__':
    model = DanQModel()
    model.run()
