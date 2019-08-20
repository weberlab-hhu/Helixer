#! /usr/bin/env python3
import random
import numpy as np

from keras_layer_normalization import LayerNormalization
from keras.models import Sequential
from keras.layers import (Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional, MaxPooling1D, Dropout, Reshape,
                          Activation)
from HelixerModel import (HelixerModel, HelixerSequence,
                          acc_row, acc_g_row, acc_ig_row, acc_ig_oh, acc_g_oh)


class DanQSequence(HelixerSequence):
    def __getitem__(self, idx):
        pool_size = self.model.pool_size
        usable_idx_slice = self.usable_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.stack(self.x_dset[sorted(list(usable_idx_slice))])  # got to provide a sorted list of idx
        y = np.stack(self.y_dset[sorted(list(usable_idx_slice))])
        sw = np.ones((y.shape[0], y.shape[1] // pool_size), dtype=np.int8)
        if pool_size > 1:
            if y.shape[1] % pool_size != 0:
                # add additional values and mask them so everything divides evenly
                overhang = pool_size - (y.shape[1] % pool_size)
                y = np.pad(y, ((0, 0), (0, overhang), (0, 0)), 'constant',
                           constant_values=(0, 0))
                sw = np.pad(sw, ((0, 0), (0, 1)), 'constant', constant_values=(0, 0))
            if self.one_hot:
                # make labels 2d so we can use the standart softmax / loss functions
                y = y.reshape((
                    y.shape[0],
                    y.shape[1] // pool_size,
                    pool_size,
                    y.shape[-1],
                ))
            else:
                y = y.reshape((
                    y.shape[0],
                    y.shape[1] // pool_size,
                    -1
                ))
        return X, y, sw


class DanQModel(HelixerModel):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=8)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr1', '--dropout1', type=float, default=0.0)
        self.parser.add_argument('-dr2', '--dropout2', type=float, default=0.0)
        self.parser.add_argument('-ln', '--layer-normalization', action='store_true')
        self.parse_args()

        if not self.exclude_errors:
            print('\nRunning DanQ without --exclude-errors. This should only be done in test mode.')

    def sequence_cls(self):
        return DanQSequence

    def model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filter_depth,
                         kernel_size=self.kernel_size,
                         input_shape=(self.shape_train[1], 4),
                         padding="same",
                         activation="relu"))

        if self.pool_size > 1:
            model.add(MaxPooling1D(pool_size=self.pool_size, padding='same'))

        if self.layer_normalization:
            model.add(LayerNormalization())

        model.add(Dropout(self.dropout1))
        model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))
        model.add(Dropout(self.dropout2))

        if self.one_hot:
            label_dim = 4 if self.merged_introns else 5
            model.add(Dense(self.pool_size * label_dim))
            model.add(Reshape((-1, self.pool_size, label_dim)))
            model.add(Activation('softmax'))
        else:
            model.add(Dense(self.pool_size * 3,  activation='sigmoid'))
        return model

    def compile_model(self, model):
        if self.one_hot:
            model.compile(optimizer=self.optimizer,
                          loss='categorical_crossentropy',
                          sample_weight_mode='temporal',
                          metrics=[
                              'accuracy',
                              acc_g_oh,
                              acc_ig_oh,
                          ])
        else:
            model.compile(optimizer=self.optimizer,
                          loss='binary_crossentropy',
                          sample_weight_mode='temporal',
                          metrics=[
                              'accuracy',
                              acc_row,
                              acc_g_row,
                              acc_ig_row,
                          ])


if __name__ == '__main__':
    model = DanQModel()
    model.run()
