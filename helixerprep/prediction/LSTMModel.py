#! /usr/bin/env python3
import random
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, Dense, Bidirectional, Activation, Reshape
from HelixerModel import HelixerModel, HelixerSequence, acc_ig_oh, acc_g_oh


class LSTMSequence(HelixerSequence):
    def __getitem__(self, idx):
        pool_size = self.model.pool_size
        usable_idx_slice = self.usable_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.stack(self.x_dset[sorted(list(usable_idx_slice))])  # got to provide a sorted list of idx
        y = np.stack(self.y_dset[sorted(list(usable_idx_slice))])
        # sw = np.stack(self.sw_dset[sorted(list(usable_idx_slice))])

        # input multiple points at a time
        X = X.reshape((
            X.shape[0],
            X.shape[1] // pool_size,
            -1
        ))
        # make labels 2d so we can use the standard softmax / loss functions
        y = y.reshape((
            y.shape[0],
            y.shape[1] // pool_size,
            pool_size,
            y.shape[-1],
        ))
        return X, y


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-l', '--layers', type=int, default=1)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parse_args()
        assert self.exclude_errors

    def sequence_cls(self):
        return LSTMSequence

    def model(self):
        model = Sequential()

        model.add(Bidirectional(
            CuDNNLSTM(self.units, return_sequences=True, input_shape=(None, self.pool_size * 4)),
            input_shape=(None, self.pool_size * 4)
        ))

        # potential next layers
        if self.layers > 1:
            for _ in range(self.layers - 1):
                model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))

        model.add(Dense(self.pool_size * self.label_dim))
        model.add(Reshape((-1, self.pool_size, self.label_dim)))
        model.add(Activation('softmax'))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', acc_g_oh, acc_ig_oh])


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
