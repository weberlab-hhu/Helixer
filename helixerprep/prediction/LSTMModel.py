#! /usr/bin/env python3
import random
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, Dense, Bidirectional
from HelixerModel import HelixerModel, acc_row, acc_g_row, acc_ig_row


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-l', '--layers', type=int, default=1)
        self.parse_args()

    def model(self):
        model = Sequential()
        # input layer
        if self.only_cpu:
            model.add(Bidirectional(
                LSTM(self.units, return_sequences=True, input_shape=(None, 4)),
                input_shape=(None, 4)
            ))
        else:
            model.add(Bidirectional(
                CuDNNLSTM(self.units, return_sequences=True, input_shape=(None, 4)),
                input_shape=(None, 4)
            ))

        # potential next layers
        if self.layers > 1:
            for _ in range(self.layers - 1):
                if self.only_cpu:
                    model.add(Bidirectional(LSTM(self.units, return_sequences=True)))
                else:
                    model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))

        model.add(Dense(3, activation='sigmoid'))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      sample_weight_mode='temporal',
                      metrics=[
                          'accuracy',
                          acc_row,
                          acc_g_row,
                          acc_ig_row,
                      ])

    def _gen_data(self, h5_file, shuffle, exclude_err_seqs=False):
        n_seq = h5_file['/data/X'].shape[0]
        if exclude_err_seqs:
            err_samples = np.array(h5_file['/data/err_samples'])
        X, y, sw = [], [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for n, i in enumerate(seq_indexes):
                if exclude_err_seqs and err_samples[i]:
                    continue
                sw = h5_file['/data/sample_weights'][i]
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                if n == len(seq_indexes) - 1 or len(X) == self.batch_size:
                    yield (
                        np.stack(X, axis=0),
                        np.stack(y, axis=0),
                        np.stack(sw, axis=0)
                    )
                    X, y, sw = [], [], []


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
