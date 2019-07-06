#! /usr/bin/env python3
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, TimeDistributed, Dense, Bidirectional
from HelixerModel import HelixerModel, get_col_accuracy_fn


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-l', '--layers', type=int, default=1)
        self.parser.add_argument('-bd', '--bidirectional', action='store_true')
        self.parse_args()

    def model(self):
        model = Sequential()
        # input layer
        if self.bidirectional:
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
        else:
            if self.only_cpu:
                model.add(LSTM(self.units, return_sequences=True, input_shape=(None, 4)))
            else:
                model.add(CuDNNLSTM(self.units, return_sequences=True, input_shape=(None, 4)))

        # potential next layers
        if self.layers > 1:
            for _ in range(self.layers - 1):
                if self.bidirectional:
                    if self.only_cpu:
                        model.add(Bidirectional(LSTM(self.units, return_sequences=True)))
                    else:
                        model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))
                else:
                    if self.only_cpu:
                        model.add(LSTM(self.units, return_sequences=True))
                    else:
                        model.add(CuDNNLSTM(self.units, return_sequences=True))

        model.add(Dense(3, activation='sigmoid'))
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


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
