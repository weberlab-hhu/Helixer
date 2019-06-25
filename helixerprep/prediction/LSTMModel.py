#! /usr/bin/env python3
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, TimeDistributed, Dense, Bidirectional
from HelixerModel import HelixerModel, get_col_accuracy_fn


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parse_args()

    def model(self):
        model = Sequential()
        model.add(Bidirectional(
            CuDNNLSTM(self.units, return_sequences=True, input_shape=(None, 4)),
            input_shape=(None, 4)
        ))
        # model.add(Bidirectional(
            # LSTM(self.units, return_sequences=True, input_shape=(None, 4)),
            # input_shape=(None, 4)
        # ))
        # model.add(CuDNNLSTM(self.units, return_sequences=True, input_shape=(None, 4)))
        # model.add(LSTM(self.units, return_sequences=True, input_shape=(None, 4)))
        # model.add(TimeDistributed(Dense(3, activation='sigmoid')))
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
