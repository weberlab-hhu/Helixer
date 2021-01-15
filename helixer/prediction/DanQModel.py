#! /usr/bin/env python3
import numpy as np

from keras_layer_normalization import LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Bidirectional, MaxPooling1D, Dropout, Reshape,
                                     Activation, Input, BatchNormalization)
from HelixerModel import HelixerModel, HelixerSequence


class DanQSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, batch_size, shuffle):
        super().__init__(model, h5_file, mode, batch_size, shuffle)
        if self.class_weights is not None:
            assert not mode == 'test'  # only use class weights during training and validation

    def __getitem__(self, idx):
        X, y, sw, _, transitions, _ = self._get_batch_data(idx)
        pool_size = self.model.pool_size

        if pool_size > 1:
            if y.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = y.shape[1] % pool_size
                X = X[:, :-overhang]
                y = y[:, :-overhang]
                sw = sw[:, :-overhang]
                if self.transition_weights is not None:
                    transitions = transitions[:, :-overhang]

            # make labels 2d so we can use the standard softmax / loss functions
            y = y.reshape((
                y.shape[0],
                y.shape[1] // pool_size,
                pool_size,
                y.shape[-1],
            ))

            sw = sw.reshape((sw.shape[0], -1, pool_size))
            sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.int8)

            if self.class_weights is not None:
                # class weights are additive for the individual timestep predictions
                # giving even more weight to transition points
                # class weights without pooling not supported yet
                # cw = np.array([1.0, 1.2, 1.0, 0.8], dtype=np.float32)
                cls_arrays = [np.any((y[:, :, :, col] == 1), axis=2) for col in range(4)]
                cls_arrays = np.stack(cls_arrays, axis=2).astype(np.int8)
                # add class weights to applicable timesteps
                cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:2] + (1,)))
                cw = np.sum(cw_arrays, axis=2)
                sw = np.multiply(cw, sw)

            if self.transition_weights is not None:
                transitions = transitions.reshape((
                    transitions.shape[0],
                    transitions.shape[1] // pool_size,
                    pool_size,
                    transitions.shape[-1],
                ))
                # todo, this looks very redundant with LSTMModel around _squish_tw_to_sw
                #   could both go to HelixerModel?
                sw_t = [np.any((transitions[:, :, :, col] == 1), axis=2) for col in range(6)]
                sw_t = np.stack(sw_t, axis=2).astype(np.int8)
                sw_t = np.multiply(sw_t, self.transitions)

                sw_t = np.sum(sw_t, axis=2)
                where_are_ones = np.where(sw_t == 0)
                sw_t[where_are_ones[0], where_are_ones[1]] = 1
                sw = np.multiply(sw_t, sw)

        return X, y, sw


class DanQModel(HelixerModel):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--cnn-layers', type=int, default=1)
        self.parser.add_argument('--lstm-layers', type=int, default=1)
        self.parser.add_argument('--units', type=int, default=32)
        self.parser.add_argument('--filter-depth', type=int, default=32)
        self.parser.add_argument('--kernel-size', type=int, default=26)
        self.parser.add_argument('--pool-size', type=int, default=10)
        self.parser.add_argument('--dropout1', type=float, default=0.0)
        self.parser.add_argument('--dropout2', type=float, default=0.0)
        self.parser.add_argument('--layer-normalization', action='store_true')
        self.parse_args()

    @staticmethod
    def sequence_cls():
        return DanQSequence

    def model(self):
        overhang = self.shape_train[1] % self.pool_size
        main_input = Input(shape=(None, 4), dtype=self.float_precision,
                           name='main_input')
        x = Conv1D(filters=self.filter_depth,
                   kernel_size=self.kernel_size,
                   padding="same",
                   activation="relu")(main_input)

        # if there are additional CNN layers
        for _ in range(self.cnn_layers - 1):
            x = BatchNormalization()(x)
            x = Conv1D(filters=self.filter_depth,
                       kernel_size=self.kernel_size,
                       padding="same",
                       activation="relu")(x)

        if self.pool_size > 1:
            x = Reshape((-1, self.pool_size * self.filter_depth))(x)
            # x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.layer_normalization:
            x = LayerNormalization()(x)
        if self.dropout1 > 0.0:
            x = Dropout(self.dropout1)(x)

        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        for _ in range(self.lstm_layers - 1):
            if self.layer_normalization:
                x = LayerNormalization()(x)
            x = Bidirectional(LSTM(self.units, return_sequences=True))(x)

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            x = Dropout(self.dropout2)(x)

        x = Dense(self.pool_size * 4)(x)
        x = Reshape((-1, self.pool_size, 4))(x)
        x = Activation('softmax', name='main')(x)

        outputs = [x]
        model = Model(inputs=main_input, outputs=outputs)
        return model

    def compile_model(self, model):

        losses = ['categorical_crossentropy']
        loss_weights = [1.0]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal')


if __name__ == '__main__':
    model = DanQModel()
    model.run()
