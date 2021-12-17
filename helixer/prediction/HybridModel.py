#! /usr/bin/env python3
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Bidirectional, MaxPooling1D, Dropout, Reshape,
                                     Activation, Input, BatchNormalization)
from helixer.prediction.HelixerModel import HelixerModel, HelixerSequence


class HybridSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, batch_size, shuffle):
        super().__init__(model, h5_file, mode, batch_size, shuffle)
        if self.class_weights is not None:
            assert not mode == 'test'  # only use class weights during training and validation

    def __getitem__(self, idx):
        X, y, sw, transitions, phases, _, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size

        if pool_size > 1:
            if X.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = X.shape[1] % pool_size
                X = X[:, :-overhang]
                if not self.only_predictions:
                    y = y[:, :-overhang]
                    sw = sw[:, :-overhang]
                    if self.predict_phase:
                        phases = phases[:, :-overhang]
                    if self.mode == 'train' and self.transition_weights is not None:
                        transitions = transitions[:, :-overhang]

            if not self.only_predictions:
                y = self._mk_timestep_pools_class_last(y)
                sw = sw.reshape((sw.shape[0], -1, pool_size))
                sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.int8)

            if self.mode == 'train':
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

                # todo, while now compressed, the following is still 1:1 with LSTM model... --> HelixerModel
                if self.transition_weights is not None:
                    transitions = self._mk_timestep_pools_class_last(transitions)
                    # more reshaping and summing  up transition weights for multiplying with sample weights
                    sw_t = self.compress_tw(transitions)
                    sw = np.multiply(sw_t, sw)

                if self.coverage_weights:
                    coverage_scores = coverage_scores.reshape((coverage_scores.shape[0], -1, pool_size))
                    # maybe offset coverage scores [0,1] by small number (bc RNAseq has issues too), default 0.0
                    if self.coverage_offset > 0.:
                        coverage_scores = np.add(coverage_scores, self.coverage_offset)
                    coverage_scores = np.mean(coverage_scores, axis=2)
                    sw = np.multiply(coverage_scores, sw)

            if self.predict_phase and not self.only_predictions:
                y_phase = self._mk_timestep_pools_class_last(phases)
                y = [y, y_phase]

        if self.only_predictions:
            return X
        else:
            return X, y, sw


class HybridModel(HelixerModel):
    def __init__(self, cli_args=None):
        super().__init__(cli_args=cli_args)
        self.parser.add_argument('--cnn-layers', type=int, default=1)
        self.parser.add_argument('--lstm-layers', type=int, default=1)
        self.parser.add_argument('--units', type=int, default=32)
        self.parser.add_argument('--filter-depth', type=int, default=32)
        self.parser.add_argument('--kernel-size', type=int, default=26)
        self.parser.add_argument('--pool-size', type=int, default=10)
        self.parser.add_argument('--dropout1', type=float, default=0.0)
        self.parser.add_argument('--dropout2', type=float, default=0.0)
        self.parse_args()

    @staticmethod
    def sequence_cls():
        return HybridSequence

    def model(self):
        overhang = self.shape_train[1] % self.pool_size
        values_per_bp = 4
        if self.input_coverage:
            values_per_bp = 6
        main_input = Input(shape=(None, values_per_bp), dtype=self.float_precision,
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

        if self.dropout1 > 0.0:
            x = Dropout(self.dropout1)(x)

        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        for _ in range(self.lstm_layers - 1):
            x = Bidirectional(LSTM(self.units, return_sequences=True))(x)

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            x = Dropout(self.dropout2)(x)

        if self.predict_phase:
            x = Dense(self.pool_size * 4 * 2)(x)  # predict twice a many floats
            x_genic, x_phase = tf.split(x, 2, axis=-1)

            x_genic = Reshape((-1, self.pool_size, 4))(x_genic)
            x_genic = Activation('softmax', name='genic')(x_genic)

            x_phase = Reshape((-1, self.pool_size, 4))(x_phase)
            x_phase = Activation('softmax', name='phase')(x_phase)

            outputs = [x_genic, x_phase]
        else:
            x = Dense(self.pool_size * 4)(x)
            x = Reshape((-1, self.pool_size, 4))(x)
            x = Activation('softmax', name='main')(x)
            outputs = [x]

        model = Model(inputs=main_input, outputs=outputs)
        return model

    def compile_model(self, model):
        if self.predict_phase:
            losses = ['categorical_crossentropy', 'categorical_crossentropy']
            loss_weights = [0.8, 0.2]
        else:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal')


if __name__ == '__main__':
    model = HybridModel()
    model.run()
