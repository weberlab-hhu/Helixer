#! /usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np

from keras_layer_normalization import LayerNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Dropout, Reshape, Activation, Input)

from helixer.prediction.HelixerModel import HelixerModel, HelixerSequence


class LSTMSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, batch_size, shuffle):
        super().__init__(model, h5_file, mode, batch_size, shuffle)

    def __getitem__(self, idx):
        X, y, sw, transitions, _, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size
        assert pool_size > 1, 'pooling size of <= 1 oh oh..'
        assert y.shape[1] % pool_size == 0, 'pooling size has to evenly divide seq len'

        X = self._mk_timestep_pools(X)
        y = self._mk_timestep_pools_class_last(y)

        # mark any multi-base timestep as error if any base has an error
        sw = sw.reshape((sw.shape[0], -1, pool_size))
        sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.float32)

        # only change sample weights during training (not even validation) as we don't calculate
        # a validation loss at the moment
        if self.mode == 'train':
            if self.class_weights is not None:
                # class weights are only used during training and validation to keep the loss
                # comparable and are additive for the individual timestep predictions
                # giving even more weight to transition points
                # class weights without pooling not supported yet
                # cw = np.array([0.8, 1.4, 1.2, 1.2], dtype=np.float32)
                cls_arrays = [np.any((y[:, :, :, col] == 1), axis=2) for col in range(4)]
                cls_arrays = np.stack(cls_arrays, axis=2).astype(np.int8)
                # add class weights to applicable timesteps
                cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:2] + (1,)))
                cw = np.sum(cw_arrays, axis=2)
                # multiply with previous sample weights
                sw = np.multiply(sw, cw)

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

        return X, y, sw


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--units', type=int, default=4, help='how many units per LSTM layer')
        self.parser.add_argument('--layers', type=str, default='1', help='how many LSTM layers')
        self.parser.add_argument('--pool-size', type=int, default=10, help='how many bp to predict at once')
        self.parser.add_argument('--dropout', type=float, default=0.0)
        self.parser.add_argument('--layer-normalization', action='store_true')
        self.parse_args()

        if self.layers.isdigit():
            n_layers = int(self.layers)
            self.layers = [self.units] * n_layers
        else:
            self.layers = eval(self.layers)
            assert isinstance(self.layers, list)
        for key in ["save_model_path", "prediction_output_path", "test_data",
                    "load_model_path", "data_dir"]:
            self.__dict__[key] = self.append_pwd(self.__dict__[key])

    @staticmethod
    def append_pwd(path):
        if path.startswith('/'):
            return path
        else:
            pwd = os.getcwd()
            return '{}/{}'.format(pwd, path)

    def sequence_cls(self):
        return LSTMSequence

    def model(self):
        values_per_bp = 4
        if self.input_coverage:
            values_per_bp = 6
        main_input = Input(shape=(None, self.pool_size * values_per_bp), dtype=self.float_precision,
                           name='main_input')
        x = Bidirectional(LSTM(self.layers[0], return_sequences=True))(main_input)

        # potential next layers
        if len(self.layers) > 1:
            for layer_units in self.layers[1:]:
                if self.dropout > 0.0:
                    x = Dropout(self.dropout)(x)
                if self.layer_normalization:
                    x = LayerNormalization()(x)
                x = Bidirectional(LSTM(layer_units, return_sequences=True))(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        x = Dense(self.pool_size * 4)(x)
        if self.pool_size > 1:
            x = Reshape((-1, self.pool_size, 4))(x)
        x = Activation('softmax', name='main')(x)

        model = Model(inputs=main_input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      sample_weight_mode='temporal')


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
