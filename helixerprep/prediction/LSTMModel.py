#! /usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import random
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras_layer_normalization import LayerNormalization
from keras.models import Sequential, Model
from keras.layers import (Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional, Dropout, Reshape, Activation,
                          concatenate, Input, Lambda)
from helixerprep.prediction.HelixerModel import HelixerModel, HelixerSequence


class LSTMSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, shuffle, filter_by_score=False, filter_quantile=0.05):
        super().__init__(model, h5_file, mode, shuffle, filter_by_score, filter_quantile)
        if self.class_weights is not None:
            assert not mode == 'test'  # only use class weights during training and validation
        if self.error_weights:
            assert not mode == 'test'

    def __getitem__(self, idx):
        X, y, sw, error_rates, gene_lengths, transitions, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size
        assert pool_size > 1, 'pooling size of <= 1 oh oh..'
        assert y.shape[1] % pool_size == 0, 'pooling size has to evenly divide seq len'

        # augment first, before anything else and only during training
        if self.augment and self.mode == 'train':
            X, y, sw = self._augment(X, y, sw)

        X = X.reshape((
            X.shape[0],
            X.shape[1] // pool_size,
            -1
        ))
        # make labels 2d so we can use the standard softmax / loss functions
        # y = y.reshape((
            # 2,
            # y.shape[1],
            # y.shape[2] // pool_size,
            # pool_size,
            # y.shape[-1],
        # ))

        y = y.reshape((
            y.shape[0],
            y.shape[1] // pool_size,
            pool_size,
            y.shape[-1],
        ))

        # mark any multi-base timestep as error if any base has an error
        sw = sw.reshape((sw.shape[0], -1, pool_size))
        sw = np.logical_not(np.any(sw == 0, axis=-1)).astype(np.float32)

        # split y and sw for the two outputs
        # y_split = self._split_and_squeeze(y)
        # sw_split = self._split_and_squeeze(sw)
        # return X, y_split[0], sw_split[0]

        return X, y, sw

    @staticmethod
    def _split_and_squeeze(arr):
        arr = np.split(arr, 2, axis=0)
        arr = [np.squeeze(sub_arr) for sub_arr in arr]
        return arr


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-l', '--layers', type=str, default='1')
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr', '--dropout', type=float, default=0.0)
        self.parser.add_argument('-ln', '--layer-normalization', action='store_true')
        self.parse_args()

        if self.layers.isdigit():
            n_layers = int(self.layers)
            self.layers = [self.units] * n_layers
        else:
            self.layers = eval(self.layers)
            assert isinstance(self.layers, list)

    def sequence_cls(self):
        return LSTMSequence

    def model(self):
        main_input = Input(shape=(None, self.pool_size * 4), dtype=self.float_precision,
                           name='main_input')
        x = Bidirectional(CuDNNLSTM(self.layers[0], return_sequences=True))(main_input)

        # potential next layers
        if len(self.layers) > 1:
            for layer_units in self.layers[1:]:
                if self.dropout > 0.0:
                    x = Dropout(self.dropout)(x)
                if self.layer_normalization:
                    x = LayerNormalization()(x)
                x = Bidirectional(CuDNNLSTM(layer_units, return_sequences=True))(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        # x = Dense(self.pool_size * 4 * 2)(x)
        # x1, x2 = Lambda(lambda a: tf.split(a, 2, axis=2))(x)
        # x1 = Reshape((-1, self.pool_size, 4))(x1)
        # x2 = Reshape((-1, self.pool_size, 4))(x2)
        # x1 = Activation('softmax', name='y_plus')(x1)
        # x2 = Activation('softmax', name='y_minus')(x2)

        x = Dense(self.pool_size * 4)(x)
        x = Reshape((-1, self.pool_size, 4))(x)
        x = Activation('softmax', name='y_plus')(x)

        model = Model(inputs=main_input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      # loss_weights=[0.5, 0.5],
                      sample_weight_mode='temporal')


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
