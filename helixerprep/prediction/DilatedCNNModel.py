#! /usr/bin/env python3
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from HelixerModel import HelixerModel, HelixerSequence

class DilatedCNNSequence(HelixerSequence):
    def __getitem__(self, idx):
        X, y, sw, _, _, _, = self._get_batch_data(idx)
        # sample weights are applied through the custom loss function
        self.sample_weights = sw
        return X, y

class DilatedCNNModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--kernel-size', type=int, default=7)
        self.parser.add_argument('--filter-depth', type=int, default=64)
        self.parser.add_argument('--double-filter-every', type=int, default=2)
        self.parser.add_argument('--dilation-multiplier', type=int, default=3)
        self.parser.add_argument('--dilation-max', type=int, default=100)
        self.parser.add_argument('--n-conv-layers', type=int, default=2)
        self.parser.add_argument('--n-hidden-layers', type=int, default=1)
        self.parser.add_argument('--hidden-layer-size', type=int, default=128)
        self.parser.add_argument('--dropout', type=float, default=0.1)
        self.parse_args()

        assert self.n_conv_layers > 1

    def sequence_cls(self):
        return DilatedCNNSequence

    def custom_loss(self, sample_weights):
        def loss(y_true, y_pred):
            # calculation taken from
            # https://github.com/keras-team/keras/blob/fb7f49ef5b07f2ceee1d2d6c45f273df6672734c/keras/backend/tensorflow_backend.py#L3570
            axis = -1
            y_pred /= tf.reduce_sum(y_pred, axis, True)
            # manual computation of crossentropy
            _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
            y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
            neg_log_likelihood = - tf.reduce_sum(y_true * tf.log(y_pred), axis)
            # mask by sample weights
            masked = tf.math.multiply(neg_log_likelihood, sample_weights)
            return masked
        return loss

    def model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filter_depth,
                         kernel_size=self.kernel_size,
                         input_shape=(self.shape_train[1], 4),
                         padding="same",
                         activation="relu"))

        dilation = 1
        for l in range(self.n_conv_layers - 1):
            if dilation * self.dilation_multiplier <= self.dilation_max:
                dilation *= self.dilation_multiplier
            if (l + 1) % self.double_filter_every == 0:
                self.filter_depth *= 2

            model.add(Conv1D(filters=self.filter_depth,
                             kernel_size=self.kernel_size,
                             dilation_rate=dilation,
                             padding="same",
                             activation="relu"))

        for _ in range(self.n_hidden_layers):
            model.add(Dropout(self.dropout))
            model.add(Dense(self.hidden_layer_size, activation="relu"))

        model.add(Dropout(self.dropout))
        model.add(Dense(4, activation="relu"))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      # loss='categorical_crossentropy',
                      loss=self.custom_loss(),
                      metrics=['accuracy'])


if __name__ == '__main__':
    model = DilatedCNNModel()
    model.run()