#! /usr/bin/env python3
from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Flatten, Reshape, Input, BatchNormalization, Activation, MaxPool1D, Dropout, \
    Concatenate
from HelixerModel import HelixerModel, get_col_accuracy_fn
import random
import numpy as np


class InceptionModel(HelixerModel):
    """Mini-inception like 1D CNN"""
    # This is specifically modelled to replicate the version used by Christopher Guenter in his bachelor's thesis
    # so that we can compare current results (with different data) to previous

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--no_conv_dropout', action="store_true")
        self.parse_args()

    def model(self):
        # some non-parameters, that should be
        momentum_vals = [0.7, 0.9, 0.9]
        dropout_vals = [0.5, 0.6, 0.7]
        kernel_size_vals = [[21], [1], [9]]
        to_pool = [True, False, False]

        input = Input(shape=(self.shape_train[1], 4))

        # pre-inception stem (3X convolution)
        current = input
        for momentum, dropout, kernel_size, pool in zip(momentum_vals,
                                                        dropout_vals,
                                                        kernel_size_vals,
                                                        to_pool):

            conv1 = Conv1D(filters=24,
                           kernel_size=kernel_size,
                           padding="same",
                           activation=None)(current)

            bn1 = BatchNormalization(axis=-1,
                                     momentum=momentum,
                                     epsilon=0.001,
                                     center=True,
                                     scale=True)(conv1)

            current = Activation(activation="relu")(bn1)

            if pool:
                current = MaxPool1D(pool_size=[2], strides=2)(current)

            if not self.no_conv_dropout:
                current = Dropout(rate=dropout)(current)

        # inception modules
        # more probably should-be-paramenters
        inception_dropout = 0.9
        filter_depth_vars = [48, 96, 96]
        current = MaxPool1D(pool_size=[2], strides=2)(current)
        for filters in filter_depth_vars:
            # 1x
            conv_in1 = Conv1D(filters=filters,
                              kernel_size=[1],
                              padding="same",
                              activation="relu")(current)
            # 9x
            conv_in2 = Conv1D(filters=filters,
                              kernel_size=[1],
                              padding="same",
                              activation="relu")(current)
            conv_in2 = Conv1D(filters=48,
                              kernel_size=[9],
                              padding="same",
                              activation="relu")(conv_in2)

            # 15x
            conv_in3 = Conv1D(filters=filters,
                              kernel_size=[1],
                              padding="same",
                              activation="relu")(current)
            conv_in3 = Conv1D(filters=filters,
                              kernel_size=[15],
                              padding="same",
                              activation="relu")(conv_in3)

            if not self.no_conv_dropout:
                conv_in1 = Dropout(rate=inception_dropout)(conv_in1)
                conv_in2 = Dropout(rate=inception_dropout)(conv_in2)
                conv_in3 = Dropout(rate=inception_dropout)(conv_in3)

            current = Concatenate(axis=2)([conv_in1, conv_in2, conv_in3])

        # post-inception FCs
        flattened = Flatten()(current)
        dense = Dense(units=192, activation="relu")(flattened)
        dense_drop = Dropout(rate=0.8)(dense)

        flat_preds = Dense(units=3 * self.shape_train[1], activation="sigmoid")(dense_drop)
        pred_logits = Reshape(target_shape=(self.shape_train[1], 3))(flat_preds)

        # add input & output to actual keras Model
        model = Model(inputs=input, outputs=pred_logits)

        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      #sample_weight_mode='temporal',
                      metrics=[
                          'accuracy',
                          get_col_accuracy_fn(0),
                          get_col_accuracy_fn(1),
                          get_col_accuracy_fn(2),
                      ])

    def _gen_data(self, h5_file, shuffle, exclude_err_seqs=False, sample_intergenic=False):
        n_seq = h5_file['/data/X'].shape[0]
        if exclude_err_seqs:
            err_samples = np.array(h5_file['/data/err_samples'])
        if sample_intergenic and self.intergenic_chance < 1.0:
            fully_intergenic_samples = np.array(h5_file['/data/fully_intergenic_samples'])
            intergenic_rolls = np.random.random((n_seq,))  # a little bit too much, but simpler so
        X, y = [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for i in seq_indexes:
                if exclude_err_seqs and err_samples[i]:
                    continue
                if (sample_intergenic and self.intergenic_chance < 1.0
                        and fully_intergenic_samples[i]
                        and intergenic_rolls[i] > self.intergenic_chance):
                    continue
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                # apply intergenic sample weight value
                if len(X) == self.batch_size:
                    yield (
                        np.stack(X, axis=0),
                        np.stack(y, axis=0)
                    )
                    X, y = [], []


if __name__ == '__main__':
    model = InceptionModel()
    model.run()
