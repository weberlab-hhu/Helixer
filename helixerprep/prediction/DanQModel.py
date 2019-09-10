#! /usr/bin/env python3
import random
import numpy as np

from keras_layer_normalization import LayerNormalization
from keras.models import Sequential, Model
from keras.layers import (Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional, MaxPooling1D, Dropout, Reshape,
                          Activation, concatenate, Input)
from HelixerModel import HelixerModel, HelixerSequence, acc_ig_oh, acc_g_oh


class DanQSequence(HelixerSequence):
    def __getitem__(self, idx):
        pool_size = self.model.pool_size
        usable_idx_slice = self.usable_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        usable_idx_slice = sorted(list(usable_idx_slice))  # got to always provide a sorted list of idx
        X = np.stack(self.x_dset[usable_idx_slice])
        y = np.stack(self.y_dset[usable_idx_slice])

        if pool_size > 1:
            if y.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = y.shape[1] % pool_size
                X = X[:, :-overhang]
                y = y[:, :-overhang]

            if self.additional_input:
                # copy of the input so the LSTM can attend to the raw input sequence after pooling
                # first merge last 2 axis so we can split axis 1 with reshape
                X_add = np.copy(X)
                X_add = X_add.reshape((X_add.shape[0], -1))
                X_add = X_add.reshape((X_add.shape[0], X_add.shape[1] // (pool_size * 4), -1))

            # make labels 2d so we can use the standard softmax / loss functions
            y = y.reshape((
                y.shape[0],
                y.shape[1] // pool_size,
                pool_size,
                y.shape[-1],
            ))

        if self.meta_losses:
            gc = np.stack(self.gc_contents[usable_idx_slice])
            gc = np.repeat(gc[:, None], y.shape[1], axis=1)  # repeat for every time step
            lengths = np.stack(self.coord_lengths[usable_idx_slice])
            lengths = np.repeat(lengths[:, None], y.shape[1], axis=1)
            meta = np.stack([gc, lengths], axis=2)
            if self.additional_input:
                return [X, X_add], [y, meta]
            else:
                return X, [y, meta]
        else:
            if self.additional_input:
                return [X, X_add], y
            else:
                return X, y


class DanQModel(HelixerModel):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=32)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=32)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr1', '--dropout1', type=float, default=0.0)
        self.parser.add_argument('-dr2', '--dropout2', type=float, default=0.0)
        self.parser.add_argument('-mlw', '--meta-loss-weight', type=float, default=5.0)
        self.parser.add_argument('-ln', '--layer-normalization', action='store_true')
        self.parse_args()

        if not self.exclude_errors:
            print('\nRunning DanQ without --exclude-errors. This should only be done in test mode.')

    def sequence_cls(self):
        return DanQSequence

    def model(self):
        overhang = self.shape_train[1] % self.pool_size
        main_input = Input(shape=(self.shape_train[1] - overhang, 4), dtype=self.float_precision,
                           name='main_input')
        x = Conv1D(filters=self.filter_depth,
                   kernel_size=self.kernel_size,
                   padding="same",
                   activation="relu")(main_input)

        if self.pool_size > 1:
            x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.layer_normalization:
            x = LayerNormalization()(x)
        x = Dropout(self.dropout1)(x)

        if self.additional_input:
            len_after_pooling = self.shape_train[1] // self.pool_size
            add_input = Input(shape=(len_after_pooling, 4 * self.pool_size),
                              dtype=self.float_precision,
                              name='add_input')
            # add additional input to output  of
            x = concatenate([x, add_input])

        x = Bidirectional(CuDNNLSTM(self.units, return_sequences=True))(x)
        x = Dropout(self.dropout2)(x)

        if self.meta_losses:
            meta_output = Dense(2, activation='sigmoid', name='meta_output')(x)

        x = Dense(self.pool_size * self.label_dim)(x)
        x = Reshape((-1, self.pool_size, self.label_dim))(x)
        x = Activation('softmax', name='main_output')(x)

        inputs = [main_input, add_input] if self.additional_input else main_input
        outputs = [x, meta_output] if self.meta_losses else [x]
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, model):
        if self.meta_losses:
            losses = ['categorical_crossentropy', 'mean_squared_error']
            loss_weights = [1.0, self.meta_loss_weight]
            metrics = {
                'main_output': ['accuracy', acc_g_oh, acc_ig_oh],
            }
        else:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]
            metrics = ['accuracy', acc_g_oh, acc_ig_oh]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal',
                      metrics=metrics)


if __name__ == '__main__':
    model = DanQModel()
    model.run()
