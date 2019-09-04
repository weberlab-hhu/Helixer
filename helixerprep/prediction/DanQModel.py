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
        sw = np.ones((y.shape[0], y.shape[1] // pool_size), dtype=np.int8)

        if pool_size > 1:
            if y.shape[1] % pool_size != 0:
                # add additional values and mask them so everything divides evenly
                overhang = pool_size - (y.shape[1] % pool_size)
                y = np.pad(y, ((0, 0), (0, overhang), (0, 0)), 'constant',
                           constant_values=(0, 0))
                sw = np.pad(sw, ((0, 0), (0, 1)), 'constant', constant_values=(0, 0))
            # make labels 2d so we can use the standard softmax / loss functions
            y = y.reshape((
                y.shape[0],
                y.shape[1] // pool_size,
                pool_size,
                y.shape[-1],
            ))

        if self.add_meta_losses:
            gc = np.stack(self.gc_contents[usable_idx_slice])
            gc = np.repeat(gc[:, None], y.shape[1], axis=1)  # repeat for every time step
            lengths = np.stack(self.coord_lengths[usable_idx_slice])
            lengths = np.repeat(lengths[:, None], y.shape[1], axis=1)
            meta = np.stack([gc, lengths], axis=2)
            return X, [y, meta], [sw, sw]
        else:
            return X, y, sw


class DanQModel(HelixerModel):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=128)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=128)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr1', '--dropout1', type=float, default=0.0)
        self.parser.add_argument('-dr2', '--dropout2', type=float, default=0.0)
        self.parser.add_argument('-mlw', '--meta-loss-weight', type=float, default=5.0)
        self.parser.add_argument('-ln', '--layer-normalization', action='store_true')
        self.parser.add_argument('-meta-output', '--add-meta-losses', action='store_true')
        self.parse_args()

        if not self.exclude_errors:
            print('\nRunning DanQ without --exclude-errors. This should only be done in test mode.')

    def sequence_cls(self):
        return DanQSequence

    def model(self):
        main_input = Input(shape=(self.shape_train[1], 4), dtype=self.float_precision, name='main_input')
        x = Conv1D(filters=self.filter_depth,
                   kernel_size=self.kernel_size,
                   padding="same",
                   activation="relu")(main_input)

        if self.pool_size > 1:
            x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.layer_normalization:
            x = LayerNormalization()(x)

        x = Dropout(self.dropout1)(x)
        x = Bidirectional(CuDNNLSTM(self.units, return_sequences=True))(x)
        x = Dropout(self.dropout2)(x)

        if self.add_meta_losses:
            meta_output = Dense(2, activation='sigmoid', name='meta_output')(x)

        x = Dense(self.pool_size * self.label_dim)(x)
        x = Reshape((-1, self.pool_size, self.label_dim))(x)
        x = Activation('softmax', name='main_output')(x)

        outputs = [x, meta_output] if self.add_meta_losses else [x]
        model = Model(inputs=[main_input], outputs=outputs)
        return model

    def compile_model(self, model):
        if self.add_meta_losses:
            losses = ['categorical_crossentropy', 'mean_squared_error']
            loss_weights = [1.0, self.meta_loss_weight]
            metrics = {
                'main_output': ['accuracy', acc_g_oh, acc_ig_oh],
            }
        else:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]
            metrics=['accuracy', acc_g_oh, acc_ig_oh]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal',
                      metrics=metrics)


if __name__ == '__main__':
    model = DanQModel()
    model.run()
