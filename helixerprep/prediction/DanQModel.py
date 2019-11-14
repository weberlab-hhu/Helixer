#! /usr/bin/env python3
import random
import numpy as np

from keras_layer_normalization import LayerNormalization
from keras.models import Sequential, Model
from keras.layers import (Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional, MaxPooling1D, Dropout, Reshape,
                          Activation, concatenate, Input, BatchNormalization)
from HelixerModel import HelixerModel, HelixerSequence


class DanQSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, shuffle):
        super().__init__(model, h5_file, mode, shuffle)
        if self.class_weights is not None:
            assert not mode == 'test'  # only use class weights during training and validation

    def __getitem__(self, idx):
        pool_size = self.model.pool_size
        usable_idx_slice = self.usable_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        usable_idx_slice = sorted(list(usable_idx_slice))  # got to always provide a sorted list of idx
        X = np.stack(self.x_dset[usable_idx_slice])
        y = np.stack(self.y_dset[usable_idx_slice])
        sw = np.stack(self.sw_dset[usable_idx_slice])
        transitions = np.stack(self.transitions_dset[usable_idx_slice])

        if pool_size > 1:
            if y.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = y.shape[1] % pool_size
                X = X[:, :-overhang]
                y = y[:, :-overhang]
                sw = sw[:, :-overhang]
                transitions = transitions[:, :-overhang]

            # make labels 2d so we can use the standard softmax / loss functions
            y = y.reshape((
                y.shape[0],
                y.shape[1] // pool_size,
                pool_size,
                y.shape[-1],
            ))

            transitions = transitions.reshape((
                transitions.shape[0],
                transitions.shape[1] // pool_size,
                pool_size,
                transitions.shape[-1],
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
            if self.transitions is not None:
                sw_t= [np.any((transitions[:, :, :, col] == 1), axis=2) for col in range(6)]
                sw_t = np.stack(sw_t, axis = 2).astype(np.int8)
                sw_t = np.multiply(sw_t, self.transitions)

                sw_t = np.sum(sw_t, axis = 2)
                where_are_ones = np.where(sw_t == 0)
                sw_t[where_are_ones[0], where_are_ones[1]] = 1
                sw = np.multiply(sw_t, sw)


        # put together returned inputs/outputs
        if self.meta_losses:
            gc = np.stack(self.gc_contents[usable_idx_slice])
            gc = np.repeat(gc[:, None], y.shape[1], axis=1)  # repeat for every time step
            lengths = np.stack(self.coord_lengths[usable_idx_slice])
            lengths = np.repeat(lengths[:, None], y.shape[1], axis=1)
            meta = np.stack([gc, lengths], axis=2)
            labels = [y, meta]
            sample_weights = [sw, sw]
        else:
            labels = y
            sample_weights = sw

        return X, labels, sample_weights


class DanQModel(HelixerModel):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=32)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=32)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parser.add_argument('-cl', '--cnn-layers', type=int, default=1)
        self.parser.add_argument('-ps', '--pool-size', type=int, default=10)
        self.parser.add_argument('-dr1', '--dropout1', type=float, default=0.0)
        self.parser.add_argument('-dr2', '--dropout2', type=float, default=0.0)
        self.parser.add_argument('-ln', '--layer-normalization', action='store_true')
        self.parse_args()

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

        # if there are additional CNN layers
        for _ in range(self.cnn_layers - 1):
            x = BatchNormalization()(x)
            x = Conv1D(filters=self.filter_depth,
                       kernel_size=self.kernel_size,
                       padding="same",
                       activation="relu")(x)

        if self.pool_size > 1:
            x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.layer_normalization:
            x = LayerNormalization()(x)
        x = Dropout(self.dropout1)(x)

        x = Bidirectional(CuDNNLSTM(self.units, return_sequences=True))(x)
        x = Dropout(self.dropout2)(x)

        if self.meta_losses:
            meta_output = Dense(2, activation='sigmoid', name='meta')(x)

        x = Dense(self.pool_size * 4)(x)
        x = Reshape((-1, self.pool_size, 4))(x)
        x = Activation('softmax', name='main')(x)

        outputs = [x, meta_output] if self.meta_losses else [x]
        model = Model(inputs=main_input, outputs=outputs)
        return model

    def compile_model(self, model):
        if self.meta_losses:
            meta_loss_weight = 2.0 if self.class_weights else 5.0  # adjust loss weight to class weights
            losses = ['categorical_crossentropy', 'mean_squared_error']
            loss_weights = [1.0, meta_loss_weight]
        else:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal')


if __name__ == '__main__':
    model = DanQModel()
    model.run()
