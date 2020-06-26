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
        # assert y.shape[1] % pool_size == 0, 'pooling size has to evenly divide seq len'

        X = X.reshape((
            X.shape[0],
            X.shape[1] // pool_size,
            -1
        ))
        # make labels 2d so we can use the standard softmax / loss functions
        y = y.reshape((
            2,
            y.shape[1],
            y.shape[2] // pool_size,
            pool_size,
            y.shape[-1],
        ))

        # mark any multi-base timestep as error if any base has an error
        sw = sw.reshape((2, sw.shape[1], -1, pool_size))
        sw = np.logical_not(np.any(sw == 0, axis=-1)).astype(np.float32)

        # only change sample weights during training (not even validation) as we don't calculate
        # a validation loss at the moment
        if self.mode == 'train':
            if self.class_weights is not None:
                # class weights are only used during training and validation to keep the loss
                # comparable and are additive for the individual timestep predictions
                # giving even more weight to transition points
                # class weights without pooling not supported yet
                # cw = np.array([0.8, 1.4, 1.2, 1.2], dtype=np.float32)
                cls_arrays = [np.any((y[:, :, :, :, col] == 1), axis=3) for col in range(4)]
                cls_arrays = np.stack(cls_arrays, axis=3).astype(np.int8)
                # add class weights to applicable timesteps
                cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:3] + (1,)))
                cw = np.sum(cw_arrays, axis=3)
                # multiply with previous sample weights
                sw = np.multiply(sw, cw)

            if self.gene_lengths:
                gene_lengths = gene_lengths.reshape((gene_lengths.shape[0], -1, pool_size))
                gene_lengths = np.max(gene_lengths, axis=-1)  # take the maximum per pool_size (block)
                # scale gene_length to a sample weight that is 1 for the average
                gene_idx = np.where(gene_lengths)
                ig_idx = np.where(gene_lengths == 0)
                gene_weights = gene_lengths.astype(np.float32)
                scaled_gene_lengths = self.gene_lengths_average / gene_lengths[gene_idx]
                # the exponent controls the steepness of the curve
                scaled_gene_lengths = np.power(scaled_gene_lengths, self.gene_lengths_exponent)
                scaled_gene_lengths = np.clip(scaled_gene_lengths, 0.1, self.gene_lengths_cutoff)
                gene_weights[gene_idx] = scaled_gene_lengths.astype(np.float32)
                # important to set all intergenic weight to 1
                gene_weights[ig_idx] = 1.0
                sw = np.multiply(gene_weights, sw)

            if self.transition_weights is not None:
                transitions = transitions.reshape((
                    transitions.shape[0],
                    transitions.shape[1] // pool_size,
                    pool_size,
                    transitions.shape[-1],
                ))
                # more reshaping and summing  up transition weights for multiplying with sample weights
                sw_t = self.compress_tw(transitions)
                sw = np.multiply(sw_t, sw)

            if self.coverage:
                coverage_scores = coverage_scores.reshape((coverage_scores.shape[0], -1, pool_size))
                zero_positions = np.where(coverage_scores == 0)
                # scale coverage scores [0,1] by adding small numbers, default = 0.1
                # fairly good positions don't lose importance
                coverage_scores = np.add(coverage_scores, self.coverage_scaling)
                coverage_scores[zero_positions[0], zero_positions[1], zero_positions[2]] = 0
                coverage_scores = np.sum(coverage_scores, axis=2)
                # average scores according to pool_size
                coverage_scores = np.divide(coverage_scores, pool_size).astype(np.float32)
                sw = np.multiply(coverage_scores, sw)

            if self.error_weights:
                # finish by multiplying the sample_weights with the error rate
                # 1 - error_rate^(1/3) seems to have the shape we need for the weights
                # given the error rate
                error_weights = 1 - np.power(error_rates, 1/3)
                sw *= np.expand_dims(error_weights, axis=1)

            # split y and sw for the two outputs
            y_split = self._split_and_squeeze(y)
            sw_split = self._split_and_squeeze(sw)

        return X, y_split, sw_split

    def compress_tw(self, transitions):
        return self._squish_tw_to_sw(transitions, self.transition_weights, self.stretch_transition_weights)

    @staticmethod
    def _split_and_squeeze(arr):
        arr = np.split(arr, 2, axis=0)
        arr = [np.squeeze(sub_arr) for sub_arr in arr]
        return arr

    @staticmethod
    def _squish_tw_to_sw(transitions, tw, stretch):
        sw_t = [np.any((transitions[:, :, :, col] == 1),axis=2) for col in range(6)]
        sw_t = np.stack(sw_t, axis=2).astype(np.int8)
        sw_t = np.multiply(sw_t, tw)

        sw_t = np.sum(sw_t, axis=2)
        where_are_ones = np.where(sw_t == 0)
        sw_t[where_are_ones[0], where_are_ones[1]] = 1
        if stretch is not 0:
            sw_t = LSTMSequence._expand_rf(sw_t, stretch)
        return sw_t

    @staticmethod
    def _expand_rf(reshaped_sw_t, rf):

        reshaped_sw_t = np.array(reshaped_sw_t)
        dilated_rf = np.ones(np.shape(reshaped_sw_t))

        where = np.where(reshaped_sw_t > 1)
        i = np.array(where[0]) # i unverändert
        j = np.array(where[1]) # j +/- step

        #find dividers depending on the size of the dilated rf
        dividers = []
        for distance in range(1,rf+1):
            dividers.append(2**distance)

        for z in range(rf,0,-1):
            dilated_rf[i,np.maximum(np.subtract(j,z), 0)] = np.maximum(reshaped_sw_t[i,j]/dividers[z-1],1)
            dilated_rf[i,np.minimum(np.add(j,z),len(dilated_rf[0])-1)] = np.maximum(reshaped_sw_t[i,j]/dividers[z-1],1)
        dilated_rf[i,j] = np.maximum(reshaped_sw_t[i,j],1)
        return dilated_rf

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

    def _split_layer(self, x):
       x1, x2 = tf.split(x, 2, axis=1)
       return [x1, x2]

    def _split_layer_output_shape(self, input_shapes):
        return [(None, self.pool_size, 4)] * 2

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

        x = Dense(self.pool_size * 4 * 2)(x)
        x1, x2 = Lambda(lambda x: tf.split(x, 2, axis=1))(x)
        x1 = Reshape((-1, self.pool_size, 4))(x1)
        x2 = Reshape((-1, self.pool_size, 4))(x2)
        x1 = Activation('softmax', name='y_forward')(x1)
        x2 = Activation('softmax', name='y_backwards')(x2)

        model = Model(inputs=main_input, outputs=[x1, x2])
        return model

    def compile_model(self, model):
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        run_metadata = tf.RunMetadata()
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      loss_weights=[0.5, 0.5],
                      sample_weight_mode='temporal',
                      options=run_options,
                      run_metadata=run_metadata)


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
