#! /usr/bin/env python3
import numpy as np
import h5py
import os
import subprocess
import tensorflow as tf
import datetime
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
        X, y, sw, transitions, phases, _, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size

        if pool_size > 1:
            if X.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = X.shape[1] % pool_size
                X = X[:, :-overhang].astype(np.int8)
                if not self.only_predictions:
                    y = y[:, :-overhang]
                    sw = sw[:, :-overhang]
                    if self.predict_phase:
                        phases = phases[:, :-overhang]
                    if self.mode == 'train' and self.transition_weights is not None:
                        transitions = transitions[:, :-overhang]

            if not self.only_predictions:
                y = y[:, :, [0, 1, 3]].astype(np.int8)
                y = self._mk_timestep_pools_class_last(y)
                if self.predict_phase:
                    #split up data
                    y_phase = phases[:, :, [1, 2, 3]].astype(np.int8)
                    y_phase = self._mk_timestep_pools_class_last(y_phase)
                    y = np.concatenate((y, y_phase), axis=3).astype(np.int8)
                sw = sw.reshape((sw.shape[0], -1, pool_size))
                sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.int8)

            if self.mode == 'train':
                if self.class_weights is not None:
                    # class weights are additive for the individual timestep predictions
                    # giving even more weight to transition points
                    # class weights without pooling not supported yet
                    # cw = np.array([1.0, 1.2, 1.0, 0.8], dtype=np.float32) 
                    cls_arrays = [np.any((y[:, :, :, col] == 1), axis=2) for col in range(6)]
                    cls_arrays = np.stack(cls_arrays, axis=2).astype(np.int8)
                    # add class weights to applicable timesteps
                    cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:2] + (1,))).astype(np.float16)
                    cw = np.sum(cw_arrays, axis=2).astype(np.float16)
                    sw = np.multiply(cw, sw).astype(np.float16)

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
                pass

        if self.only_predictions:
            return X
        else:
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
        self.parse_args()

    #method to create the predictions .h5 file
    def _make_predictions(self, model):
        # loop through batches and continuously expand output dataset as everything might
        # not fit in memory
        pred_out = h5py.File(self.prediction_output_path, 'w')
        test_sequence = self.gen_test_data()

        for batch_index in range(len(test_sequence)):
            if self.verbose:
                print(batch_index, '/', len(test_sequence), end='\r')
            if not self.only_predictions:
                input_data = test_sequence[batch_index][0]
            else:
                input_data = test_sequence[batch_index]
            predictions = model.predict_on_batch(input_data)
            if self.predict_phase:
                dtype_ = predictions.dtype
                shape_ = list(predictions.shape)
                shape_[-1] = 1
                #make phase none as 1- the sum(rest)
                phase = predictions[:, :, :, 3:]
                phase_sum = 1- np.sum(phase, axis=3).reshape(shape_)
                phase = np.concatenate((phase_sum, phase), axis=3)

                #CDS class to the max(3 phases)
                genic = predictions[:, :, :, 0:3] #creation of genic prediction array
                #array containing max values of phase predictions
                genic_CDS = np.amax(phase[:, :, :, 1:], axis=3).reshape(shape_)

                #x = [intergenic, UTR, intron]. x = x * (1 - CDS) / sum(x) for rescaling of the others
                genic = genic * (1-genic_CDS) / np.sum(genic, axis=3).reshape(shape_)
                #addition of the CDS probabilities
                genic = np.roll(genic, 1, axis=3)
                genic = np.concatenate((genic, genic_CDS), axis=3)
                genic = np.roll(genic, -1, axis=3)
                #generate final list
                predictions = [genic, phase]

            if isinstance(predictions, list):
                # when we have two outputs, one is for phase
                output_names = ['predictions', 'predictions_phase']
            else:
                # if we just had one output
                predictions = (predictions,)
                output_names = ['predictions']

            for dset_name, pred_dset in zip(output_names, predictions):
                # join last two dims when predicting one hot labels
                pred_dset = pred_dset.reshape(pred_dset.shape[:2] + (-1,))
                # reshape when predicting more than one point at a time
                label_dim = 4
                if pred_dset.shape[2] != label_dim:
                    n_points = pred_dset.shape[2] // label_dim
                    pred_dset = pred_dset.reshape(
                        pred_dset.shape[0],
                        pred_dset.shape[1] * n_points,
                        label_dim,
                    )
                    # add 0-padding if needed
                    n_removed = self.shape_test[1] - pred_dset.shape[1]
                    if n_removed > 0:
                        zero_padding = np.zeros((pred_dset.shape[0], n_removed, pred_dset.shape[2]),
                                                dtype=pred_dset.dtype)
                        pred_dset = np.concatenate((pred_dset, zero_padding), axis=1)
                else:
                    n_removed = 0  # just to avoid crashing with Unbound Local Error setting attrs for dCNN

                if self.overlap:
                    pred_dset = test_sequence.ol_helper.overlap_predictions(batch_index, pred_dset)

                # prepare h5 dataset and save the predictions to disk
                if batch_index == 0:
                    old_len = 0
                    pred_out.create_dataset(dset_name,
                                            data=pred_dset,
                                            maxshape=(None,) + pred_dset.shape[1:],
                                            chunks=(1,) + pred_dset.shape[1:],
                                            dtype='float16',
                                            compression='lzf',
                                            shuffle=True)
                else:
                    old_len = pred_out[dset_name].shape[0]
                    pred_out[dset_name].resize(old_len + pred_dset.shape[0], axis=0)
                pred_out[dset_name][old_len:] = pred_dset

        # add model config and other attributes to predictions
        h5_model = h5py.File(self.load_model_path, 'r')
        pred_out.attrs['model_config'] = h5_model.attrs['model_config']
        pred_out.attrs['n_bases_removed'] = n_removed
        pred_out.attrs['test_data_path'] = self.test_data
        pred_out.attrs['model_path'] = self.load_model_path
        pred_out.attrs['timestamp'] = str(datetime.datetime.now())
        pred_out.attrs['model_md5sum'] = self.loaded_model_hash #gave error --> method was not inherited?!
        pred_out.close()
        h5_model.close()

    def _print_model_info(self, model):
        pwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        try:
            cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            branch = subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip().decode()
            cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
            commit = subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip().decode()
            print(f'Current Helixer branch: {branch} ({commit})')
        except subprocess.CalledProcessError:
            version = pkg_resources.require('helixer')[0].version
            print(f'Current Helixer version: {version}')

        try:
            if os.path.isfile(self.load_model_path):
                cmd = ['md5sum', self.load_model_path]
                self.loaded_model_hash = subprocess.check_output(cmd).strip().decode()
                print(f'Md5sum of the loaded model: {self.loaded_model_hash}')
        except subprocess.CalledProcessError:
            print('An error occurred while running a subprocess, unable to record loaded_model_hash')
            self.loaded_model_hash = 'error'

        print()
        if self.verbose:
            print(model.summary())
        else:
            print('Total params: {:,}'.format(model.count_params()))
        os.chdir(pwd)  # return to previous directory


    @staticmethod
    def sequence_cls():
        return DanQSequence

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
            #x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.dropout1 > 0.0:
            x = Dropout(self.dropout1)(x)

        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        for _ in range(self.lstm_layers - 1):
            x = Bidirectional(LSTM(self.units, return_sequences=True))(x)

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            x = Dropout(self.dropout2)(x)

        if self.predict_phase:
            x = Dense(self.pool_size * 6)(x)
            x = Reshape((-1, self.pool_size, 6))(x)
            x = Activation('softmax', name='main')(x)
            outputs = [x]

        else:
            x = Dense(self.pool_size * 4)(x)
            x = Reshape((-1, self.pool_size, 4))(x)
            x = Activation('softmax', name='main')(x)
            outputs = [x]

        model = Model(inputs=main_input, outputs=outputs)
        return model

    def compile_model(self, model):
        if self.predict_phase:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]
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
