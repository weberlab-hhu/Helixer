from abc import ABC, abstractmethod
import os
import sys
try:
    import nni
except ImportError:
    pass
import time
import h5py
import random
import argparse
import datetime
import importlib
import subprocess
import numpy as np
import tensorflow as tf
from pprint import pprint
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import timeline

from keras_layer_normalization import LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback
from keras import optimizers
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model, Sequence

from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix


class SaveEveryEpoch(Callback):
    def __init__(self, output_dir):
        super(SaveEveryEpoch, self).__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, _):
        path = os.path.join(self.output_dir, f'model{epoch}.h5')
        self.model.save(path)
        print(f'saved model at {path}')

class ConfusionMatrixTrain(Callback):
    def __init__(self, save_model_path, val_generator, canary_generator=None, report_to_nni=False):
        self.save_model_path = save_model_path
        self.val_generator = val_generator
        self.canary_generator = canary_generator
        self.report_to_nni = report_to_nni
        self.best_val_genic_f1 = 0.0
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        val_genic_f1 = HelixerModel.run_confusion_matrix(self.val_generator, self.model)
        if self.canary_generator:
            print('canary cm:')
            _ = HelixerModel.run_confusion_matrix(self.canary_generator, self.model)
        if self.report_to_nni:
            nni.report_intermediate_result(val_genic_f1)
        if val_genic_f1 > self.best_val_genic_f1:
            self.best_val_genic_f1 = val_genic_f1
            self.model.save(self.save_model_path)
            print('saved new best model with genic f1 of {} at {}'.format(self.best_val_genic_f1,
                                                                          self.save_model_path))
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            # hard-coded patience of 2 for now
            if self.epochs_without_improvement > 1:
                self.model.stop_training = True
        # hard-coded check of genic f1 of 0.5 at epoch 10
        if epoch == 10 and val_genic_f1 < 0.5:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.report_to_nni:
            nni.report_final_result(self.best_val_genic_f1)


class HelixerSequence(Sequence):
    def __init__(self, model, h5_file, mode, shuffle, filter_by_score=False, filter_quantile=0.05):
        assert mode in ['train', 'val', 'test']
        self.model = model
        self.h5_file = h5_file
        self.mode = mode
        self._cp_into_namespace(['batch_size', 'float_precision', 'class_weights', 'augment',
                                 'transition_weights','stretch_transition_weights','coverage','coverage_scaling',
                                 'overlap', 'overlap_offset', 'core_length', 'min_seqs_for_overlapping',
                                 'debug', 'exclude_errors', 'error_weights', 'gene_lengths',
                                 'gene_lengths_average', 'gene_lengths_exponent', 'gene_lengths_cutoff'])
        self.x_dset = h5_file['/data/X']
        self.y_dset = h5_file['/data/y']
        self.sw_dset = h5_file['/data/sample_weights']
        self.seqids_dset = h5_file['/data/seqids']
        if self.mode == 'train':
            if self.transition_weights is not None:
                self.transitions_dset = h5_file['/data/transitions']
            if self.coverage:
                self.coverage_dset = h5_file['/scores/by_bp']
            if self.gene_lengths:
                self.gene_lengths_dset = h5_file['/data/gene_lengths']
        self.chunk_size = self.y_dset.shape[1]

        # set array of usable indexes, always exclude all erroneous sequences during training
        if self.exclude_errors:
            self.usable_idx = np.flatnonzero(np.array(h5_file['/data/err_samples']) == False)
        else:
            self.usable_idx = list(range(self.x_dset.shape[0]))

        print('total chunks: {}'.format(len(self.usable_idx)))
        if filter_by_score:
            self._filter_usable_idx_by_score(quantile=filter_quantile)
            print('total filtered chunks: {}'.format(len(self.usable_idx)))

        if shuffle:
            random.shuffle(self.usable_idx)

    def _filter_usable_idx_by_score(self, quantile=0.05, score_dataset="scores/one_centered", hard=True, prob=0.5):
        """filters or down samples data by a evidence derived score for the reference annotation in any chunk"""
        scores = self.h5_file[score_dataset][:]
        threshold = np.quantile(scores, quantile)

        self.usable_idx = np.where(self.h5_file[score_dataset][self.usable_idx] > threshold)[0]
        if not hard:
            to_review = np.where(self.h5_file[score_dataset][self.usable_idx] < threshold)[0]
            n_to_keep = int(to_review.size * prob)
            to_keep = np.random.choice(to_review, n_to_keep, replace=False)
            self.usable_idx = np.sort(np.concatenate((self.usable_idx, to_keep)))

    def _cp_into_namespace(self, names):
        """Moves class properties from self.model into this class for brevity"""
        for name in names:
            self.__dict__[name] = self.model.__dict__[name]

    def _usable_idx_batch(self, idx):
        n_seqs = self._seqs_per_batch()
        usable_idx_slice = self.usable_idx[idx * n_seqs:(idx + 1) * n_seqs]
        usable_idx_slice = sorted(list(usable_idx_slice))  # got to always provide a sorted list of idx
        return usable_idx_slice

    def _get_batch_data(self, idx):
        usable_idx_batch = self._usable_idx_batch(idx)
        if self.overlap:
            X = self.x_dset[usable_idx_batch]
            seqid_borders = self._get_seqid_borders(idx)
            # split data along these borders
            X_by_seqid = np.array_split(X, seqid_borders)
            overlapping_X = []
            for seqid_x in X_by_seqid:
                if len(seqid_x) >= self.min_seqs_for_overlapping:
                    seq = np.concatenate(seqid_x, axis=0)
                    # apply sliding window
                    overlapping_X += [seq[i:i+self.chunk_size]
                                      for i in range(0, len(seq) - self.chunk_size + 1,
                                                     self.overlap_offset)]
                else:
                    # do not overlap short sequences
                    overlapping_X += [seqid_x[i] for i in range(len(seqid_x))]
            X = np.stack(overlapping_X)
        else:
            X = self.x_dset[usable_idx_batch]

        y = self.y_dset[usable_idx_batch]
        sw = self.sw_dset[usable_idx_batch]

        # calculate base level error rate for each sequence
        error_rates = (np.count_nonzero(sw == 0, axis=1) / y.shape[1]).astype(np.float32)

        if self.mode == 'train' and self.transition_weights is not None:
            transitions = self.transitions_dset[:, usable_idx_batch]
        else:
            transitions = None
        if self. mode == 'train' and self.coverage:
            coverage_scores = self.coverage_dset[:, usable_idx_batch]
        else:
            coverage_scores = None
        if self.mode == 'train' and self.gene_lengths:
            gene_lengths = self.gene_lengths_dset[:, usable_idx_batch]
        else:
            gene_lengths = None

        return X, y, sw, error_rates, gene_lengths, transitions, coverage_scores

    def _get_seqid_borders(self, idx):
        seqids = self.seqids_dset[self._usable_idx_batch(idx)]
        idx_border = np.argwhere(seqids[:-1] != seqids[1:])[:, 0]
        if len(idx_border) > 0:
            # if there are changes in seqid
            idx_border = np.add(idx_border, 1)  # add 1 for splitting with np.split()
        return idx_border

    def _get_seqids_for_batch(self, idx):
        usable_idx_batch = self._usable_idx_batch(idx)
        seqids = self.seqids_dset[usable_idx_batch]
        return seqids

    def _seqs_per_batch(self, batch_idx=None):
        """Calculates how many original sequences are needed to fill a batch. Necessary
        if --overlap is on"""
        if self.overlap:
            n_seqs = self.batch_size / (self.chunk_size / self.overlap_offset)
        else:
            n_seqs = self.batch_size
        if batch_idx and batch_idx == len(self) - 1 and len(self.usable_idx) % n_seqs > 0:
            n_seqs = len(self.usable_idx) % n_seqs  # calculate overhang when at the end
        return int(n_seqs)

    def _augment(self, X, y, sw):
        def flip_strands(arr):
            assert arr.shape[0] == 2, 'Does not appear to be double stranded'
            arr = np.flip(arr, axis=2)  # reverse sequence order
            arr[0], arr[1] = arr[0], arr[1].copy()  # exchange strands
            return arr

        flip = np.random.rand(y.shape[1]) < 0.5
        X[flip] = np.flip(X[flip], axis=(1, 2))  # reverse complement
        y[:, flip] = flip_strands(y[:, flip])
        sw[:, flip] = flip_strands(sw[:, flip])
        return X, y, sw

    def __len__(self):
        if self.debug:
            return 1
        else:
            return int(np.ceil(len(self.usable_idx) / self._seqs_per_batch()))

    @abstractmethod
    def __getitem__(self, idx):
        pass


class HelixerModel(ABC):
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data-dir', type=str, default='')
        self.parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')
        # training params
        self.parser.add_argument('-e', '--epochs', type=int, default=10000)
        self.parser.add_argument('-bs', '--batch-size', type=int, default=8)
        self.parser.add_argument('-loss', '--loss', type=str, default='')
        self.parser.add_argument('-cn', '--clip-norm', type=float, default=1.0)
        self.parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
        self.parser.add_argument('-cw', '--class-weights', type=str, default='None')
        self.parser.add_argument('-tw', '--transition-weights', type=str, default='None')
        self.parser.add_argument('-s-tw', '--stretch-transition-weights', type=int, default=0)
        self.parser.add_argument('-cov','--coverage',action='store_true')
        self.parser.add_argument('-covs','--coverage-scaling', type=float, default=0.1)
        self.parser.add_argument('-can', '--canary-dataset', type=str, default='')
        self.parser.add_argument('-aug', '--augment', action='store_true')
        self.parser.add_argument('-res', '--resume-training', action='store_true')
        self.parser.add_argument('-ee', '--exclude-errors', action='store_true')
        self.parser.add_argument('-ew', '--error-weights', action='store_true')
        self.parser.add_argument('-qu', '--quantile-filter', type=float)
        # testing
        self.parser.add_argument('-lm', '--load-model-path', type=str, default='')
        self.parser.add_argument('-td', '--test-data', type=str, default='')
        self.parser.add_argument('-po', '--prediction-output-path', type=str, default='predictions.h5')
        self.parser.add_argument('-ev', '--eval', action='store_true')
        # overlap options
        self.parser.add_argument('-overlap', '--overlap', action='store_true')
        self.parser.add_argument('-overlap-offset', '--overlap-offset', type=int, default=2500)
        self.parser.add_argument('-core-len', '--core-length', type=int, default=10000)
        self.parser.add_argument('-min-seqs', '--min-seqs-for-overlapping', type=int, default=3)
        # gene length adjustments
        self.parser.add_argument('-gl', '--gene-lengths', action='store_true')
        self.parser.add_argument('-glavg', '--gene-lengths-average', type=int, default=3350)
        self.parser.add_argument('-glexp', '--gene-lengths-exponent', type=float, default=1.0)
        self.parser.add_argument('-glco', '--gene-lengths-cutoff', type=float, default=5.0)
        # resources
        self.parser.add_argument('-fp', '--float-precision', type=str, default='float32')
        self.parser.add_argument('-gpus', '--gpus', type=int, default=1)
        self.parser.add_argument('-cpus', '--cpus', type=int, default=8)
        self.parser.add_argument('-gpuid', '--gpu-id', type=int, default=-1)
        # misc flags
        self.parser.add_argument('-see', '--save-every-epoch', action='store_true')
        self.parser.add_argument('-nni', '--nni', action='store_true')
        self.parser.add_argument('-trace', '--trace', action='store_true')
        self.parser.add_argument('-v', '--verbose', action='store_true')
        self.parser.add_argument('-db', '--debug', action='store_true')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        self.__dict__.update(args)
        self.testing = bool(self.load_model_path and not self.resume_training)
        assert not (self.testing and self.data_dir)
        assert not (not self.testing and self.test_data)
        assert not (self.resume_training and (not self.load_model_path or not self.data_dir))

        if self.nni:
            hyperopt_args = nni.get_next_parameter()
            assert all([key in args for key in hyperopt_args.keys()]), 'Unknown nni parameter'
            self.__dict__.update(hyperopt_args)
            nni_save_model_path = os.path.expandvars('$NNI_OUTPUT_DIR/best_model.h5')
            nni_pred_output_path = os.path.expandvars('$NNI_OUTPUT_DIR/predictions.h5')
            self.__dict__['save_model_path'] = nni_save_model_path
            self.__dict__['prediction_output_path'] = nni_pred_output_path
            args.update(hyperopt_args)
            # for the print out
            args['save_model_path'] = nni_save_model_path
            args['prediction_output_path'] = nni_pred_output_path

        self.class_weights = eval(self.class_weights)
        if type(self.class_weights) is list:
            self.class_weights = np.array(self.class_weights, dtype=np.float32)

        self.transition_weights = eval(self.transition_weights)
        if type(self.transition_weights) is list:
            self.transition_weights = np.array(self.transition_weights, dtype = np.float32)

        if self.verbose:
            print()
            pprint(args)

    def generate_callbacks(self):
        canary_gen = self.gen_canary_data() if self.canary_dataset else None
        callbacks = [ConfusionMatrixTrain(self.save_model_path, self.gen_validation_data(),
                                          canary_gen, report_to_nni=self.nni)]
        if self.save_every_epoch:
            callbacks.append(SaveEveryEpoch(os.path.dirname(self.save_model_path)))
        return callbacks

    def set_resources(self):
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        K.set_floatx(self.float_precision)
        if self.gpu_id > -1:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID';
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

    def gen_training_data(self):
        SequenceCls = self.sequence_cls()
        if self.quantile_filter is None:
            filter_by_score = False
        else:
            filter_by_score = True
        return SequenceCls(model=self,
                           h5_file=self.h5_train,
                           mode='train',
                           shuffle=True,
                           filter_by_score=filter_by_score,
                           filter_quantile=self.quantile_filter)

    def gen_validation_data(self):
        SequenceCls = self.sequence_cls()
        return SequenceCls(model=self,
                           h5_file=self.h5_val,
                           mode='val',
                           shuffle=False)

    def gen_test_data(self):
        SequenceCls = self.sequence_cls()
        return SequenceCls(model=self,
                           h5_file=self.h5_test,
                           mode='test',
                           shuffle=False)

    def gen_canary_data(self):
        SequenceCls = self.sequence_cls()
        return SequenceCls(model=self,
                           h5_file=self.h5_canary,
                           mode='val',
                           shuffle=False)

    @staticmethod
    def run_confusion_matrix(generator, model):
        start = time.time()
        cm_calculator = ConfusionMatrix(generator)
        genic_f1 = cm_calculator.calculate_cm(model)
        if np.isnan(genic_f1):
            genic_f1 = 0.0
        print('\ncm calculation took: {:.2f} minutes\n'.format(int(time.time() - start) / 60))
        return genic_f1

    @abstractmethod
    def sequence_cls(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    def plot_model(self, model):
        from keras.utils import plot_model
        plot_model(model, to_file='model.png')
        print('Plotted to model.png')
        sys.exit()

    def open_data_files(self):
        def get_n_correct_seqs(h5_file):
            err_samples = np.array(h5_file['/data/err_samples'])
            n_correct = np.count_nonzero(err_samples == False)
            if n_correct == 0:
                print('WARNING: no fully correct sample found')
            return n_correct

        def get_n_intergenic_seqs(h5_file):
            ic_samples = np.array(h5_file['/data/fully_intergenic_samples'])
            n_fully_ig = np.count_nonzero(ic_samples == True)
            if n_fully_ig == 0:
                print('WARNING: no fully intergenic samples found')
            return n_fully_ig

        if not self.testing:
            self.h5_train = h5py.File(os.path.join(self.data_dir, 'training_data.h5'), 'r')
            self.h5_val = h5py.File(os.path.join(self.data_dir, 'validation_data.h5'), 'r')
            self.shape_train = self.h5_train['/data/X'].shape
            self.shape_val = self.h5_val['/data/X'].shape

            n_train_correct_seqs = get_n_correct_seqs(self.h5_train)
            n_val_correct_seqs = get_n_correct_seqs(self.h5_val)

            if self.exclude_errors:
                n_train_seqs = n_train_correct_seqs
            else:
                n_train_seqs = self.shape_train[0]
            n_val_seqs = self.shape_val[0]  # always validate on all

            n_intergenic_train_seqs = get_n_intergenic_seqs(self.h5_train)
            n_intergenic_val_seqs = get_n_intergenic_seqs(self.h5_val)
        else:
            self.h5_test = h5py.File(self.test_data, 'r')
            self.shape_test = self.h5_test['/data/X'].shape

            n_test_correct_seqs = get_n_correct_seqs(self.h5_test)
            n_test_seqs_with_intergenic = self.shape_test[0]
            n_intergenic_test_seqs = get_n_intergenic_seqs(self.h5_test)

        if self.canary_dataset:
            self.h5_canary = h5py.File(self.canary_dataset, 'r')
            print('\nCanary data config: ')
            print(dict(self.h5_canary.attrs))

        if self.verbose:
            print('\nData config: ')
            if not self.testing:
                print(dict(self.h5_train.attrs))
                print('\nTraining data shape: {}'.format(self.shape_train[:2]))
                print('Validation data shape: {}'.format(self.shape_val[:2]))
                print('\nTotal est. training sequences: {}'.format(n_train_seqs))
                print('Total est. val sequences: {}'.format(n_val_seqs))
                print('\nEst. intergenic train/val seqs: {:.2f}% / {:.2f}%'.format(
                    n_intergenic_train_seqs / n_train_seqs * 100,
                    n_intergenic_val_seqs / n_val_seqs * 100))
                print('Fully correct train/val seqs: {:.2f}% / {:.2f}%\n'.format(
                    n_train_correct_seqs / self.shape_train[0] * 100,
                    n_val_correct_seqs / self.shape_val[0] * 100))
            else:
                print(dict(self.h5_test.attrs))
                print('\nTest data shape: {}'.format(self.shape_test[:2]))
                print('\nIntergenic test seqs: {:.2f}%'.format(
                    n_intergenic_test_seqs / n_test_seqs_with_intergenic * 100))
                print('Fully correct test seqs: {:.2f}%\n'.format(
                    n_test_correct_seqs / self.shape_test[0] * 100))

    def _overlap_predictions(self, batch_idx, test_sequence, predictions):
        # some shortcut variables
        chunk_size = predictions.shape[1]
        seq_overhang = int((chunk_size - self.core_length) / 2)
        n_overhang_seqs = seq_overhang // self.overlap_offset
        n_original_seqs = test_sequence._seqs_per_batch(batch_idx=batch_idx)
        n_overlapping_seqs = self.core_length // self.overlap_offset

        all_predictions = np.empty((0, ) + predictions.shape[1:])
        seqid_borders = list(test_sequence._get_seqid_borders(batch_idx))
        # get number of sequences for each seqid from border distance
        seqid_sizes = np.diff(np.array([0] + seqid_borders + [n_original_seqs]))
        print(batch_idx, seqid_sizes)
        pred_offset = 0
        for seqid_size in seqid_sizes:
            if seqid_size > 2:
                n_seqid_seqs = (seqid_size - 1) * chunk_size // self.overlap_offset + 1
            else:
                n_seqid_seqs = seqid_size
            predictions_seqid = predictions[pred_offset:pred_offset + n_seqid_seqs]
            pred_offset += n_seqid_seqs
            if seqid_size >= self.min_seqs_for_overlapping:
                # actual overlapping; save first and last sequence for special handling later
                first, last = predictions_seqid[0], predictions_seqid[-1]
                # cut to the core
                predictions_seqid = [s[seq_overhang:-seq_overhang] for s in predictions_seqid]
                # generate zero'd out filler sequences for the start and end
                filler_seqs = [np.zeros((self.core_length, 4))] * n_overhang_seqs
                predictions_seqid = filler_seqs + predictions_seqid + filler_seqs
                # stack eveything
                predictions_seqid = np.stack(predictions_seqid).astype(predictions.dtype)
                # add overhang edge data from first/last seq that can not be overlapped
                predictions_seqid[0, :seq_overhang] = first[:seq_overhang]
                predictions_seqid[-1, -seq_overhang:] = last[-seq_overhang:]

                # merge and stack efficiently so everything can be averaged
                n_predicted_bases = seqid_size * chunk_size
                stacked = np.zeros((n_overlapping_seqs, n_predicted_bases, 4), dtype=predictions.dtype)
                for j in range(n_overlapping_seqs):
                    # get idx of every n_overlapping_seqs'th seq starting at j
                    idx = list(range(j, predictions_seqid.shape[0], n_overlapping_seqs))
                    seq = np.concatenate(predictions_seqid[idx], axis=0)
                    start_base = j * self.overlap_offset
                    stacked[j, start_base:start_base+seq.shape[0]] = seq

                # average individual softmax values
                # does change pseudo-probability dist at the edge but not the argmax afterwards
                # (causes values to be lower there)
                averages = np.mean(stacked, axis=0)
                predictions_seqid = np.stack(np.split(averages, seqid_size))
            all_predictions = np.concatenate([all_predictions, predictions_seqid], axis=0)
        assert all_predictions.shape[0] == n_original_seqs
        return all_predictions

    def _make_predictions(self, model):
        # loop through batches and continously expand output dataset as everything might
        # not fit in memory
        pred_out = h5py.File(self.prediction_output_path, 'w')
        test_sequence = self.gen_test_data()

        for i in range(len(test_sequence)):
            if self.verbose:
                print(i, '/', len(test_sequence), end='\r')
            predictions = model.predict_on_batch(test_sequence[i][0])
            # join last two dims when predicting one hot labels
            predictions = predictions.reshape(predictions.shape[:2] + (-1,))
            # reshape when predicting more than one point at a time
            label_dim = 4
            if predictions.shape[2] != label_dim:
                n_points = predictions.shape[2] // label_dim
                predictions = predictions.reshape(
                    predictions.shape[0],
                    predictions.shape[1] * n_points,
                    label_dim,
                )
                # add 0-padding if needed
                n_removed = self.shape_test[1] - predictions.shape[1]
                if n_removed > 0:
                    zero_padding = np.zeros((predictions.shape[0], n_removed, predictions.shape[2]),
                                            dtype=predictions.dtype)
                    predictions = np.concatenate((predictions, zero_padding), axis=1)

            if self.overlap and predictions.shape[0] > 1:
                predictions = self._overlap_predictions(i, test_sequence, predictions)

            # prepare h5 dataset and save the predictions to disk
            if i == 0:
                old_len = 0
                pred_out.create_dataset('/predictions',
                                        data=predictions,
                                        maxshape=(None,) + predictions.shape[1:],
                                        chunks=(1,) + predictions.shape[1:],
                                        dtype='float32',
                                        compression='lzf',
                                        shuffle=True)
            else:
                old_len = pred_out['/predictions'].shape[0]
                pred_out['/predictions'].resize(old_len + predictions.shape[0], axis=0)
            pred_out['/predictions'][old_len:] = predictions

        # add model config and other attributes to predictions
        h5_model = h5py.File(self.load_model_path, 'r')
        pred_out.attrs['model_config'] = h5_model.attrs['model_config']
        pred_out.attrs['n_bases_removed'] = n_removed
        pred_out.attrs['test_data_path'] = self.test_data
        pred_out.attrs['model_path'] = self.load_model_path
        pred_out.attrs['timestamp'] = str(datetime.datetime.now())
        pred_out.attrs['model_md5sum'] = self.loaded_model_hash
        pred_out.close()
        h5_model.close()

    def _load_helixer_model(self):
        model = load_model(self.load_model_path, custom_objects = {
            'LayerNormalization': LayerNormalization,
        })
        return model

    def _print_model_info(self, model):
        os.chdir(os.path.dirname(__file__))
        try:
            cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            branch = subprocess.check_output(cmd).strip().decode()
            cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
            commit = subprocess.check_output(cmd).strip().decode()
            print(f'Current helixerprep branch: {branch} ({commit})')
            if self.load_model_path:
                cmd = ['md5sum', self.load_model_path]
                self.loaded_model_hash = subprocess.check_output(cmd).strip().decode()
                print(f'Md5sum of the loaded model: {self.loaded_model_hash}')
        except subprocess.CalledProcessError:
            print('An error occured while running a subprocess')

        print()
        if self.verbose:
            print(model.summary())
        else:
            print('Total params: {:,}'.format(model.count_params()))

    def _trace(self, model, generator):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fn = K.function(model.inputs, model.outputs, options=run_options, run_metadata=run_metadata)
        for i in range(4):
            x = K.variable(generator[i][0], dtype='float32')
            fn([x])
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(f'timeline_{i}.json', 'w') as f:
                f.write(ctf)
                print(f'trace {i} printed')

    def run(self):
        self.set_resources()
        self.open_data_files()
        # we either train or predict
        if not self.testing:
            if self.resume_training:
                model = self._load_helixer_model()
            else:
                model = self.model()
            if self.trace:
                self._trace(model, self.gen_training_data())
                exit()
            if self.gpus >= 2:
                model = multi_gpu_model(model, gpus=self.gpus)
            self._print_model_info(model)

            self.optimizer = optimizers.Adam(lr=self.learning_rate, clipnorm=self.clip_norm)
            self.compile_model(model)

            model.fit_generator(generator=self.gen_training_data(),
                                epochs=self.epochs,
                                workers=0,  # run in main thread
                                # workers=1,
                                # validation_data=self.gen_validation_data(),
                                callbacks=self.generate_callbacks(),
                                verbose=True)
        else:
            assert self.test_data.endswith('.h5'), 'Need a h5 test data file when loading a model'
            assert self.load_model_path.endswith('.h5'), 'Need a h5 model file'
            if self.overlap:
                assert self.testing  # only use overlapping during test time
                assert self.overlap_offset < self.core_length
                # check if everything divides evenly to avoid further head aches
                assert (self.shape_test[1] / self.overlap_offset).is_integer()
                assert (self.batch_size / (self.shape_test[1] / self.overlap_offset)).is_integer()
                assert ((self.shape_test[1] - self.core_length) / 2 / self.overlap_offset).is_integer()

            model = self._load_helixer_model()
            self._print_model_info(model)

            if self.eval:
                _ = HelixerModel.run_confusion_matrix(self.gen_test_data(), model)
            else:
                if os.path.isfile(self.prediction_output_path):
                    print(f'{self.prediction_output_path} already existing and will be overridden.')
                self._make_predictions(model)
            self.h5_test.close()
