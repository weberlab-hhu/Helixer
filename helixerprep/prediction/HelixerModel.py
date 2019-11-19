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

from ConfusionMatrix import ConfusionMatrix


class SaveEveryEpoch(Callback):
    def __init__(self, output_dir):
        super(SaveEveryEpoch, self).__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, _):
        path = os.path.join(self.output_dir, f'model{epoch}.h5')
        self.model.save(path)
        print(f'saved model at {path}')

class ConfusionMatrixTrain(Callback):
    def __init__(self, generator, save_model_path, report_to_nni=False):
        self.generator = generator
        self.save_model_path = save_model_path
        self.report_to_nni = report_to_nni
        self.best_genic_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        genic_f1 = HelixerModel.run_confusion_matrix(self.generator, self.model)
        if self.report_to_nni:
            nni.report_intermediate_result(genic_f1)
        if genic_f1 > self.best_genic_f1:
            self.best_genic_f1 = genic_f1
            self.model.save(self.save_model_path)
            print('saved new best model with genic f1 of {} at {}'.format(self.best_genic_f1,
                                                                          self.save_model_path))

    def on_train_end(self, logs=None):
        if self.report_to_nni:
            nni.report_final_result(self.best_genic_f1)


class HelixerSequence(Sequence):
    def __init__(self, model, h5_file, mode, shuffle):
        assert mode in ['train', 'val', 'test']
        assert mode != 'test' or model.load_model_path  # assure that the mode param is correct
        self.model = model
        self.h5_file = h5_file
        self.mode = mode
        self._cp_into_namespace(['batch_size', 'float_precision', 'class_weights', 'meta_losses',
                                 'transitions', 'overlap', 'overlap_offset', 'core_length'])
        self.x_dset = h5_file['/data/X']
        self.y_dset = h5_file['/data/y']
        self.sw_dset = h5_file['/data/sample_weights']
        self.seqids_dset = h5_file['/data/seqids']
        if self.transitions is not None:
            self.transitions_dset = h5_file['data/transitions']
        self.chunk_size = self.y_dset.shape[1]
        self._load_and_scale_meta_info()
        self.debug = self.model.debug

        # set array of usable indexes, always exclude all erroneous sequences during training
        if mode == 'train':
            self.usable_idx = np.flatnonzero(np.array(h5_file['/data/err_samples']) == False)
        else:
            self.usable_idx = list(range(self.x_dset.shape[0]))
        if shuffle:
            random.shuffle(self.usable_idx)

    def _cp_into_namespace(self, names):
        """Moves class properties from self.model into this class for brevity"""
        for name in names:
            self.__dict__[name] = self.model.__dict__[name]

    def _load_and_scale_meta_info(self):
        self.gc_contents = np.array(self.h5_file['/data/gc_contents'], dtype=self.float_precision)
        self.coord_lengths = np.array(self.h5_file['/data/coord_lengths'], dtype=self.float_precision)
        # scale gc content by their coord lengths
        self.gc_contents /= self.coord_lengths
        # log transform and standardize coord_lengths to [0, 1]
        # gc_contents should have a fine scale already
        self.coord_lengths = np.log(self.coord_lengths)
        self.coord_lengths = self.coord_lengths.reshape(-1, 1)
        self.coord_lengths = MinMaxScaler().fit(self.coord_lengths).transform(self.coord_lengths)
        # need to clip as values can be slightly above 1.0 (docs say otherwise..)
        self.coord_lengths = np.clip(self.coord_lengths, 0.0, 1.0).squeeze()
        assert np.all(np.logical_and(self.gc_contents >= 0.0, self.gc_contents <= 1.0))
        assert np.all(np.logical_and(self.coord_lengths >= 0.0, self.coord_lengths <= 1.0))

    def _usable_idx_batch(self, idx):
        n_seqs = self._seqs_per_batch()
        usable_idx_slice = self.usable_idx[idx * n_seqs:(idx + 1) * n_seqs]
        usable_idx_slice = sorted(list(usable_idx_slice))  # got to always provide a sorted list of idx
        return usable_idx_slice

    def _get_batch_data(self, idx):
        usable_idx_batch = self._usable_idx_batch(idx)

        if self.overlap:
            X = self.x_dset[usable_idx_batch]
            X = np.concatenate(X, axis=0)
            # apply sliding window
            X = [X[i:i+self.chunk_size]
                 for i in range(0, len(X) - self.chunk_size + 1, self.overlap_offset)]
            X = np.stack(X)
        else:
            X = self.x_dset[usable_idx_batch]

        y = self.y_dset[usable_idx_batch]
        sw = self.sw_dset[usable_idx_batch]
        if self.transitions is not None:
            transitions = self.transitions_dset[usable_idx_batch]
        else:
            transitions = None
        return X, y, sw, transitions

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
        if batch_idx and batch_idx == len(self) - 1:
            n_seqs = len(self.usable_idx) % n_seqs  # calculate overhang when at the end
        return int(n_seqs)

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
        self.parser.add_argument('-meta-losses', '--meta-losses', action='store_true')
        self.parser.add_argument('-t', '--transitions', type=str, default='None')
        # testing
        self.parser.add_argument('-lm', '--load-model-path', type=str, default='')
        self.parser.add_argument('-td', '--test-data', type=str, default='')
        self.parser.add_argument('-po', '--prediction-output-path', type=str, default='predictions.h5')
        self.parser.add_argument('-ev', '--eval', action='store_true')
        # overlap options
        self.parser.add_argument('-overlap', '--overlap', action='store_true')
        self.parser.add_argument('-overlap-offset', '--overlap-offset', type=int, default=2500) # 2500
        self.parser.add_argument('-core-len', '--core-length', type=int, default=10000)
        # resources
        self.parser.add_argument('-fp', '--float-precision', type=str, default='float32')
        self.parser.add_argument('-gpus', '--gpus', type=int, default=1)
        self.parser.add_argument('-cpus', '--cpus', type=int, default=8)
        self.parser.add_argument('--specific-gpu-id', type=int, default=-1)
        # misc flags
        self.parser.add_argument('-see', '--save-every-epoch', action='store_true')
        self.parser.add_argument('-nni', '--nni', action='store_true')
        self.parser.add_argument('-trace', '--trace', action='store_true')
        self.parser.add_argument('-v', '--verbose', action='store_true')
        self.parser.add_argument('-db', '--debug', action='store_true')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        self.__dict__.update(args)

        self.class_weights = eval(self.class_weights)
        if type(self.class_weights) is list:
            self.class_weights = np.array(self.class_weights, dtype=np.float32)

        self.transitions = eval(self.transitions)
        if type(self.transitions) is list:
            self.transitions = np.array(self.transitions, dtype = np.float32)

        if self.overlap:
            assert self.load_model_path  # only use overlapping during test time
            assert self.overlap_offset < self.core_length
            # check if everything divides evenly to avoid further head aches
            assert (20000 / self.core_length).is_integer()  # assume 20000 chunk size
            assert (self.batch_size / (20000 / self.overlap_offset)).is_integer()
            assert ((20000 - self.core_length) / 2 / self.overlap_offset).is_integer()

        if self.nni:
            hyperopt_args = nni.get_next_parameter()
            self.__dict__.update(hyperopt_args)
            nni_save_model_path = os.path.expandvars('$NNI_OUTPUT_DIR/best_model.h5')
            nni_pred_output_path = os.path.expandvars('$NNI_OUTPUT_DIR/predictions.h5')
            self.__dict__['save_model_path'] = nni_save_model_path
            self.__dict__['prediction_output_path'] = nni_pred_output_path
            args.update(hyperopt_args)
            # for the print out
            args['save_model_path'] = nni_save_model_path
            args['prediction_output_path'] = nni_pred_output_path
        if self.verbose:
            print()
            pprint(args)

    def generate_callbacks(self):
        callbacks = [ConfusionMatrixTrain(self.gen_validation_data(), self.save_model_path,
                                          report_to_nni=self.nni)]
        if self.save_every_epoch:
            callbacks.append(SaveEveryEpoch(os.path.dirname(self.save_model_path)))
        return callbacks

    def set_resources(self):
        K.set_floatx(self.float_precision)
        if self.specific_gpu_id > -1:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID';
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.specific_gpu_id)

    def gen_training_data(self):
        SequenceCls = self.sequence_cls()
        return SequenceCls(model=self,
                           h5_file=self.h5_train,
                           mode='train',
                           shuffle=True)

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

        if not self.load_model_path:
            self.h5_train = h5py.File(os.path.join(self.data_dir, 'training_data.h5'), 'r')
            self.h5_val = h5py.File(os.path.join(self.data_dir, 'validation_data.h5'), 'r')
            self.shape_train = self.h5_train['/data/X'].shape
            self.shape_val = self.h5_val['/data/X'].shape

            n_train_correct_seqs = get_n_correct_seqs(self.h5_train)
            n_val_correct_seqs = get_n_correct_seqs(self.h5_val)

            n_train_seqs = n_train_correct_seqs
            n_val_seqs = self.shape_val[0]  # always validate on all

            n_intergenic_train_seqs = get_n_intergenic_seqs(self.h5_train)
            n_intergenic_val_seqs = get_n_intergenic_seqs(self.h5_val)
        else:
            self.h5_test = h5py.File(self.test_data, 'r')
            self.shape_test = self.h5_test['/data/X'].shape

            n_test_correct_seqs = get_n_correct_seqs(self.h5_test)
            n_test_seqs_with_intergenic = self.shape_test[0]
            n_intergenic_test_seqs = get_n_intergenic_seqs(self.h5_test)

        if self.verbose:
            print('\nData config: ')
            if not self.load_model_path:
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
        # first zero out any predictions that were made across seqid borders
        chunk_size = predictions.shape[1]
        seqids = test_sequence._get_seqids_for_batch(batch_idx)
        n_cross_seq_preds = chunk_size // self.overlap_offset - 1
        for i in range(len(seqids) - 1):
            if seqids[i] != seqids[i + 1]:
                start_idx = i * (n_cross_seq_preds + 1) + 1
                zeros = np.zeros((n_cross_seq_preds,) + predictions.shape[1:], dtype=predictions.dtype)
                predictions[start_idx:start_idx + n_cross_seq_preds] = zeros

        # actual overlapping; save first and last sequence for special handling later
        first, last = predictions[0], predictions[-1]
        # cut to the core
        seq_overhang = int((chunk_size - self.core_length) / 2)
        predictions = [s[seq_overhang:-seq_overhang] for s in predictions]
        # generate sequences at the start and end
        start_seqs = [first[j:j+self.core_length]
                      for j in range(0, seq_overhang, self.overlap_offset)]
        end_seqs = [last[j-self.core_length:j]
                    for j in range(chunk_size - seq_overhang + self.overlap_offset,
                                   chunk_size + 1,
                                   self.overlap_offset)]
        predictions = start_seqs + predictions + end_seqs
        predictions = np.stack(predictions)

        # merge and stack efficiently so everything can be averaged
        n_overlapping_seqs = self.core_length // self.overlap_offset
        n_predicted_original_seqs = test_sequence._seqs_per_batch(batch_idx=batch_idx)
        n_predicted_bases = n_predicted_original_seqs * chunk_size

        stacked = np.zeros((n_overlapping_seqs, n_predicted_bases, 4), dtype=predictions.dtype)
        for j in range(n_overlapping_seqs):
            # get idx of every n_overlapping_seqs'th seq starting at j
            idx = list(range(j, predictions.shape[0], n_overlapping_seqs))
            seq = np.concatenate(predictions[idx], axis=0)
            start_base = j * self.overlap_offset
            stacked[j, start_base:start_base+seq.shape[0]] = seq

        # average individual softmax values
        # does change pseudo-probability dist at the edge but not the argmax afterwards
        # (causes values to be lower there)
        averages = np.mean(stacked, axis=0)
        predictions = np.stack(np.split(averages, n_predicted_original_seqs))
        return predictions

    def _make_predictions(self, model):
        # loop through batches and continously expand output dataset as everything might
        # not fit in memory
        pred_out = h5py.File(self.prediction_output_path, 'w')
        test_sequence = self.gen_test_data()
        for i in range(len(test_sequence)):
            if self.verbose:
                print(i, '/', len(test_sequence), end='\r')
            predictions = model.predict_on_batch(test_sequence[i][0])
            if type(predictions) is list:
                predictions, meta_predictions = predictions
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
        pred_out.attrs['timestamp'] = str(datetime.datetime.now())
        pred_out.close()
        h5_model.close()

    def _load_helixer_model(self):
        model = load_model(self.load_model_path, custom_objects = {
            'LayerNormalization': LayerNormalization,
        })
        return model

    def _print_model_info(self, model):
        os.chdir(os.path.dirname(__file__))
        cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        branch = subprocess.check_output(cmd).strip().decode()
        cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
        commit = subprocess.check_output(cmd).strip().decode()
        print(f'Current helixerprep branch: {branch} ({commit})\n')

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
        if not self.load_model_path:
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
                                validation_data=self.gen_validation_data(),
                                callbacks=self.generate_callbacks(),
                                verbose=True)

            # set all model instance variables so predictions are made on the validation set
            self.h5_test = self.h5_val
            self.shape_test = self.shape_val
            self.load_model_path = self.save_model_path
            self.test_data = os.path.join(self.data_dir, 'validation_data.h5')
            self.class_weights = None
            model = self._load_helixer_model()
            self._make_predictions(model)
            print(f'Predictions made with {self.load_model_path} on {self.test_data} '
                  + f'and saved to {self.prediction_output_path}')

            self.h5_train.close()
            self.h5_val.close()

        # predict instead of train
        else:
            assert self.test_data.endswith('.h5'), 'Need a h5 test data file when loading a model'
            assert self.load_model_path.endswith('.h5'), 'Need a h5 model file'
            model = self._load_helixer_model()
            self._print_model_info(model)

            if self.eval:
                _ = HelixerModel.run_confusion_matrix(self.gen_test_data(), model)
            else:
                if os.path.isfile(self.prediction_output_path):
                    print(f'{self.prediction_output_path} already existing and will be overridden.')
                self._make_predictions(model)
            self.h5_test.close()
