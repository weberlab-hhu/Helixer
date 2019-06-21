import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from abc import ABC, abstractmethod
import nni
import os
import sys
import h5py
import random
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import configparser
from pprint import pprint
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback
from keras import optimizers
from keras import backend as K

def get_col_accuracy_fn(col):
    def col_accuracy(y_true, y_pred, col):
        return K.cast(K.equal(y_true[:, :, col], K.round(y_pred[:, :, col])), K.floatx())
    fn = partial(col_accuracy, col=col)
    if col == 0:
        fn.__name__ = 'acc_t'
    elif col == 1:
        fn.__name__ = 'acc_c'
    elif col == 2:
        fn.__name__ = 'acc_i'
    return fn


class SaveEveryEpoch(Callback):
    def __init__(self):
        super(SaveEveryEpoch, self).__init__()

    def on_epoch_end(self, epoch, _):
        self.model.save('model' + str(epoch) + '.h5')


class ReportIntermediateResult(Callback):
    def __init__(self):
        super(ReportIntermediateResult, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        nni.report_intermediate_result(logs['val_loss'])

# used for development
TRUNCATE = 100000
# TRUNCATE = 100

class Generators(object):
    def __init__(self, h5_train, h5_val):
        self.h5_train = h5_train
        self.h5_val = h5_val

    def _gen_data(self, h5_file, shuffle, batch_size=2**2):
        n_seq = h5_file['/data/X'].shape[0]
        X, y, sample_weights = [], [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for i in seq_indexes:
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                sample_weights.append(h5_file['/data/sample_weights'][i])
                while len(X) == batch_size:
                    yield (
                        np.stack(X, axis=0)[:, :TRUNCATE, :],
                        np.stack(y, axis=0)[:, :TRUNCATE, :],
                        np.stack(sample_weights, axis=0)[:, :TRUNCATE],
                    )
                    X, y, sample_weights = [], [], []

    def gen_training_data(self, batch_size=2**2):
        gen = self._gen_data(self.h5_train, True, batch_size)
        while True:
            yield next(gen)

    def gen_validation_data(self, batch_size=2**2):
        gen = self._gen_data(self.h5_val, False, batch_size)
        while True:
            yield next(gen)


class HelixerModel(ABC):

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data_dir', type=str, default='')
        self.parser.add_argument('-p', '--patience', type=int, default=10)
        self.parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')
        self.parser.add_argument('-lm', '--load-model-path', type=str, default='')
        self.parser.add_argument('-e', '--epochs', type=int, default=10000)
        self.parser.add_argument('-bs', '--batch-size', type=int, default=8)
        self.parser.add_argument('-opt', '--optimizer', type=str, default='adam')
        self.parser.add_argument('-loss', '--loss', type=str, default='')
        self.parser.add_argument('-cn', '--clip-norm', type=float, default=1.0)
        self.parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
        self.parser.add_argument('-fp', '--float-precision', type=str, default='float32')
        self.parser.add_argument('-gpus', '--gpus', type=int, default=1)
        self.parser.add_argument('-cpus', '--cpus', type=int, default=8)
        self.parser.add_argument('-only-cpu', '--only-cpu', action='store_true')
        self.parser.add_argument('-plot', '--plot', action='store_true')
        self.parser.add_argument('-nni', '--nni', action='store_true')
        self.parser.add_argument('-v', '--verbose', action='store_true')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        self.__dict__.update(args)

        if self.nni:
            hyperopt_args = nni.get_next_parameter()
            self.__dict__.update(hyperopt_args)
        if self.verbose:
            print()
            pprint(args)

        self.h5_train = h5py.File(os.path.join(self.data_dir, 'training_data.h5'), 'r')
        self.h5_val = h5py.File(os.path.join(self.data_dir, 'validation_data.h5'), 'r')
        self.train_shape = self.h5_train['/data/X'].shape
        self.val_shape = self.h5_val['/data/X'].shape
        if self.verbose:
            print('\nTraining data shape: {}'.format(self.train_shape[:2]))
            print('Validation data shape: {}\n'.format(self.val_shape[:2]))
            # print('\n Data config:')
            # pprint(config_dict)
            # print()
        # self.__dict__.update(config_dict)

    def generate_callbacks(self):
        callbacks = [
            History(),
            CSVLogger('history.log'),
            EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1),
            ModelCheckpoint(self.save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        if self.nni:
            callbacks.append(ReportIntermediateResult())
        return callbacks

    def set_resources(self, n_cpus=8, use_gpu=True, n_gpus=1, fp_precision='float32'):
        K.set_floatx(fp_precision)

        if use_gpu:
            device_count = {'CPU': n_cpus, 'GPU': n_gpus}
        else:
            device_count = {'CPU': n_cpus, 'GPU': 0}

        config = tf.ConfigProto(intra_op_parallelism_threads=n_cpus,
                                inter_op_parallelism_threads=n_cpus,
                                allow_soft_placement=True,
                                device_count=device_count)
        session = tf.Session(config=config)
        K.set_session(session)

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    def run(self):
        self.set_resources(n_cpus=self.cpus,
                           use_gpu=not self.only_cpu,
                           n_gpus=self.gpus,
                           fp_precision=self.float_precision)
        model = self.model()

        if self.verbose:
            print(model.summary())
        else:
            print('Total params: {:,}'.format(model.count_params()))

        if self.plot:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png')
            print('Plotted to model.png')
            sys.exit()

        # we either train or predict
        if not self.load_model_path:
            if self.optimizer == 'adam':
                self.optimizer = optimizers.Adam(lr=self.learning_rate,
                                                 clipnorm=self.clip_norm)
            elif self.optimizer == 'rmsprop':
                self.optimizer = optimizers.RMSprop(lr=self.learning_rate,
                                                    clipnorm=self.clip_norm)
            elif self.optimizer == 'adagrad':
                print('learning rate not changed from default for adagrad')
                self.optimizer = optimizers.Adagrad(clipnorm=self.clip_norm)
            else:
                raise ValueError('Unknown Optimizer')

            self.compile_model(model)

            generators = Generators(self.h5_train, self.h5_val)
            model.fit_generator(generator=generators.gen_training_data(self.batch_size),
                                steps_per_epoch=self.train_shape[0] // self.batch_size,
                                # steps_per_epoch=10,
                                epochs=self.epochs,
                                validation_data=generators.gen_validation_data(self.batch_size),
                                validation_steps=self.val_shape[0] // self.batch_size,
                                # validation_steps=1,
                                callbacks=self.generate_callbacks(),
                                # do not use without keras.utils.Sequence
                                # use_multiprocessing=True,
                                # workers=4,
                                verbose=True)

            best_val_loss = min(model.history.history['val_loss'])
            if self.nni:
                nni.report_final_result(best_val_loss)

        self.h5_train.close()
        self.h5_val.close()
