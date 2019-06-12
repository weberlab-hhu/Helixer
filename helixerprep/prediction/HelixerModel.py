import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from abc import ABC, abstractmethod
import os
import sys
import random
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import configparser
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback
from keras import optimizers


class SaveEveryEpoch(Callback):
    def on_epoch_end(self, epoch, _):
        self.model.save('model' + str(epoch) + '.h5')

# used for development
TRUNCATE = 10000

class Generators(object):
    """Provides the data generator for the training and validation. The generators
    return data that has been padded to the length of the longest sample in a batch.
    The sample weights are set to 0 for the padded bases (just as for annotation errors).

    Omits the first sequence in each file for validation.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_file_paths(self):
        return [os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                    if f != 'data_config.ini']

    def gen_training_data(self, batch_size=2**2):
        X, y, sample_weights = None, None, None
        while True:
            files = self.data_file_paths()
            random.shuffle(files)
            for file_path in files:
                f = np.load(file_path, allow_pickle=True)
                if X is None:
                    X = f['X'][1:]
                    y = f['y'][1:]
                    sample_weights = f['sample_weights'][1:]
                else:
                    X = np.concatenate((X, f['X'][1:]))
                    y = np.concatenate((y, f['y'][1:]))
                    sample_weights = np.concatenate((sample_weights, f['sample_weights'][1:]))
                f.close()
                while len(X) >= batch_size:
                    yield (
                        X[:batch_size, :TRUNCATE, :],
                        y[:batch_size, :TRUNCATE, :],
                        sample_weights[:batch_size, :TRUNCATE]
                    )
                    # reset collected samples
                    X, y = X[batch_size:], y[batch_size:]
                    sample_weights = sample_weights[batch_size:]

    def gen_validation_data(self):
        """Returns the first sequence in each file as validation set.
        Very redundant due to strange errors when trying to resolve the redundancy.
        """
        X, y, sample_weights = None, None, None
        for file_path in self.data_file_paths():
            f = np.load(file_path, allow_pickle=True)
            if X is None:
                X = np.expand_dims(f['X'][0], axis=0)
                y = np.expand_dims(f['y'][0], axis=0)
                sample_weights = np.expand_dims(f['sample_weights'][0], axis=0)
            else:
                X = np.concatenate((X, np.expand_dims(f['X'][0], axis=0)))
                y = np.concatenate((y, np.expand_dims(f['y'][0], axis=0)))
                sample_weights = np.concatenate((sample_weights,
                                                np.expand_dims(f['sample_weights'][0], axis=0)))
            f.close()
        while True:
            yield (X[:, :TRUNCATE, :], y[:, :TRUNCATE, :], sample_weights[:, :TRUNCATE])


class HelixerModel(ABC):

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data-dir', type=str, default='data/data')
        self.parser.add_argument('-p', '--patience', type=int, default=10)
        self.parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')
        self.parser.add_argument('-e', '--epochs', type=int, default=10000)
        self.parser.add_argument('-bs', '--batch-size', type=int, default=8)
        self.parser.add_argument('-opt', '--optimizer', type=str, default='adam')
        self.parser.add_argument('-loss', '--loss', type=str, default='')
        self.parser.add_argument('-cn', '--clip-norm', type=float, default=1.0)
        self.parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
        self.parser.add_argument('-see', '--save-every-epoch', action='store_true')
        self.parser.add_argument('-v', '--verbose', action='store_true')
        self.parser.add_argument('-plot', '--plot', action='store_true')
        self.parser.add_argument('-nni', '--nni', action='store_true')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        self.__dict__.update(args)

        if self.nni:
            import nni
            hyperopt_args = nni.get_next_parameter()
            self.__dict__.update(hyperopt_args)
        if self.verbose:
            print()
            pprint(args)

        config = configparser.ConfigParser()
        config.read(os.path.join(self.data_dir, 'data_config.ini'))
        config_dict = {}
        for key, value in config['data'].items():
            # we only have boolean and int config values at the moment
            if value in ['True', 'False']:
                config_dict[key] = config.getboolean('data', key)
            else:
                config_dict[key] = config.getint('data', key)
        if self.verbose:
            print('\n Data config:')
            pprint(config_dict)
            print()
        self.__dict__.update(config_dict)

    def generate_callbacks(self):
        callbacks = [
            History(),
            CSVLogger('history.log'),
            # EarlyStopping(monitor='val_loss', patience=self.patience),
        ]

        if self.save_every_epoch:
            callbacks.append(SaveEveryEpoch())
        else:
            checkpoint_cb = ModelCheckpoint(self.save_model_path,
                                            monitor='val_acc',
                                            save_best_only=True,
                                            verbose=0)
            callbacks.append(checkpoint_cb)
        return callbacks

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    @abstractmethod
    def plot_history(self, history):
        pass

    @abstractmethod
    def training_summary(self):
        pass

    def run(self):
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

        generators = Generators(self.data_dir)
        # run validation generator once first to avoid race conditions due to same file access
        print('\nstart loading validation data')
        val_gen = generators.gen_validation_data()
        next(val_gen)
        print('validation data loaded\n')

        model.fit_generator(generator=generators.gen_training_data(self.batch_size),
                            steps_per_epoch=(self.n_seq_total - self.n_files) // self.batch_size,
                            epochs=self.epochs,
                            validation_data=val_gen,
                            validation_steps=1,
                            callbacks=self.generate_callbacks(),
                            # do not use without keras.utils.Sequence
                            # use_multiprocessing=True,
                            # workers=4,
                            verbose=True)

        best_val_acc = max(model.history.history['val_acc'])
        if self.nni:
            nni.report_final_result(best_val_acc)

        # print the overall summary
        # self.training_summary()
        # plot the collected history
        # self.plot_histor(collected_histories)
