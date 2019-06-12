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
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback
from keras import optimizers


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


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

    def gen_training_data(self, batch_size=2**2):
        X, y, sample_weights = None, None, None
        while True:
            files = listdir_fullpath(self.data_dir)
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
        for file_path in listdir_fullpath(self.data_dir):
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

    def _plot_confusion_matrix(self, cm,
                               classes,
                               normalize=False,
                               title='Confusion matrix',
                               output_name='confusion_matrix.png',
                               cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(14, 14), dpi=200)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(output_name)
        print('confusion matrix saved to', output_name)

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

        # the config is the same for all individual data files
        # f = h5py.File(listdir_fullpath(self.data_dir)[0])
        # self.config = f['config'].attrs
        # f.close()
        if self.verbose:
            print('\n Data config:')
            pprint(self.config)
            print()

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
        val_gen = generators.gen_validation_data()
        next(val_gen)

        model.fit_generator(generator=generators.gen_training_data(self.batch_size),
                            steps_per_epoch=10,  # todo read from config
                            epochs=self.epochs,
                            validation_data=val_gen,
                            validation_steps=1,
                            callbacks=self.generate_callbacks(),
                            # do not use without keras.utils.Sequence
                            # use_multiprocessing=True,
                            # workers=4,
                            verbose=True)

        # best_val_acc = min(model.history.history['val_mean_absolute_error'])
        if self.nni:
            nni.report_final_result(best_val_acc)

        # print the overall summary
        # self.training_summary()
        # plot the collected history
        # self.plot_histor(collected_histories)

        # if the task is classification, print a classification report
        # and save confusion matrix
        """
        if self.one_hot_labels:
            model = load_model(self.save_model_path)
            one_hot_predictions = model.predict(self.test_data)
            y_pred = np.argmax(one_hot_predictions, axis=1)
            y_true = np.argmax(self.y_test, axis=1)
            report = classification_report(y_true,
                                           y_pred,
                                           target_names=self.class_names)
            print(report)

            cnf_matrix = confusion_matrix(y_true, y_pred)
            self._plot_confusion_matrix(cnf_matrix, self.class_names, normalize=True)
        """

