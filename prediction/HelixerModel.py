import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from abc import ABC, abstractmethod
import os
import sys
import h5py
import argparse
import itertools
import numpy as np
import subprocess
import deepdish as dd
import tensorflow as tf
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import (EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback)
from keras.layers import (Input, Conv1D, Dense, Flatten,
                          Dropout, Subtract, Concatenate, BatchNormalization,
                          Activation)
from keras.regularizers import l2
from keras.models import load_model
from keras import optimizers


class SaveEveryEpoch(Callback):
    def on_epoch_end(self, epoch, _):
        self.model.save('model' + str(epoch) + '.h5')


class Generators(object):
    def __init__(self, path):
        self.path = path

    def gen_training_data(self, batch_size=2**10):
        while True:



class HelixerModel(ABC):

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--dataset', type=str, default='data/data.h5')
        self.parser.add_argument('-p', '--patience', type=int, default=10)
        self.parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')
        self.parser.add_argument('-e', '--epochs', type=int, default=10000)
        self.parser.add_argument('-r', '--runs', type=int, default=1)
        self.parser.add_argument('-bs', '--batch-size', type=int, default=32)
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

        self.config = dd.io.load(self.dataset, group='/config')
        if self.verbose:
            print('\n Data config:')
            pprint(self.config)
            print()

    def generate_callbacks(self):
        callbacks = [
            History(),
            CSVLogger('history.log'),
            EarlyStopping(monitor='val_loss', patience=self.patience),
        ]

        if self.save_every_epoch:
            callbacks.append(SaveEveryEpoch())
        else:
            checkpoint_cb = ModelCheckpoint(self.save_model_path,
                                            monitor='val_loss',
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

        print('\nRun ', num_run)
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

        model.fit(self.train_data,
                  self.y_train,
                  sample_weight=self.weights_train,
                  validation_data=(self.test_data, self.y_test, self.weights_test),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=self.generate_callbacks(num_run),
                  verbose=True)

        best_val_acc = min(model.history.history['val_mean_absolute_error'])
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

