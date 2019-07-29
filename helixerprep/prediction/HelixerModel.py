import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from F1Scores import F1Calculator

from abc import ABC, abstractmethod
import nni
import os
import sys
import h5py
import random
import argparse
import numpy as np
import tensorflow as tf
from pprint import pprint
from functools import partial
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, CSVLogger, Callback
from keras import optimizers
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model


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


class F1ResultsTest(Callback):
    def __init__(self, generator, n_steps):
        self.calculator = F1Calculator(generator, n_steps)
        super(F1ResultsTest, self).__init__()

    def on_test_end(self, logs=None):
        self.calculator.count_and_calculate(self.model)


class F1ResultsTrain(Callback):
    def __init__(self, generator, n_steps):
        self.calculator = F1Calculator(generator, n_steps)
        super(F1ResultsTrain, self).__init__()

    def on_test_end(self, logs=None):
        self.calculator.count_and_calculate(self.model)


class HelixerModel(ABC):
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data-dir', type=str, default='')
        self.parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')
        # training params
        self.parser.add_argument('-e', '--epochs', type=int, default=10000)
        self.parser.add_argument('-p', '--patience', type=int, default=10)
        self.parser.add_argument('-bs', '--batch-size', type=int, default=8)
        self.parser.add_argument('-opt', '--optimizer', type=str, default='adam')
        self.parser.add_argument('-loss', '--loss', type=str, default='')
        self.parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
        self.parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
        self.parser.add_argument('-igsw', '--intergenic-sample-weight', type=float, default=1)
        self.parser.add_argument('-ic', '--intergenic-chance', type=float, default=1.0)
        self.parser.add_argument('-ee', '--exclude-errors', action='store_true')
        # testing
        self.parser.add_argument('-lm', '--load-model-path', type=str, default='')
        self.parser.add_argument('-td', '--test-data', type=str, default='')
        self.parser.add_argument('-po', '--prediction-output-path', type=str, default='predictions.h5')
        self.parser.add_argument('-ev', '--eval', action='store_true')
        # resources
        self.parser.add_argument('-fp', '--float-precision', type=str, default='float32')
        self.parser.add_argument('-gpus', '--gpus', type=int, default=1)
        self.parser.add_argument('-cpus', '--cpus', type=int, default=8)
        self.parser.add_argument('--specific-gpu-id', type=int, default=-1)
        self.parser.add_argument('-only-cpu', '--only-cpu', action='store_true')
        # misc flags
        self.parser.add_argument('-plot', '--plot', action='store_true')
        self.parser.add_argument('-nni', '--nni', action='store_true')
        self.parser.add_argument('-v', '--verbose', action='store_true')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        self.__dict__.update(args)

        if self.nni:
            hyperopt_args = nni.get_next_parameter()
            self.__dict__.update(hyperopt_args)
            self.__dict__['save_model_path'] = '\'$NNI_OUTPUT_DIR\'/best_model.h5'
            args.update(hyperopt_args)
        if self.verbose:
            print()
            pprint(args)

    def generate_callbacks(self):
        callbacks = [
            History(),
            CSVLogger('history.log'),
            EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1),
            ModelCheckpoint(self.save_model_path, monitor='val_loss', save_best_only=True, verbose=1),
            F1ResultsTrain(self.gen_validation_data(), self.n_steps_val)
        ]
        if self.nni:
            callbacks.append(ReportIntermediateResult())
        return callbacks

    def set_resources(self):
        K.set_floatx(self.float_precision)
        if self.only_cpu:
            device_count = {'CPU': self.cpus, 'GPU': 0}
            config = tf.ConfigProto(intra_op_parallelism_threads=self.cpus,
                                    inter_op_parallelism_threads=self.cpus,
                                    allow_soft_placement=True,
                                    device_count=device_count)
            session = tf.Session(config=config)
            K.set_session(session)
        elif self.specific_gpu_id > -1:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID';
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.specific_gpu_id)

    @abstractmethod
    def _gen_data(self, h5_file, shuffle, exclude_err_seqs=False, sample_intergenic=False):
        pass

    def gen_training_data(self):
        gen = self._gen_data(self.h5_train, shuffle=True, exclude_err_seqs=self.exclude_errors,
                             sample_intergenic=True)
        while True:
            yield next(gen)

    def gen_validation_data(self):
        # reasons for the parameter setup of the generator: no need to shuffle, when we exclude
        # errorneous seqs during training we should do it here and we probably also want to
        # have a comparable validation set accross all possible parameters
        gen = self._gen_data(self.h5_val, shuffle=False, exclude_err_seqs=self.exclude_errors,
                             sample_intergenic=False)
        while True:
            yield next(gen)

    def gen_test_data(self):
        gen = self._gen_data(self.h5_test, shuffle=False, exclude_err_seqs=self.exclude_errors,
                             sample_intergenic=False)
        while True:
            yield next(gen)

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

    def set_optimizer(self):
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

    def open_data_files(self):
        def get_n_correct_seqs(h5_file):
            err_samples = np.array(h5_file['/data/err_samples'])
            return np.count_nonzero(err_samples == False)

        def get_n_intergenic_seqs(h5_file):
            ic_samples = np.array(h5_file['/data/fully_intergenic_samples'])
            return np.count_nonzero(ic_samples == True)

        def calculate_steps(n_seqs):
            if n_seqs % self.batch_size == 0:
                return n_seqs // self.batch_size
            else:
                return n_seqs // self.batch_size + 1

        if not self.load_model_path:
            self.h5_train = h5py.File(os.path.join(self.data_dir, 'training_data.h5'), 'r')
            self.h5_val = h5py.File(os.path.join(self.data_dir, 'validation_data.h5'), 'r')
            self.shape_train = self.h5_train['/data/X'].shape
            self.shape_val = self.h5_val['/data/X'].shape

            n_train_correct_seqs = get_n_correct_seqs(self.h5_train)
            n_val_correct_seqs = get_n_correct_seqs(self.h5_val)

            if self.exclude_errors:
                n_train_seqs_with_intergenic = n_train_correct_seqs
                n_val_seqs_with_intergenic = n_val_correct_seqs
            else:
                n_train_seqs_with_intergenic = self.shape_train[0]
                n_val_seqs_with_intergenic = self.shape_val[0]

            # potentially account for intergenic seqs
            n_intergenic_train_seqs = get_n_intergenic_seqs(self.h5_train)
            n_intergenic_val_seqs = get_n_intergenic_seqs(self.h5_val)

            if self.intergenic_chance < 1.0:
                self.n_train_seqs = n_train_seqs_with_intergenic - n_intergenic_train_seqs + \
                                    int(n_intergenic_train_seqs * self.intergenic_chance)
            else:
                self.n_train_seqs = n_train_seqs_with_intergenic
            # do not adjust the validation count for intergenic sampling as we don't do that then
            self.n_val_seqs = n_val_seqs_with_intergenic

            # set steps
            self.n_steps_train = calculate_steps(self.n_train_seqs)
            self.n_steps_val = calculate_steps(self.n_val_seqs)
            # self.n_steps_train = 50
            # self.n_steps_val = 20
        else:
            self.h5_test = h5py.File(self.test_data, 'r')
            self.shape_test = self.h5_test['/data/X'].shape
            self.n_test_seqs = self.shape_test[0]
            n_test_correct_seqs = get_n_correct_seqs(self.h5_test)
            if self.exclude_errors:
                n_test_seqs_with_intergenic = n_test_correct_seqs
            else:
                n_test_seqs_with_intergenic = self.shape_test[0]
            n_intergenic_test_seqs = get_n_intergenic_seqs(self.h5_test)

            self.n_steps_test = calculate_steps(self.shape_test[0])
            # self.n_steps_test = 2

        if self.verbose:
            print('\nData config: ')
            if not self.load_model_path:
                print(dict(self.h5_train.attrs))
                print('\nTraining data shape: {}'.format(self.shape_train[:2]))
                print('Validation data shape: {}'.format(self.shape_val[:2]))
                print('\nTotal est. training sequences: {}'.format(self.n_train_seqs))
                print('Total est. val sequences: {}'.format(self.n_val_seqs))
                print('\nEst. intergenic train/val seqs: {:.2f}% / {:.2f}%'.format(
                    n_intergenic_train_seqs / n_train_seqs_with_intergenic * 100,
                    n_intergenic_val_seqs / n_val_seqs_with_intergenic * 100))
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

    def run(self):
        self.set_resources()
        self.open_data_files()
        # we either train or predict
        if not self.load_model_path:
            model = self.model()
            if not self.only_cpu and self.gpus >= 2:
                model = multi_gpu_model(model, gpus=self.gpus)

            if self.verbose:
                print(model.summary())
            else:
                print('Total params: {:,}'.format(model.count_params()))

            if self.plot:
                self.plot_model(model)

            self.set_optimizer()
            self.compile_model(model)

            model.fit_generator(generator=self.gen_training_data(),
                                steps_per_epoch=self.n_steps_train,
                                epochs=self.epochs,
                                validation_data=self.gen_validation_data(),
                                validation_steps=self.n_steps_val,
                                callbacks=self.generate_callbacks(),
                                verbose=True)

            best_val_loss = min(model.history.history['val_loss'])
            if self.nni:
                nni.report_final_result(best_val_loss)

            self.h5_train.close()
            self.h5_val.close()

        # predict instead of train
        else:
            assert self.test_data.endswith('.h5'), 'Need a h5 test data file when loading a model'
            assert self.load_model_path.endswith('.h5'), 'Need a h5 model file'

            model = load_model(self.load_model_path, custom_objects = {
                'acc_t': get_col_accuracy_fn(0),
                'acc_c': get_col_accuracy_fn(1),
                'acc_i': get_col_accuracy_fn(2)
            })
            if self.eval:
                callback = [F1ResultsTest(self.gen_test_data(), self.n_steps_test)]
                metrics = model.evaluate_generator(generator=self.gen_test_data(),
                                                   steps=self.n_steps_test,
                                                   callbacks=callback,
                                                   verbose=True)
                metrics_names = model.metrics_names
                print({z[0]: z[1] for z in zip(metrics_names, metrics)})
            else:
                if os.path.isfile(self.prediction_output_path):
                    print('{} already existing and will be overridden.'.format(
                        self.prediction_output_path
                    ))
                predictions = model.predict_generator(generator=self.gen_test_data(),
                                                      steps=self.n_steps_test,
                                                      verbose=True)
                predictions = predictions.astype(np.float32)  # in case of predicting with float64

                h5_model = h5py.File(self.load_model_path, 'r')
                pred_out = h5py.File(self.prediction_output_path, 'w')
                pred_out.create_dataset('/predictions', data=predictions, compression='lzf',
                                        shuffle=True)
                # add model config to predictions
                pred_out.attrs['model_config'] = h5_model.attrs['model_config']
                pred_out.close()
                h5_model.close()

            self.h5_test.close()
