#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from HelixerModel import HelixerModel


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parse_args()

    def model(self):
        model = Sequential()
        model.add(LSTM(self.units, return_sequences=True, input_shape=(None, 4)))
        model.add(TimeDistributed(Dense(3, activation='sigmoid')))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      sample_weight_mode='temporal',
                      metrics=['accuracy'])

    def plot_history(self, history):
        fig, axes = plt.subplots(len(histories),
                                 2,
                                 sharey='col',
                                 figsize=(9, 3 * len(histories)))
        if len(histories) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, hist in enumerate(histories):
            ax_losses = axes[i, 0]
            ax_losses.plot(hist['val_loss'], label='val_loss')
            ax_losses.plot(hist['loss'], label='loss')
            ax_losses.set_xlabel('epochs')
            ax_losses.legend()

            if self.quantile_size:
                val_metric = hist['val_acc']
                metric = hist['acc']
                metric_label = 'acc'
            else:
                val_metric = hist['val_mean_absolute_error']
                metric = hist['mean_absolute_error']
                metric_label = 'mae'

            ax_metric = axes[i, 1]
            ax_metric.plot(val_metric, 'b-', label=metric_label + ' test')
            ax_metric.plot(metric, 'b:', label=metric_label + ' train')
            ax_metric.set_xlabel('epochs')
            ax_metric.set_ylabel('mae', color='b')
            ax_metric.tick_params('y', colors='b')
            ax_metric.legend(loc=2, framealpha=1.0)

        fig.tight_layout()
        plt.savefig('losses_and_metrics.png', dpi=300)
        plt.close()
        print('history plotted')

    def training_summary(self):
        print('\nbest val_loss', "{0:.4f}".format(min(self.best_val_losses)),
              'avg val_loss', "{0:.4f}".format(np.mean(self.best_val_losses)),
              'std val_loss', "{0:.4f}".format(np.std(self.best_val_losses)), sep='\t')
        best_val_metric = min(self.best_val_metrics)
        print('best val_metric', "{0:.4f}".format(best_val_metric),
              'avg val_metric', "{0:.4f}".format(np.mean(self.best_val_metrics)),
              'std val_metric', "{0:.4f}".format(np.std(self.best_val_metrics)),
              '\n', sep='\t')


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
