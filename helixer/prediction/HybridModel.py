#! /usr/bin/env python3
import click
#import tensorflow as tf

#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Bidirectional, Dropout, Reshape,
#                                     Activation, Input, BatchNormalization)
from helixer.prediction.HelixerModel import HelixerModel, HelixerSequence
from helixer.cli.model_cli import helixer_base_model_parameters, hybrid_model_parameters
from helixer.cli.cli_formatter import HelpGroupCommand


class HybridSequence(HelixerSequence):
    def __init__(self, model, zarr_files, mode, batch_size, rank, world_size):
        super().__init__(model, zarr_files, mode, batch_size, rank, world_size)

    def __getitem__(self, idx):
        X, y, sw, transitions, phases, _, coverage_scores = self._generic_get_item(idx)

        if self.only_predictions:
            return X
        else:
            return X, y, sw


class HybridModel(HelixerModel):
    def __init__(self, cnn_layers, lstm_layers, units, filter_depth,
                 kernel_size, pool_size, dropout1, dropout2):
        super().__init__()
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.units = units
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    @staticmethod
    def sequence_cls():
        return HybridSequence

    def model(self):
        values_per_bp = 4
        if self.input_coverage:
            values_per_bp += self.coverage_count * 2

            raw_input = Input(shape=(None, values_per_bp), dtype=self.float_precision,
                              name='raw_input')
            main_input, coverage_input = tf.split(raw_input, [4, 2 * self.coverage_count],
                                                  axis=-1)
            model_input = raw_input
        else:
            main_input = Input(shape=(None, values_per_bp), dtype=self.float_precision,
                               name='main_input')
            model_input = main_input
            coverage_input = None

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
            # x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)

        if self.dropout1 > 0.0:
            x = Dropout(self.dropout1)(x)

        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        for _ in range(self.lstm_layers - 1):
            x = Bidirectional(LSTM(self.units, return_sequences=True))(x)

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            x = Dropout(self.dropout2)(x)

        outputs = self.model_hat((x, coverage_input))

        model = Model(inputs=model_input, outputs=outputs)
        return model

    def model_hat(self, penultimate_layers):
        x, coverage_input = penultimate_layers
        # maybe concatenate coverage and add one extra dense at this point
        if self.input_coverage:
            coverage_input = Reshape((-1, self.pool_size * self.coverage_count * 2))(coverage_input)
            x = tf.concat([x, coverage_input], axis=-1)
            if self.post_coverage_hidden_layer:
                x = Dense(self.units // 2)(x)

        if self.predict_phase:
            x = Dense(self.pool_size * 4 * 2)(x)  # predict twice a many floats
            x_genic, x_phase = tf.split(x, 2, axis=-1)

            x_genic = Reshape((-1, self.pool_size, 4), name='reshape_hat')(x_genic)
            x_genic = Activation('softmax', name='genic')(x_genic)

            x_phase = Reshape((-1, self.pool_size, 4), name='reshape_hat1')(x_phase)
            x_phase = Activation('softmax', name='phase')(x_phase)

            outputs = [x_genic, x_phase]
        else:
            x = Dense(self.pool_size * 4)(x)
            x = Reshape((-1, self.pool_size, 4), name='reshape_hat')(x)
            x = Activation('softmax', name='main')(x)
            outputs = [x]

        return outputs

    def compile_model(self, model):
        if self.predict_phase:
            losses = ['categorical_crossentropy', 'categorical_crossentropy']
            loss_weights = [0.8, 0.2]
        else:
            losses = ['categorical_crossentropy']
            loss_weights = [1.0]

        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      loss_weights=loss_weights,
                      sample_weight_mode='temporal')


# hint: CLI loading slowly? --> cause: importing large packages at the start of the file
@click.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@helixer_base_model_parameters
@hybrid_model_parameters
def run_hybrid_model(**kwargs):
    """Run Helixer's Hybrid Model directly for training, evaluation or prediction."""
    pass
    # todo: fabric setup function here, can be called from multiple scripts fabric.launch()
    #model = HybridModel(**kwargs)  # launch fabric on model init or outside??
    #model.run()


if __name__ == '__main__':
    run_hybrid_model()
