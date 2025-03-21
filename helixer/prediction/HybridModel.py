#! /usr/bin/env python3
import click
import torch
import torch.nn as nn

from helixer.prediction.HelixerModel import (HelixerModel, HelixerSequence, HelixerTrainer,
                                             HelixerTester, HelixerPredictor)
from helixer.cli.model_cli import hybrid_model_parameters, train_options, test_options, predict_options, cli
from helixer.cli.cli_formatter import HelpGroupCommand, ColumnHelpFormatter
# file name will be changed, working title
from helixer.prediction.TorchHelixerModel import Reshape, TransposeDimsOneTwo, bLSTM

click.Context.formatter_class = ColumnHelpFormatter

class HybridSequence(HelixerSequence):
    def __init__(self, model, zarr_files, mode, batch_size, rank, world_size):
        super().__init__(model, zarr_files, mode, batch_size, rank, world_size)

    def __getitem__(self, idx):
        X, y, sw, transitions, phases, _, coverage_scores = self._generic_get_item(idx)

        if self.only_predictions:
            return X
        else:
            return X, y, sw


class ModelHat(nn.Module):
    def __init__(self, units, pool_size):
        super(ModelHat, self).__init__()
        self.linear_layer = nn.Linear(units * 2, pool_size * 4 * 2)

        self.output_stack = nn.Sequential()
        self.output_stack.append(Reshape((-1, pool_size * 4)))
        self.output_stack.append(nn.Softmax(dim=2))  # last dim

    def forward(self, x):
        # phase is always predicted on default now
        x = self.linear_layer(x)
        x_genic, x_phase = torch.tensor_split(x, 2, dim=-1)
        x_genic = self.output_stack(x_genic)
        x_phase = self.output_stack(x_phase)
        return [x_genic, x_phase]


class HybridModel(HelixerModel):
    def __init__(self, cnn_layers, lstm_layers, units, filter_depth,
                 kernel_size, pool_size, dropout1, dropout2, n_classes):
        super().__init__()
        print("layers:", cnn_layers)
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.units = units
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        # extract n_classes from the input file for dynamical stuff that may come later
        self.n_classes = n_classes
        # needs to be at the end
        self.hparams = self.get_hparams()

        # WARNING!!: without RNA-seq coverage support so far

        # Add CNN stack
        # ---------------------------------
        self.cnn_blstm_stack = nn.Sequential()

        self.cnn_blstm_stack.append(TransposeDimsOneTwo())
        self.cnn_blstm_stack.append(nn.Conv1d(n_classes, self.filter_depth, self.kernel_size, padding='same'))
        self.cnn_blstm_stack.append(nn.ReLU())

        # if there are additional CNN layers
        for _ in range(self.cnn_layers - 1):
            # doesn't work like tensorflow, because of diff. available parameters and diff. definition of momentum
            self.cnn_blstm_stack.append(nn.BatchNorm1d(self.filter_depth))
            self.cnn_blstm_stack.append(nn.Conv1d(n_classes, self.filter_depth, self.kernel_size, padding='same'))
            self.cnn_blstm_stack.append(nn.ReLU())

        self.cnn_blstm_stack.append(TransposeDimsOneTwo())

        # Add bLSTM (and others) stack
        # --------------------------------
        if self.pool_size > 1:
            self.cnn_blstm_stack.append(Reshape((-1, self.pool_size * self.filter_depth)))

        if self.dropout1 > 0.0:
            self.cnn_blstm_stack.append(nn.Dropout(self.dropout1))

        self.cnn_blstm_stack.append(bLSTM(self.filter_depth, self.units, self.lstm_layers))

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            self.cnn_blstm_stack.append(nn.Dropout(self.dropout2))

        self.cnn_blstm_stack.append(ModelHat(self.units, self.pool_size))

    def forward(self, x):
        logits = self.cnn_blstm_stack(x)
        return logits

# todo: integrate these losses into the model setup!! as well as the sample weight mode
def compile_model(model):
    #if self.predict_phase:
    losses = ['categorical_crossentropy', 'categorical_crossentropy']
    loss_weights = [0.8, 0.2]
    #else:
    #    losses = ['categorical_crossentropy']
    #    loss_weights = [1.0]

    model.compile(optimizer=optimizer, # not part of PyTorch models
                  loss=losses,
                  loss_weights=loss_weights,
                  sample_weight_mode='temporal')

@cli.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@train_options
@hybrid_model_parameters
@click.pass_context
def train(ctx):
    """Train the Helixer Hybrid model."""
    ctx.params['model_class'] = HybridModel
    HelixerTrainer(**ctx.params).run()

@cli.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@test_options
@click.pass_context
def test(ctx):
    """Test the Helixer Hybrid model."""
    ctx.params['model_class'] = HybridModel
    HelixerTester(**ctx.params).run()

@cli.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@predict_options
@click.pass_context
def predict(ctx):
    """Predict with the Helixer Hybrid model."""
    ctx.params['model_class'] = HybridModel
    HelixerPredictor(**ctx.params).run()

if __name__ == '__main__':
    cli()
