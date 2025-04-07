import click
import functools

from helixer.cli.cli_callbacks import *
from helixer.cli.cli_formatter import *
from helixer.cli.shared_options import *

help_groups = HelpGroups()  # not necessary is in shared options like everything else


# todo: add more checks to give clearer error messages (i.e. file not found, corrupted zarr etc.)
@click.group()
def cli():
    """Choose to train, test or predict."""
    pass


# Train options
# ----------------------------------------------------------------
def train_options(func):
    # io
    @click.option('-d', '--data-dir',
                  type=click.Path(exists=True),
                  default=None,
                  help='Directory containing training and validation data (.zarr files); '
                       'naming convention: "training_data(...).zarr", "validation_data(...).zarr"',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @click.option('-s', '--save-model-path',
                  type=str,
                  default='./best_helixer_model.ckpt',
                  callback=validate_path_fragment,
                  help='Path to save the model with the best validation genic (CDS, UTR and Intron) F1 to',
                  cls=HelpGroupOption, help_group=help_groups.io)
    # resources
    @float_precision_option()
    @device_option()
    @num_devices_option()
    @click.option('--num-workers',
                  type=click.IntRange(1,),
                  default=0,
                  help='Number of subprocesses to use for data loading (number of CPU cores), '
                       '0 means data is loaded on the main process',
                  cls=HelpGroupOption, help_group=help_groups.resource)
    # training
    @click.option('-e', '--epochs',
                  type=click.IntRange(1,),
                  default=10000,
                  help='Number of training runs',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @batch_size_option('training set batch size', help_groups.train)
    @click.option('--val-batch-size',
                  type=click.IntRange(1,),
                  default=64,
                  help='validation set batch size',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--seed',
                  type=int,
                  default=None,
                  help='Random seed for training reproducibility',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--patience',
                  type=click.IntRange(1,),
                  default=3,
                  help='Allowed epochs without the validation genic F1 improving before stopping training',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--check-every-nth-batch',
                  type=click.IntRange(1,),
                  default=None,
                  help='Check validation genic F1 every nth batch (if not set: check once after every epoch)',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--optimizer',
                  type=str,
                  default='adamw',
                  help='Optimizer algorithm; options: adam or adamw',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--clip-norm',
                  type=float,
                  default=3.0,
                  help='The gradient of each weight is individually clipped so that its norm is '
                       'no higher than this value',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--learning-rate',
                  type=click.FloatRange(0, 1),
                  default=3e-4,
                  help='Learning rate for training',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--weight-decay',
                  type=click.FloatRange(0, .1),
                  default=3.5e-5,
                  help='Weight decay for training; penalizes complexity and prevents overfitting',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--class-weights',
                  type=str,
                  default='None',
                  callback=validate_weights,
                  help='Weighting of the 4 classes: Intergenic, UTR, CDS, Intron (Helixer predictions)',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @click.option('--transition-weights',
                  type=str,
                  default='None',
                  callback=validate_weights,
                  help='Weighting of the 6 transition categories: transcription start site, start codon, '
                       'donor splice site, transcription stop site, stop codon, acceptor splice site',
                  cls=HelpGroupOption, help_group=help_groups.train)
    # todo: check the rest of the code, predict phase is now the only option ever!!!
    @click.option('--resume-training',
                  is_flag=True,
                  help='Add this to resume training (pretrained model checkpoint necessary)',
                  cls=HelpGroupOption, help_group=help_groups.train)
    @load_model_path_option(help_groups.train)
    @workers_option('Number of threads used to fetch input data for training. Consider '
                    'setting to match the number of GPUs.')
    # misc
    @click.option('--save-every-check',
                  is_flag=True,
                  help='Add to save a model checkpoint every validation genic F1 check (see '
                       '--check-every-nth-batch in training parameters)',
                  cls=HelpGroupOption, help_group=help_groups.misc)
    # @click.option('--nni', is_flag=True) todo: think about reintegration later
    # maybe always be verbose to an extent and if debug is set be MORE verbose
    @click.option('-v', '--verbose',
                  is_flag=True,
                  help='Add to run Helixer in verbosity mode (additional information will be printed)',
                  cls=HelpGroupOption, help_group=help_groups.misc)
    @click.option('--debug',
                  is_flag=True,
                  help='Add to run in debug mode; truncates input data to small example (for training: '
                       'just runs a few epochs)',
                  cls=HelpGroupOption, help_group=help_groups.misc)
    # fine tune
    @click.option('--fine-tune',
                  is_flag=True,
                  help='Add/Use with --resume-training to replace and fine tune just the very last layer',
                  cls=HelpGroupOption, help_group='Fine-tuning parameters')
    @click.option('--pretrained-model-path',
                  type=click.Path(exists=True),
                  help='Required when predicting with a model fine tuned with coverage',
                  cls=HelpGroupOption, help_group='Fine-tuning parameters')
    @click.option('--input-coverage',
                  is_flag=True,
                  help='Add to use "evaluation/rnaseq_(spliced_)coverage" from Zarr training/validation '
                       'files as additional input for a late layer of the model',
                  cls=HelpGroupOption, help_group='Fine-tuning parameters')
    @click.option('--coverage-norm',
                  default=None,
                  help='Input coverage normalization (None, linear or log (recommended))',
                  cls=HelpGroupOption, help_group='Fine-tuning parameters')
    @click.option('--add-hidden-layer',
                  is_flag=True,
                  help='Adds extra dense layer between concatenating coverage and final output layer',
                  cls=HelpGroupOption, help_group='Fine-tuning parameters')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def test_options(func):
    @click.option('-t', '--test-data-path',
                  type=click.Path(exists=True),  # maybe validate_path_fragment for a folder test in a loop
                  default=None,
                  help='Path to one test Zarr file.',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @load_model_path_option(help_groups.io)
    # resources
    @float_precision_option()
    @device_option()
    @num_devices_option()
    # test
    @batch_size_option('test batch size', help_groups.test)
    @overlap_option('Add to improve test metrics quality at subsequence ends by creating and overlapping '
                    'sliding-window predictions (with proportional increase in time usage)', help_groups.test)
    @overlap_offset_option(help_groups.test)
    @overlap_core_length_option(help_groups.test)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def predict_options(func):
    @click.option('-i', '--input-data-path',
                  type=click.Path(exists=True),
                  default=None,
                  help='Path to one Zarr file to predict genes for.',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @click.option('-p', '--prediction-output-path',
                  type=str,
                  default='./predictions.zarr',
                  callback=combine_callbacks(validate_path_fragment, validate_file_extension),
                  help='Output path of the Zarr prediction file. (Helixer base-wise predictions)',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @load_model_path_option(help_groups.io)
    # resources
    @float_precision_option()
    @device_option()
    @num_devices_option()
    # prediction
    @batch_size_option('prediction batch size', help_groups.pred)
    @overlap_option('Add to improve prediction quality at subsequence ends by creating and overlapping '
                    'sliding-window predictions (with proportional increase in time usage)', help_groups.pred)
    @overlap_offset_option(help_groups.pred)
    @overlap_core_length_option(help_groups.pred)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# TODO: ALMOST everything should have no default, so we can safely fail early because something is missing or not hooked up correctly!!
# add bypass if it works
def hybrid_model_parameters(func):
    @click.option('--cnn-layers',
                  type=click.IntRange(1,),
                  default=1,
                  help='Number of convolutional layers',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--lstm-layers',
                  type=click.IntRange(1,),
                  default=1,
                  help='Number of bidirectional LSTM layers',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--units',
                  type=click.IntRange(1,),
                  default=32,
                  help='Number of LSTM units per bLSTM layer',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--filter-depth',
                  type=click.IntRange(1,),
                  default=32,
                  help='Filter depth for convolutional layers',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--kernel-size',
                  type=click.IntRange(1,),
                  default=26,
                  help='Kernel size for convolutional layers',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--pool-size',
                  type=click.IntRange(1,),
                  default=9,
                  help='Best set to a multiple of 3 (codon/nucleotide triplet size)',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--dropout1',
                  type=click.FloatRange(0.0, 1.0),
                  default=0.0,
                  help='If > 0, will add dropout layer with given dropout probability after the CNN.',
                  cls=HelpGroupOption, help_group='Model parameters')
    @click.option('--dropout2',
                  type=click.FloatRange(0.0, 1.0),
                  default=0.0,
                  help='If > 0, will add dropout layer with given dropout probability after the bLSTM block.',
                  cls=HelpGroupOption, help_group='Model parameters')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # todo: add config file stuff, then use **params in return instead kwargs
        return func(*args, **kwargs)
    return wrapper
