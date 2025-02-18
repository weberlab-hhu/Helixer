import click
import functools

from helixer.cli.cli_callbacks import *
from helixer.cli.cli_formatter import ColumnHelpFormatter, HelpGroupOption

click.Context.formatter_class = ColumnHelpFormatter

# todo Helixer.py check_lineage model as callback
# todo: add more checks to give clearer error messages (i.e. file not found, corrupted zarr etc.)
# todo: seperate train eval and test into click groups?, at least separate the run logic from the model
def helixer_base_model_parameters(func):
    # training params
    @click.option('-d', '--data-dir',
                  type=click.Path(exists=True),
                  default=None,
                  help='Directory containing training and validation data (.zarr files); '
                       'naming convention: "training_data(...).zarr", "validation_data(...).zarr"',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('-s', '--save-model-path',
                  type=str,
                  default='./best_helixer_model.ckpt',
                  callback=validate_path_fragment,
                  help='Path to save the model with the best validation genic (CDS, UTR and Intron) F1 to',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('-e', '--epochs',
                  type=click.IntRange(1,),
                  default=10000,
                  help='Number of training runs',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('-b', '--batch-size',
                  type=click.IntRange(1,),
                  default=8,
                  help='Batch size for training data',
                  cls=HelpGroupOption, help_group='Training parameters')
    # todo: also belongs to test predict parameters right now, will be separated
    @click.option('--val-test-batch-size',
                  type=click.IntRange(1,),
                  default=32,
                  help='Batch size for validation/test data',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--patience',
                  type=click.IntRange(1,),
                  default=3,
                  help='Allowed epochs without the validation genic F1 improving before stopping training',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--check-every-nth-batch',
                  type=click.IntRange(1,),
                  default=1_000_000,
                  help='Check validation genic F1 every nth batch (default: check once every epoch)',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--optimizer',
                  type=str,
                  default='adamw',
                  help='Optimizer algorithm; options: adam or adamw',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--clip-norm',
                  type=float,
                  default=3.0,
                  help='The gradient of each weight is individually clipped so that its norm is '
                       'no higher than this value',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--learning-rate',
                  type=click.FloatRange(0, 1),
                  default=3e-4,
                  help='Learning rate for training',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--weight-decay',
                  type=click.FloatRange(0, .1),
                  default=3.5e-5,
                  help='Weight decay for training; penalizes complexity and prevents overfitting',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--class-weights',
                  type=str,
                  default='None',
                  callback=validate_weights,
                  help='Weighting of the 4 classes: Intergenic, UTR, CDS, Intron (Helixer predictions)',
                  cls=HelpGroupOption, help_group='Training parameters')
    @click.option('--transition-weights',
                  type=str,
                  default='None',
                  callback=validate_weights,
                  help='Weighting of the 6 transition categories: transcription start site, start codon, '
                       'donor splice site, transcription stop site, stop codon, acceptor splice site',
                  cls=HelpGroupOption, help_group='Training parameters')
    # todo: check the rest of the code, predict phase is now the only option ever!!!
    @click.option('--resume-training',
                  is_flag=True,
                  help='Add this to resume training (pretrained model checkpoint necessary)',
                  cls=HelpGroupOption, help_group='Training parameters')
    # testing / predicting
    # todo: ckpt format soon
    @click.option('-l', '--load-model-path',
                  type=click.Path(exists=True),
                  default=None,
                  help='Path to a trained/pretrained model checkpoint (HDF5 format)',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    @click.option('-t', '--test-data-path',
                  type=click.Path(exists=True),
                  default=None,
                  help='Path to one test Zarr file.',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    @click.option('-p', '--prediction-output-path',
                  type=str,
                  default='./predictions.zarr',
                  callback=validate_path_fragment,
                  help='Output path of the Zarr prediction file. (Helixer base-wise predictions)',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    @click.option('--compression',
                  type=str,
                  default='gzip',
                  help='Compression used in the predictions Zarr file ("lzf" or "gzip")',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    # todo: own group
    @click.option('--eval',
                  is_flag=True,
                  help='Add to run test/validation run instead of predicting',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    @click.option('--overlap',
                  is_flag=True,
                  help='Add to improve prediction quality at subsequence ends by creating and overlapping '
                       'sliding-window predictions (with proportional increase in time usage)',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    # todo: check this later, needs read in zarr file
    @click.option('--overlap-offset',
                  type=click.IntRange(1,),
                  default=None,
                  help="Distance to 'step' between predicting subsequences when overlapping "
                       "(default: subsequence_length/2)",
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    @click.option('--core-length',
                  type=click.IntRange(1,),
                  default=None,
                  help='Predicted sequences will be cut to this length to increase prediction quality '
                       'if overlapping is enabled (default: subsequence_length * 3 / 4)',
                  cls=HelpGroupOption, help_group='Testing/prediction parameters')
    # resources
    @click.option('--float-precision',
                  type=str,
                  default='float32',
                  help='Precision of model weights and biases',
                  cls=HelpGroupOption, help_group='Resource parameters')
    @click.option('--device',
                  type=str,
                  default='gpu',
                  callback=validate_device,
                  help='Device to train/test/predict on (options: gpu or cpu)',
                  cls=HelpGroupOption, help_group='Resource parameters')
    @click.option('--num-devices',
                  type=click.IntRange(1,),
                  default=1,
                  help='Number of devices to use',
                  cls=HelpGroupOption, help_group='Resource parameters')
    @click.option('--workers',
                  type=click.IntRange(0,),
                  default=0,
                  help='Number of threads used to fetch input data for training. Consider '
                       'setting to match the number of GPUs.',
                  cls=HelpGroupOption, help_group='Resource parameters')
    # misc flags
    @click.option('--save-every-check',
                  is_flag=True,
                  help='Add to save a model checkpoint every validation genic F1 check (see '
                       '--check-every-nth-batch in training parameters)',
                  cls=HelpGroupOption, help_group='Miscellaneous parameters')
    # @click.option('--nni', is_flag=True) todo: think about reintegration later
    @click.option('-v', '--verbose',
                  is_flag=True,
                  help='Add to run Helixer in verbosity mode (additional information will be printed)',
                  cls=HelpGroupOption, help_group='Miscellaneous parameters')
    @click.option('--debug',
                  is_flag=True,
                  help='Add to run in debug mode; truncates input data to small example (for training: '
                       'just runs a few epochs)',
                  cls=HelpGroupOption, help_group='Miscellaneous parameters')
    # fine-tuning
    # experimental parameters for training (a) final layer(s)
    # of the model on target species (or other small dataset)
    # with or without coverage, and
    # with the rest of the model weights locked to reduce over fitting
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
