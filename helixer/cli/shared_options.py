import click

from helixer.cli.cli_formatter import HelpGroups, HelpGroupOption
from helixer.cli.cli_callbacks import *

help_groups = HelpGroups()


# I/O options
# ----------------------------------------------------------------------
def fasta_path_option():
    return click.option('--fasta-path',
                        type=click.Path(exists=True),
                        required=True,
                        help='FASTA input file.',
                        cls=HelpGroupOption, help_group=help_groups.io)


def subsequence_length_option(help_text):
    return click.option('--subsequence-length',
                        type=click.IntRange(1,),
                        default=21384,
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_groups.io)


def species_option(help_text):
    return click.option('--species',
                        type=str,
                        required=True,
                        default='',
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_groups.io)


def zarr_output_path_option(help_text):
    return click.option('--zarr-output-path',
                        type=str,
                        required=True,
                        callback=combine_callbacks(validate_path_fragment, validate_file_extension),
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_groups.io)


# Resource options
# ----------------------------------------------------------------------
def float_precision_option():
    return click.option('--float-precision',
                        type=str,
                        default='float32',
                        help='Precision of model weights and biases',
                        cls=HelpGroupOption, help_group=help_groups.resource)


def device_option():
    return click.option('--device',
                        type=click.Choice(['gpu', 'cpu']),
                        show_choices=True,
                        default='gpu',
                        callback=validate_device,
                        help='Device to train/test/predict on (options: gpu or cpu)',
                        cls=HelpGroupOption, help_group=help_groups.resource)


def num_devices_option():
    return click.option('--num-devices',
                        type=click.IntRange(1,),
                        default=1,
                        help='Number of devices to use',
                        cls=HelpGroupOption, help_group=help_groups.resource)


def workers_option(help_text):
    return click.option('--workers',
                        type=click.IntRange(0,),
                        default=0,
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_groups.resource)


# Processing options
# ----------------------------------------------------------------------
def write_by_option():
    return click.option('--write-by',
                        type=click.IntRange(1,),
                        default=21_384_000,
                        help='Convert genomic sequence in super-chunks to numerical matrices with this many '
                             'base pairs, which will be rounded to be divisible by subsequence-length; needs '
                             'to be equal to or larger than subsequence length; for lower memory consumption, '
                             'consider setting a lower number',
                        cls=HelpGroupOption, help_group=help_groups.proc)


def no_multiprocess_option():
    return click.option('--no-multiprocess',
                        is_flag=True,
                        help='Whether to not parallize the numerification of large sequences. Uses half the memory '
                             'but can be much slower when many CPU cores can be utilized.',
                        cls=HelpGroupOption, help_group=help_groups.proc)


# Training/Test/Prediction options
# ----------------------------------------------------------------------
# todo ckpt format soon
def load_model_path_option(help_group):
    return click.option('-l', '--load-model-path',
                        type=click.Path(exists=True),
                        default=None,
                        help='Path to a trained/pretrained model checkpoint (HDF5 format)',
                        cls=HelpGroupOption, help_group=help_group)  # group depends on context (train/test/predict)


def batch_size_option(help_text, help_group):
    return click.option('--batch-size',
                        type=click.IntRange(1,),
                        default=32,
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_group)  # group depends on context (train/test/predict)


def overlap_option(help_text, help_group):
    return click.option('--overlap',
                        is_flag=True,
                        help=help_text,
                        cls=HelpGroupOption, help_group=help_group)


def overlap_offset_option(help_group):
    return click.option('--overlap-offset',
                        type=click.IntRange(1,),
                        help='Offset of the overlap processing. Smaller values may lead to better '
                             'predictions but will take longer. The subsequence_length should be evenly '
                             'divisible by this value. (Default is subsequence_length / 2).',
                        cls=HelpGroupOption, help_group=help_group)


def overlap_core_length_option(help_group):
    return click.option('--overlap-core-length',
                        type=click.IntRange(1,),
                        help='Predicted sequences will be cut to this length to increase prediction '
                             'quality if overlapping is enabled. Smaller values may lead to better '
                             'predictions but will take longer. Has to be smaller than subsequence_length '
                             '(Default is subsequence_length * 3 / 4)',
                        cls=HelpGroupOption, help_group=help_group)


# Misc options options
# ----------------------------------------------------------------------
def verbose_option():
    return click.option(
        '-v', '--verbose',
        is_flag=True,
        help='Add to run Helixer in verbosity mode (additional information will be printed)',
        cls=HelpGroupOption, help_group=help_groups.misc
    )
# todo: debug param missing, will be implemented later
