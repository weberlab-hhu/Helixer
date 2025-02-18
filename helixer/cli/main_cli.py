import click
import functools

from helixer.cli.cli_callbacks import *
from helixer.cli.cli_formatter import ColumnHelpFormatter, HelpGroupOption

click.Context.formatter_class = ColumnHelpFormatter

# todo: print out versions of modules used like we already do, maybe still the commit unless we install through pipy?
#  also MAYBE the command that was used, but only maybe
def helixer_main_parameters(func):
    # IO params
    @click.option('--fasta-path',
                  type=click.Path(exists=True),
                  required=True,
                  help='FASTA input file.',
                  cls=HelpGroupOption, help_group='IO parameters')
    @click.option('--gff-output-path',
                  type=str,
                  required=True,
                  callback=validate_path_fragment,
                  help='Output GFF3 file path.',
                  cls=HelpGroupOption, help_group='IO parameters')
    @click.option('--species',
                  type=str,
                  help='Species name.',
                  cls=HelpGroupOption, help_group='IO parameters')
    @click.option('--temporary-dir',
                  type=click.Path(exists=True),
                  help='use supplied (instead of system default) for temporary directory',
                  cls=HelpGroupOption, help_group='IO parameters')
    # Data params
    @click.option('--subsequence-length',
                  type=click.IntRange(1,),
                  default=None,
                  help='How to slice the genomic sequence. Set moderately longer than length of '
                       'typical genic loci. Tested up to 213840. Must be evenly divisible by the '
                       'timestep width of the used model, which is typically 9. (Default is '
                       'lineage dependent from 21384 to 213840).',
                  cls=HelpGroupOption, help_group='Data parameters')
    @click.option('--write-by',
                  type=click.IntRange(1,),
                  default=20_000_000,
                  help='Convert genomic sequence in super-chunks to numerical matrices with this many '
                       'base pairs, which will be rounded to be divisible by subsequence-length; needs '
                       'to be equal to or larger than subsequence length; for lower memory consumption, '
                       'consider setting a lower number',
                  cls=HelpGroupOption, help_group='Data parameters')
    @click.option('--no-multiprocess',
                  is_flag=True,
                  help='Whether to not parallize the numerification of large sequences. Uses half the memory '
                       'but can be much slower when many CPU cores can be utilized.',
                  cls=HelpGroupOption, help_group='Data parameters')
    @click.option('--lineage',
                  type=click.Choice(['vertebrate', 'land_plant', 'fungi', 'invertebrate']),
                  show_choices=True,
                  default=None,
                  help='What model to use for the annotation.',
                  cls=HelpGroupOption, help_group='Data parameters')
    @click.option('--model-filepath',
                  type=click.Path(exists=True),
                  default=None,
                  help='set this to override the default model for any given '
                       'lineage and instead take a specific model',
                  cls=HelpGroupOption, help_group='Data parameters')
    # Prediction params
    @click.option('--batch-size',
                  type=click.IntRange(1,),
                  default=32,
                  help='The batch size for the raw predictions in TensorFlow. Should be as large as '
                       'possible on your GPU to save prediction time. (Default is 8.)',
                  cls=HelpGroupOption, help_group='Prediction parameters')
    @click.option('--no-overlap',
                  is_flag=True,
                  help='Switches off the overlapping after predictions are made. Predictions without'
                       ' overlapping will be faster, but will have lower quality towards '
                       'the start and end of each subsequence. With this parameter --overlap-offset '
                       'and --overlap-core-length will have no effect.',
                  cls=HelpGroupOption, help_group='Prediction parameters')
    @click.option('--overlap-offset',
                  type=click.IntRange(1,),
                  help='Offset of the overlap processing. Smaller values may lead to better '
                       'predictions but will take longer. The subsequence_length should be evenly '
                       'divisible by this value. (Default is subsequence_length / 2).',
                  cls=HelpGroupOption, help_group='Prediction parameters')
    @click.option('--overlap-core-length',
                  type=click.IntRange(1,),
                  help='Predicted sequences will be cut to this length to increase prediction '
                       'quality if overlapping is enabled. Smaller values may lead to better '
                       'predictions but will take longer. Has to be smaller than subsequence_length '
                       '(Default is subsequence_length * 3 / 4)',
                  cls=HelpGroupOption, help_group='Prediction parameters')
    # Post-processing parameters
    @click.option('--window-size',
                  type=click.IntRange(1,),
                  default=100,
                  help='width of the sliding window that is assessed for intergenic vs genic '
                       '(UTR/Coding Sequence/Intron) content',
                  cls=HelpGroupOption, help_group='Post-processing parameters')
    @click.option('--edge-threshold',
                  type=click.FloatRange(0, 1),
                  default=0.1,
                  help='threshold specifies the genic score which defines the start/end boundaries '
                       'of each candidate region within the sliding window',
                  cls=HelpGroupOption, help_group='Post-processing parameters')
    @click.option('--peak-threshold',
                  type=click.FloatRange(0, 1),
                  default=0.8,
                  help='threshold specifies the minimum peak genic score required to accept the '
                       'candidate region; the candidate region is accepted if it contains at least '
                       'one window with a genic score above this threshold',
                  cls=HelpGroupOption, help_group='Post-processing parameters')
    @click.option('--min-coding-length',
                  type=click.IntRange(1,),
                  default=100,
                  help='output is filtered to remove genes with a total coding length shorter '
                       'than this value',
                  cls=HelpGroupOption, help_group='Post-processing parameters')
    @click.version_option()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
