import click
import functools

from helixer.cli.cli_callbacks import *
from helixer.cli.cli_formatter import ColumnHelpFormatter, HelpGroupOption, HelpGroups
from helixer.cli.shared_options import *

click.Context.formatter_class = ColumnHelpFormatter
help_groups = HelpGroups()


# todo: print out versions of modules used like we already do, maybe still the commit unless we install through pipy?
#  also MAYBE the command that was used, but only maybe
# todo Helixer.py check_lineage model as callback
def helixer_main_options(func):
    # IO options
    @fasta_path_option()  # brackets need to be there
    @click.option('--gff-output-path',
                  type=str,
                  required=True,
                  callback=validate_path_fragment,
                  help='Output GFF3 file path.',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @species_option('Species name for the GFF output file gene and feature IDs.')
    @click.option('--temporary-dir',
                  type=click.Path(exists=True),
                  help='use supplied (instead of system default) for temporary directory',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @subsequence_length_option('How to slice the genomic sequence. Set moderately longer than length of '
                               'typical genic loci. Tested up to 213840. Must be evenly divisible by the '
                               'timestep width of the used model, which is typically 9. (Default is '
                               'lineage dependent from 21384 to 213840).')
    # Data options
    @click.option('--lineage',
                  type=click.Choice(['vertebrate', 'land_plant', 'fungi', 'invertebrate']),
                  show_choices=True,
                  default=None,
                  help='What model to use for the annotation.',
                  cls=HelpGroupOption, help_group=help_groups.data)
    @click.option('--model-filepath',
                  type=click.Path(exists=True),
                  default=None,
                  help='set this to override the default model for any given '
                       'lineage and instead take a specific model',
                  cls=HelpGroupOption, help_group=help_groups.data)
    # Processing options
    @write_by_option()
    @no_multiprocess_option()
    # Prediction options
    @batch_size_option('The batch size for the raw predictions (.zarr format). Should be as large as '
                       'possible on your GPU to save prediction time.', help_groups.pred)
    @click.option('--no-overlap',
                  is_flag=True,
                  help='Switches off the overlapping after predictions are made. Predictions without'
                       ' overlapping will be faster, but will have lower quality towards '
                       'the start and end of each subsequence. With this parameter --overlap-offset '
                       'and --overlap-core-length will have no effect.',
                  cls=HelpGroupOption, help_group=help_groups.pred)
    @overlap_offset_option()
    @overlap_core_length_option()
    # Post-processing options
    @click.option('--window-size',
                  type=click.IntRange(1,),
                  default=100,
                  help='width of the sliding window that is assessed for intergenic vs genic '
                       '(UTR/Coding Sequence/Intron) content',
                  cls=HelpGroupOption, help_group=help_groups.post)
    @click.option('--edge-threshold',
                  type=click.FloatRange(0, 1),
                  default=0.1,
                  help='threshold specifies the genic score which defines the start/end boundaries '
                       'of each candidate region within the sliding window',
                  cls=HelpGroupOption, help_group=help_groups.post)
    @click.option('--peak-threshold',
                  type=click.FloatRange(0, 1),
                  default=0.8,
                  help='threshold specifies the minimum peak genic score required to accept the '
                       'candidate region; the candidate region is accepted if it contains at least '
                       'one window with a genic score above this threshold',
                  cls=HelpGroupOption, help_group=help_groups.post)
    @click.option('--min-coding-length',
                  type=click.IntRange(1,),
                  default=100,
                  help='output is filtered to remove genes with a total coding length shorter '
                       'than this value',
                  cls=HelpGroupOption, help_group=help_groups.post)
    @verbose_option()
    @click.version_option()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
