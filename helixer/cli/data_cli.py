import click
import functools

from helixer.cli.cli_formatter import ColumnHelpFormatter, HelpGroups, HelpGroupOption
from helixer.cli.shared_options import *

click.Context.formatter_class = ColumnHelpFormatter
help_groups = HelpGroups()


def fasta2zarr_parameters(func):
    # IO params
    @fasta_path_option()
    @species_option('Species name for the Zarr output file and any subsequent files. If this file is '
                    'used for prediction the name will be used in the GFF output file gene and feature IDs.')
    @subsequence_length_option('Size of the chunks each genomic sequence gets cut into.')
    @zarr_output_path_option('Zarr output file for the encoded DNA sequence. Must end with ".zarr".')
    # Processing params
    @write_by_option()
    @no_multiprocess_option()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def geenuff2zarr_parameters(func):
    # IO params
    @click.option('--input-db-path',
                  type=click.Path(exists=True),
                  required=True,
                  help='Path to the GeenuFF SQLite input database (has to contain only one genome).',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @zarr_output_path_option('Zarr output file for the encoded GeenuFF database data. Must end with ".zarr"')
    @click.option('--additional',
                  type=str,
                  default=None,
                  help='Outputs the datasets under alternatives/{additional}/ (and checks sort order '
                       'against existing "data" datasets). Use to add e.g. additional annotations from '
                       'Augustus.',
                  cls=HelpGroupOption, help_group=help_groups.io)
    @subsequence_length_option('Length of the subsequences that the model will use at once.')
    # Processing params
    @click.option('--modes',
                  type=str,
                  default='all',
                  help='Either "all", or a comma separated list (no spaces!) with desired members of the '
                       'following {X,y,anno_meta,transitions} that should be exported. This can be '
                       'useful, for instance when skipping transitions (to reduce size/mem) or skipping '
                       'X because you are adding an additional annotation set to an existing file.',
                  cls=HelpGroupOption, help_group=help_groups.proc)
    @write_by_option()
    @no_multiprocess_option()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
