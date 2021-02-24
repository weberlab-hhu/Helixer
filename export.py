#! /usr/bin/env python3
import argparse
from pprint import pprint

from helixer.export.exporter import HelixerExportController


def main(args):
    if args.genomes != '':
        args.genomes = args.genomes.split(',')

    if args.modes == 'all':
        modes = ('X', 'y', 'anno_meta', 'transitions')  # todo, only X does anything atm
    else:
        modes = tuple(args.modes.split(','))

    if args.add_additional is not None:
        match_existing = True
        h5_group = '/alternative/' + args.add_additional + '/'
    else:
        match_existing = False
        h5_group = '/data/'

    controller = HelixerExportController(args.main_db_path, args.out_dir, args.only_test_set,
                                         match_existing=match_existing, h5_group=h5_group)
    controller.export(chunk_size=args.chunk_size, genomes=args.genomes, val_size=args.val_size,
                      keep_featureless=args.export_featureless, write_by=args.write_by, modes=modes,
                      multiprocess=not args.no_multiprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--main-db-path', type=str, required=True,
                    help='Path to the directory with GeenuFF SQLite input databases (has to be one per genome).')
    io.add_argument('--out-dir', type=str, required=True, help='Output dir for encoded data files.')
    io.add_argument('--add-additional', type=str,
                    help='outputs the datasets under alternatives/{add-additional}/ (and checks sort order against '
                         'existing "data" datasets). Use to add e.g. additional annotations from Augustus.')

    genomes = parser.add_argument_group("Genome selection")
    genomes.add_argument('--genomes', type=str, default='', required=True,
                         help=('Comma seperated list of species names to be exported. Each species must be in a '
                               'seperate GeenuFF database inside --main-db-path, and have the name {genome_name}.sqlite3.'))

    data = parser.add_argument_group("Data generation parameters")
    data.add_argument('--chunk-size', type=int, default=20000,
                      help='Size of the chunks each genomic sequence gets cut into.')
    data.add_argument('--val-size', type=float, default=0.2,
                      help='The chance for a sequence or coordinate to end up in validation_data.h5' )
    data.add_argument('--only-test-set', action='store_true',
                      help='Whether to only output a single file named test_data.h5')
    data.add_argument('--export-featureless', action='store_true',
                      help='This overrides the default behavior of ignoring coordinates without a single feature (as '
                           'these frequently were never actually annotated). Anyways generates a "data/is_annotated" '
                           'which can mask chunks from featureless coordinates that would have been skipped')
    data.add_argument('--modes', default='all',
                      help='either "all" (default), or a comma separated list with desired members of the following '
                           '{X, y, anno_meta, transitions} that should be exported. This can be useful, for '
                           'instance when skipping transitions (to reduce size/mem) or skipping X because '
                           'you are adding an additional annotation set to an existing file.')
    data.add_argument('--write-by', type=int, default=10_000_000_000,
                      help='write in super-chunks with this many bp, '
                           'must be divisible by chunk-size')
    data.add_argument('--no-multiprocess', action='store_true',
                      help='Whether to parallize numerification of large sequences. Uses 2x the memory.')

    args = parser.parse_args()
    print('Export config:')
    pprint(vars(args))
    print()

    main(args)
