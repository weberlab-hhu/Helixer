#! /usr/bin/env python3
import argparse
from pprint import pprint

from helixer.export.exporter import HelixerExportController, HelixerFastaToH5Controller


def main(args):
    if args.genomes != '':
        genomes = args.genomes.split(',')

    if args.modes == 'all':
        modes = ('X', 'y', 'anno_meta', 'transitions')
    else:
        modes = tuple(args.modes.split(','))

    if args.add_additional:
        match_existing = True
        h5_group = '/alternative/' + args.add_additional + '/'
    else:
        match_existing = False
        h5_group = '/data/'

    if args.main_db_path:
        write_by = round(args.write_by / args.chunk_size) * args.chunk_size
        controller = HelixerExportController(args.main_db_path, args.output_path, match_existing=match_existing,
                                             h5_group=h5_group)
        controller.export(chunk_size=args.chunk_size, genomes=genomes, write_by=write_by, modes=modes,
                          compression=args.compression, multiprocess=not args.no_multiprocess)
    else:
        controller = HelixerFastaToH5Controller(args.direct_fasta_to_h5_path, args.output_path)
        controller.export_fasta_to_h5(chunk_size=args.chunk_size, genome=genomes[0],
                                      compression=args.compression, multiprocess=not args.no_multiprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--main-db-path', type=str, default=None,
                    help='Path to the directory with GeenuFF SQLite input databases (has to be one per genome).')
    io.add_argument('--direct-fasta-to-h5-path', type=str, default=None,
                    help='Directly convert from a FASTA file to .h5, circumventing import into a Geenuff database')
    io.add_argument('--output-path', type=str, required=True, help='Output file for the encoded data. Must end with ".h5"')
    io.add_argument('--add-additional', type=str, default='',
                    help='outputs the datasets under alternatives/{add-additional}/ (and checks sort order against '
                         'existing "data" datasets). Use to add e.g. additional annotations from Augustus.')

    genomes = parser.add_argument_group("Genome selection")
    genomes.add_argument('--genomes', type=str, default='', required=True,
                         help=('Comma seperated list of species names to be exported. Needs to be one name if '
                               '--direct-fasta-to-h5-path. If --main-db-path is set, each species must be in a '
                               'seperate GeenuFF database inside --main-db-path, and have the name {genome_name}.sqlite3.'))

    data = parser.add_argument_group("Data generation parameters")
    data.add_argument('--chunk-size', type=int, default=20000,
                      help='Size of the chunks each genomic sequence gets cut into.')
    data.add_argument('--modes', default='all',
                      help='either "all" (default), or a comma separated list with desired members of the following '
                           '{X, y, anno_meta, transitions} that should be exported. This can be useful, for '
                           'instance when skipping transitions (to reduce size/mem) or skipping X because '
                           'you are adding an additional annotation set to an existing file.')
    data.add_argument('--write-by', type=int, default=10_000_000_000,
                      help='write in super-chunks with this many bp, '
                           'will be rounded to be divisible by chunk-size')
    data.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf'],
                      help='Compression algorithm used for the .h5 output files compression level is set as 4.')
    data.add_argument('--no-multiprocess', action='store_true',
                      help='Whether to parallize numerification of large sequences. Uses 2x the memory.')
    args = parser.parse_args()

    assert bool(args.main_db_path) ^ bool(args.direct_fasta_to_h5_path), 'need either --main-db-path or --direct-fasta-to-h5-path'
    assert args.output_path.endswith('.h5'), '--output-path must end with ".h5"'
    if args.direct_fasta_to_h5_path:
        assert not args.add_additional, '--direct-fasta-to-h5-path can not be used with --add-additional at the moment'
        assert len(args.genomes.split(',')) == 1, 'need exactly one genome name with --direct-fasta-to-h5-path'

    print('Export config:')
    pprint(vars(args))
    print()

    main(args)
