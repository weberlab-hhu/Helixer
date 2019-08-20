#! /usr/bin/env python3
import argparse
from pprint import pprint

from helixerprep.export.exporter import ExportController


def main(args):
    controller = ExportController(args.db_path_in, args.out_dir, args.only_test_set)

    if args.genomes != '':
        args.genomes = args.genomes.split(',')
    if args.exclude_genomes != '':
        args.exclude_genomes = args.exclude_genomes.split(',')

    controller.export(chunk_size=args.chunk_size, genomes=args.genomes, exclude=args.exclude_genomes,
                      val_size=args.val_size, one_hot=args.one_hot, merge_introns=args.merge_introns,
                      split_coordinates=args.split_coordinates, keep_errors=args.keep_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db-path-in', type=str, required=True,
                    help='Path to the Helixer SQLite input database.')
    io.add_argument('--out-dir', type=str, required=True, help='Output dir for encoded data files.')

    genomes = parser.add_argument_group("Genome selection")
    genomes.add_argument('--genomes', type=str, default='',
                         help=('Comma seperated list of species names to be exported. '
                               'If empty all genomes in the db are used.'))
    genomes.add_argument('--exclude-genomes', type=str, default='',
                         help=('Comma seperated list of species names to be excluded. '
                               'Can only be used when --genomes is empty'))

    data = parser.add_argument_group("Data generation parameters")
    data.add_argument('--chunk-size', type=int, default=10000,
                      help='Size of the chunks each genomic sequence gets cut into.')
    data.add_argument('--val-size', type=float, default=0.2,
                      help='The chance for a sequence or coordinate to end up in validation_data.h5' )
    data.add_argument('--one-hot', action='store_true',
                      help='Whether to use a one-hot encoding instead of multi class output.')
    data.add_argument('--merge-introns', action='store_true',
                      help=('When using one-hot encoding, whether to put coding and non-coding '
                            'introns into one class.'))
    data.add_argument('--split-coordinates', action='store_true',
                      help='Whether to split on the level of coordinates instead of sequences.')
    data.add_argument('--only-test-set', action='store_true',
                      help='Whether to only output a single file named test_data.h5')
    data.add_argument('--keep_errors', action="store_true",
                      help="Set this flag if entirely erroneous sequences should _not_ be excluded")

    args = parser.parse_args()
    assert not (args.genomes and args.exclude_genomes), 'Can not include and exclude together'
    assert not (args.merge_introns and not args.one_hot)
    pprint(vars(args))
    print()

    main(args)
