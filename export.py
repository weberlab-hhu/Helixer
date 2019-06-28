#! /usr/bin/env python3
import argparse

from helixerprep.export.exporter import ExportController


def main(args):
    controller = ExportController(args.db_path_in, args.out_dir, args.only_test_set)
    if args.genomes != '':
        args.genomes = args.genomes.split(',')
    controller.export(chunk_size=args.chunk_size, genomes=args.genomes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db-path-in', type=str, required=True,
                    help='Path to the Helixer SQLite input database.')
    io.add_argument('--out-dir', type=str, required=True, help='Output dir for encoded data files.')

    data = parser.add_argument_group("Data generation parameters")
    data.add_argument('--chunk-size', type=int, default=10000,
                      help='Size of the chunks each genomic sequence gets cut into.')
    data.add_argument('--genomes', type=str, default='',
                      help=('Comma seperated list of species names to be exported. '
                            'If empty all genomes in the db are used.'))
    data.add_argument('--only-test-set', action='store_true',
                      help='Whether to only output a single file named test_data.h5')
    args = parser.parse_args()
    main(args)
