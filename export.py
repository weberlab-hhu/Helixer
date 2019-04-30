import logging
import argparse
import os

import helixerprep.datas.annotations.slicer as slicer


def main(args):
    if not args.sliced_db_input:
        controller = slicer.SliceController(db_path_in=args.db_path_in,
                                            db_path_sliced=args.db_path_sliced)
        controller.slice_db(train_size=args.train_size,
                            dev_size=args.dev_size,
                            chunk_size=args.chunk_size,
                            seed=args.seed)
    if args.h5_out:
        # encode
        # store in .h5
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db_path_in', type=str, required=True,
                        help=('Path to the GeenuFF SQLite database. If it is an already sliced one '
                              'please provide the --sliced-db-input flag.'))
    io.add_argument('--sliced-db-input', action='store_true')
    io.add_argument('--db_path_sliced', type=str,
                        help=('Output path of the sliced GeenuFF SQLite database if a '
                              'database without a sliced was provided.'))
    io.add_argument('--h5_out', type=str,
                    help=('Output path for encoded data. If not given only the enhanced database '
                          'will be created.'))

    data_split = parser.add_argument_group("Managing the train/dev/test split")
    data_split.add_argument('--train_size', type=float, default=0.8,
                            help='Fraction of the dataset to be used as train set.')
    data_split.add_argument('--dev_size', type=float, default=0.1,
                            help='Fraction of the dataset to be used as dev set.')
    data_split.add_argument('--chunk_size', type=int, default=2000000,
                            help='Size of the chunks each genomic sequence gets cut into.')
    data_split.add_argument('--seed', default='puma',
                            help=('random seed is md5sum(sequence) + this parameter; '
                                  'don\'t change without cause.'))
    args = parser.parse_args()

    assert args.train_size + args.dev_size < 1.0, 'train and dev sizes are too big'
    assert bool(args.db_path_sliced) is not args.sliced_db_input, 'conflicting input params'

    main(args)
