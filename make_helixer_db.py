import argparse

from helixerprep.core.mers import MerController


def main(args):
    controller = MerController(args.db_path_in, args.db_path_out)  # inserts Mer table
    if args.max_k > 0:
        controller.add_mers(args.min_k, args.max_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db_path_in', type=str, required=True,
                    help=('Path to the GeenuFF SQLite input database.'))
    io.add_argument('--db_path_out', type=str, default='',
                    help=('Output path of the new Helixer SQLite database. If not provided '
                          'the input database will be replaced.'))

    fasta_specific = parser.add_argument_group("Controlling the kmer generation:")
    fasta_specific.add_argument('--min_k', help='minimum size kmer to calculate from sequence',
                                default=0, type=int)
    fasta_specific.add_argument('--max_k', help='maximum size kmer to calculate from sequence',
                                default=0, type=int)

    args = parser.parse_args()

    assert args.min_k <= args.max_k, 'min_k can not be greater than max_k'
    if args.max_k > 0 and args.min_k == 0:
        args.min_k = 1
        print('min_k parameter set to 1')

    main(args)
