#! /usr/bin/env python3
import argparse

from helixerprep.core.controller import HelixerController


def main(args):
    # insert additional tables
    controller = HelixerController(args.db_path_in, args.db_path_out, args.meta_info_root_path)
    # lookup kmers and add what we find
    controller.add_mer_counts_to_db()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db-path-in', type=str, required=True,
                    help=('Path to the GeenuFF SQLite input database.'))
    io.add_argument('--db-path-out', type=str, default='',
                    help=('Output path of the new Helixer SQLite database. If not provided '
                          'the input database will be replaced.'))

    fasta_specific = parser.add_argument_group("Controlling the kmer lookup:")
    fasta_specific.add_argument('--meta-info-root-path', type=str,
                                help=('Absolute folder path from where the kmers files are in the '
                                      'subfolder {species}/meta_collection/kmers/kmers.tsv'),
                                default='/mnt/data/ali/share/phytozome_organized/ready/train')
    args = parser.parse_args()
    main(args)
