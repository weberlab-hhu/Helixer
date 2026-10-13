#! /usr/bin/env python3
import argparse
import logging.config
from pprint import pformat
from termcolor import colored

from helixer.core.scripts import ExportParameterParser
from helixer.export.exporter import HelixerFastaToH5Controller
from helixer.core.helpers import get_log_dict


if __name__ == '__main__':
    pp = ExportParameterParser(config_file_path='config/fasta2h5_config.yaml')
    pp.io_group.add_argument('--fasta-path', type=str, default=None, required=True,
                             help='Fasta input file for direct FASTA to .h5 file conversion.')
    pp.io_group.add_argument('--species', type=str, default='', required=True,
                             help='Species name. Will be added to the .h5 file.')
    pp.data_group.add_argument('--subsequence-length', type=int,
                               help='Size of the chunks each genomic sequence gets cut into. (Default is 21384.)')
    pp.data_group.add_argument('--write-by', type=int,
                               help='write in super-chunks with this many base pairs, which will be rounded to be '
                                    'divisible by subsequence-length; needs to be equal to or larger than subsequence '
                                    'length; peak memory grows with it, not with the chromosome length. (Default is '
                                    '5_346_000, 250 subsequences of 21384 bp)')
    pp.defaults['subsequence_length'] = 21384
    pp.defaults['write_by'] = 5_346_000
    args = pp.get_args()
    logging.config.dictConfig(get_log_dict())
    logger = logging.getLogger('HelixerLogger')
    logger.info(colored('\nfasta2h5.py config:\n', 'yellow') + f'{pformat(vars(args))}\n')
    controller = HelixerFastaToH5Controller(args.fasta_path, args.h5_output_path)
    controller.export_fasta_to_h5(chunk_size=args.subsequence_length, compression=args.compression,
                                  species=args.species, write_by=args.write_by)
    logger.info(f'\nFinished writing {args.h5_output_path}.')
