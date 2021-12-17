#! /usr/bin/env python3
import os
import time
import h5py
import tempfile
import subprocess
from termcolor import colored

from helixer.core.scripts import ParameterParser
from helixer.prediction.HybridModel import HybridModel
from helixer.export.exporter import HelixerExportController, HelixerFastaToH5Controller


class HelixerParameterParser(ParameterParser):
    def __init__(self, config_file_path=''):
        super().__init__(config_file_path)
        self.io_group.add_argument('--fasta-path', type=str, required=True, help='FASTA input file.')
        self.io_group.add_argument('--gff-output-path', type=str, required=True, help='Output GFF file path.')
        self.io_group.add_argument('--species', type=str, help='Species name.')

        self.data_group.add_argument('--chunk-input-len', type=int,
                                     help='How to chunk up the genomic sequence. Should grow with expected average '
                                          'gene length until up to roughly 200000. Has to be evenly divisible by '
                                          'the block size of the used model, which is 9 at the moment. (Default is 19440.)')
        self.data_group.add_argument('--species-category', type=str, choices=['vertebrate', 'land_plant', 'fungi'],
                                     help='What model to use for the annotation. (Default is "vertebrate".)')

        self.pred_group = self.parser.add_argument_group("Prediction parameters")
        self.pred_group.add_argument('--batch-size', type=int,
                                     help='The batch size for the raw predictions in TensorFlow. Should be as large as '
                                          'possible to save prediction time. (Default is 8.)')
        self.data_group.add_argument('--overlap', action='store_true',
                                     help='Switches on the overlapping after predictions are made. Predictions with '
                                          'overlapping will take much longer, but will have better quality towards '
                                          'the start and end of each subsequence. Without this parameter --overlap-offset '
                                          'and --overlap-core-length will have no effect.')
        self.pred_group.add_argument('--overlap-offset', type=int,
                                     help='Offset of the overlap processing. Smaller values may lead to better '
                                          'predictions but will take longer. --chunk-input-len has to be evenly '
                                          'divisible by this value. (Default is 3240.)')
        self.pred_group.add_argument('--overlap-core-length', type=int,
                                     help='Predicted sequences will be cut to this length to increase prediction '
                                          'quality if overlapping is enabled. Smaller values may lead to better '
                                          'predictions but will take longer. Has to be smaller than --chunk-input-len. '
                                          '(Default is 10000.)')

        self.post_group = self.parser.add_argument_group("Post processing parameters")
        self.post_group.add_argument('--window-size', type=int, help='')
        self.post_group.add_argument('--edge-threshold', type=float, help='')
        self.post_group.add_argument('--peak-threshold', type=float, help='')
        self.post_group.add_argument('--min-coding-length', type=int, help='')

        helixer_defaults = {
            'fasta_path': '',
            'species': '',
            'chunk_input_len': 19440,
            'species_category': 'vertebrate',
            'batch_size': 8,
            'overlap': False,
            'overlap_offset': 3240,
            'overlap_core_length': 10000,
            'window_size': 100,
            'edge_threshold': 0.1,
            'peak_threshold': 0.8,
            'min_coding_length': 100,
        }
        self.defaults = {**self.defaults, **helixer_defaults}

    def check_args(self, args):
        self.model_filepath = os.path.join('models', f'{args.species_category}.h5')
        assert os.path.isfile(self.model_filepath), f'{self.model_filepath} does not exists'

        # check if model block size fits the chunk input length (has to be evenly divisible)
        with h5py.File(self.model_filepath, 'r') as model:
            model_block_size = model['/model_weights/dense_1/dense_1/bias:0'].shape[0] // 8
            msg = (f'chunk input length (currently {args.chunk_input_len}) '
                   f'has to be evenly divisible by {model_block_size}')
            assert args.chunk_input_len % model_block_size == 0, msg

        if args.overlap:
            msg = '--overlap-offset has to evenly divide --chunk-input-len'
            assert args.chunk_input_len % args.overlap_offset == 0, msg
            msg = '--overlap-core-length has to be smaller than --chunk-input-len'
            assert args.chunk_input_len > args.overlap_core_length, msg


if __name__ == '__main__':
    start_time = time.time()
    pp = HelixerParameterParser('config/helixer_config.yaml')
    args = pp.get_args()
    print(colored('helixer.py config loaded. Starting FASTA to H5 conversion.', 'green'))

    # generate the .h5 file in a temp dir, which is then deleted
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_genome_h5_path = os.path.join(tmp_dirname, f'tmp_species_{args.species}.h5')
        tmp_pred_h5_path = os.path.join(tmp_dirname, f'tmp_predictions_{args.species}.h5')

        controller = HelixerFastaToH5Controller(args.fasta_path, tmp_genome_h5_path)
        # hard coded chunk size due to how the models have been created
        controller.export_fasta_to_h5(chunk_size=args.chunk_input_len, compression=args.compression,
                                      multiprocess=not args.no_multiprocess, species=args.species)

        msg = 'with' if args.overlap else 'without'
        msg = 'FASTA to H5 conversion done. Starting neural network prediction ' + msg + ' overlapping.'
        print(colored(msg, 'green'))
        # hard coded model dir path, probably not optimal
        model_filepath = os.path.join('models', f'{args.species_category}.h5')
        assert os.path.isfile(model_filepath), f'{model_filepath} does not exists'

        hybrid_model_args = [
            '--verbose',
            '--load-model-path', model_filepath,
            '--test-data', tmp_genome_h5_path,
            '--prediction-output-path', tmp_pred_h5_path,
            '--val-test-batch-size', str(args.batch_size),
            '--overlap-offset', str(args.overlap_offset),
            '--core-length', str(args.overlap_core_length),
        ]
        if args.overlap:
            hybrid_model_args.append('--overlap')
        model = HybridModel(cli_args=hybrid_model_args)
        model.run()

        print(colored('Neural network prediction done. Starting post processing.', 'green'))

        # call to HelixerPost, has to be in PATH
        helixerpost_cmd = ['helixer_post_bin', tmp_genome_h5_path, tmp_pred_h5_path]
        helixerpost_params = [args.window_size, args.edge_threshold, args.peak_threshold, args.min_coding_length]
        helixerpost_cmd += [str(e) for e in helixerpost_params] + [args.gff_output_path]

        helixerpost_out = subprocess.run(helixerpost_cmd)
        if helixerpost_out.returncode == 0:
            run_time = time.time() - start_time
            print(colored(f'\nHelixer successfully finished the annotation of {args.fasta_path} '
                          f'in {run_time / (60 * 60):.2f} hours. '
                          f'GFF file written to {args.gff_output_path}.', 'green'))
        else:
            print(colored('\nAn error occured during post processing. Exiting.', 'red'))

