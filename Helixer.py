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

        self.data_group.add_argument('--subsequence-length', type=int,
                                     help='How to slice the genomic sequence. Set moderately longer than length of '
                                          'typical genic loci. Tested up to 200000. Must be evenly divisible by the '
                                          'timestep width of the used model, which is typically 9. (Default is 21384.)')
        self.data_group.add_argument('--lineage', type=str, choices=['vertebrate', 'land_plant', 'fungi'],
                                     help='What model to use for the annotation. (Default is "land_plant".)')

        self.pred_group = self.parser.add_argument_group("Prediction parameters")
        self.pred_group.add_argument('--batch-size', type=int,
                                     help='The batch size for the raw predictions in TensorFlow. Should be as large as '
                                          'possible to save prediction time. (Default is 8.)')
        self.data_group.add_argument('--no-overlap', action='store_true',
                                     help='Switches off the overlapping after predictions are made. Predictions without'
                                          ' overlapping will be faster, but will have lower quality towards '
                                          'the start and end of each subsequence. With this parameter --overlap-offset '
                                          'and --overlap-core-length will have no effect.')
        self.pred_group.add_argument('--overlap-offset', type=int,
                                     help='Offset of the overlap processing. Smaller values may lead to better '
                                          'predictions but will take longer. --chunk-input-len has to be evenly '
                                          'divisible by this value. (Default is 10692.)')
        self.pred_group.add_argument('--overlap-core-length', type=int,
                                     help='Predicted sequences will be cut to this length to increase prediction '
                                          'quality if overlapping is enabled. Smaller values may lead to better '
                                          'predictions but will take longer. Has to be smaller than --chunk-input-len. '
                                          '(Default is 16038.)')
        self.pred_group.add_argument('--debug', action='store_true', help='add this to quickly the code runs through'
                                                                          'without loading/predicting on the full file')

        self.post_group = self.parser.add_argument_group("Post processing parameters")
        self.post_group.add_argument('--window-size', type=int, help='')
        self.post_group.add_argument('--edge-threshold', type=float, help='')
        self.post_group.add_argument('--peak-threshold', type=float, help='')
        self.post_group.add_argument('--min-coding-length', type=int, help='')

        helixer_defaults = {
            'fasta_path': '',
            'species': '',
            'subsequence_length': 21384,
            'lineage': 'land_plant',
            'batch_size': 32,
            'no_overlap': False,
            'debug': False,
            'overlap_offset': 10692,
            'overlap_core_length': 16038,
            'window_size': 100,
            'edge_threshold': 0.1,
            'peak_threshold': 0.8,
            'min_coding_length': 100,
        }
        self.defaults = {**self.defaults, **helixer_defaults}

    def check_args(self, args):
        self.model_filepath = os.path.join('models', f'{args.lineage}.h5')
        assert os.path.isfile(self.model_filepath), f'{self.model_filepath} does not exists'

        # check if model timestep width fits the subsequence length (has to be evenly divisible)
        with h5py.File(self.model_filepath, 'r') as model:
            # todo, safer way to find this
            try:
                timestep_width = model['/model_weights/dense_1/dense_1/bias:0'].shape[0] // 8
            except KeyError:
                try:
                    timestep_width = model['/model_weights/dense/dense/bias:0'].shape[0] // 8
                except KeyError:
                    print("WARNING could not parse timestep width from model, assuming it is 9")
                    timestep_width = 9
            msg = (f'subsequence length (currently {args.subsequence_length}) '
                   f'has to be evenly divisible by {timestep_width}')
            assert args.subsequence_length % timestep_width == 0, msg

        if not args.no_overlap:
            msg = '--overlap-offset has to evenly divide --subsequence-length'
            assert args.subsequence_length % args.overlap_offset == 0, msg
            msg = '--overlap-core-length has to be smaller than --subseqeunce-length'
            assert args.subsequence_length > args.overlap_core_length, msg


if __name__ == '__main__':
    start_time = time.time()
    pp = HelixerParameterParser('config/helixer_config.yaml')
    args = pp.get_args()
    args.overlap = not args.no_overlap  # minor overlapping is a far better default for inference. Thus, this hack.
    print(colored('Helixer.py config loaded. Starting FASTA to H5 conversion.', 'green'))

    # generate the .h5 file in a temp dir, which is then deleted
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_genome_h5_path = os.path.join(tmp_dirname, f'tmp_species_{args.species}.h5')
        tmp_pred_h5_path = os.path.join(tmp_dirname, f'tmp_predictions_{args.species}.h5')

        controller = HelixerFastaToH5Controller(args.fasta_path, tmp_genome_h5_path)
        # hard coded subsequence length due to how the models have been created
        controller.export_fasta_to_h5(chunk_size=args.subsequence_length, compression=args.compression,
                                      multiprocess=not args.no_multiprocess, species=args.species)

        msg = 'with' if args.overlap else 'without'
        msg = 'FASTA to H5 conversion done. Starting neural network prediction ' + msg + ' overlapping.'
        print(colored(msg, 'green'))
        # hard coded model dir path, probably not optimal
        model_filepath = os.path.join('models', f'{args.lineage}.h5')
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
        if args.debug:
            hybrid_model_args.append('--debug')
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

