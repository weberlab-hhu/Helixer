#! /usr/bin/env python3
import os
import click
import random
import shutil
import sys
import time
import tempfile
import subprocess
from termcolor import colored

from helixer.cli.main_cli import helixer_main_parameters
from helixer.cli.cli_formatter import HelpGroupCommand
from helixer.core.data import prioritized_models, report_if_current_not_best, identify_current, MODEL_PATH
from helixer.prediction.HybridModel import HybridModel
from helixer.export.exporter import HelixerFastaToZarrController


def check_for_lineage_model(lineage):
    # which models are available?
    priorty_ms = prioritized_models(lineage)
    # which model is already downloaded / will be used?
    current_model = identify_current(lineage, priorty_ms)
    # provide feedback if not up to date, error out if missing
    report_if_current_not_best(priorty_ms, current_model)

    return os.path.join(MODEL_PATH, lineage, current_model)


def check_model_args(model_filepath, lineage, subsequence_length):
    if model_filepath is not None:
        print(f'overriding the lineage based model, '
              f'with the manually specified {model_filepath}', file=sys.stderr)
        model_filepath = model_filepath
        msg = 'when manually specifying a model, you must also specify a --subsequence-length appropriate for ' \
              'your target lineage. This should be more than long enough to easily contain most ' \
              'genomic loci. E.g. 21384, 64152, or 213840 for fungi, plants, and animals, respectively.'
        assert subsequence_length is not None, msg
    else:
        assert lineage is not None, ("Either --lineage or --model-filepath is required. Run `Helixer.py "
                                     "--help` to see lineage options.")
        model_filepath = check_for_lineage_model(lineage)
        if subsequence_length is None:
            key = {'vertebrate': 213840, 'land_plant': 64152, 'fungi': 21384, 'invertebrate': 213840}
            subsequence_length = key[lineage]

    return model_filepath, subsequence_length


def check_overlap_args(subsequence_length, overlap_offset, overlap_core_length):
    # check user params are valid or set defaults relative to subsequence_length
    # set overlap parameters no matter if overlap is used or not to prevent HelixerModel
    # from throwing an argparse type error when None gets passed as an argument
    if overlap_offset is not None:
        msg = (f'the given --overlap-offset of {overlap_offset} has to evenly '
               f'divide --subsequence-length, which is {subsequence_length}')
        assert subsequence_length % overlap_offset == 0, msg
    else:
        overlap_offset = subsequence_length // 2

    if overlap_core_length is not None:
        msg = (f'the given --overlap-core-length of {overlap_core_length} has to be smaller '
               f'than --subsequence-length, which is {subsequence_length}')
        assert subsequence_length > overlap_core_length, msg
    else:
        overlap_core_length = int(subsequence_length * 3 / 4)
    return overlap_offset, overlap_core_length


@click.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@helixer_main_parameters
def main(fasta_path, gff_output_path, species, temporary_dir, subsequence_length, write_by, no_multiprocess,
         lineage, model_filepath, batch_size, no_overlap, overlap_offset, overlap_core_length, window_size,
         edge_threshold, peak_threshold, min_coding_length):
    """Use Helixer for structural genome annotation"""
    model_filepath, subsequence_length = check_model_args(model_filepath, lineage, subsequence_length)
    helixer_post_bin = 'helixer_post_bin'
    start_time = time.time()
    # pp = HelixerParameterParser('config/helixer_config.yaml') todo: add config stuff to cli
    overlap = not no_overlap  # minor overlapping is a far better default for inference. Thus, this hack.
    if overlap:
        overlap_offset, overlap_core_length = check_overlap_args(subsequence_length, overlap_offset,
                                                                 overlap_core_length)
    # before we start, check if helixer_post_bin will (presumably) be able to run
    # first, is it there
    print(colored(f'Testing whether {helixer_post_bin} is correctly installed', 'green'))
    if not shutil.which(helixer_post_bin):
        print(colored(f'\nError: {helixer_post_bin} not found in $PATH, this is required for Helixer.py to complete.\n',
                      'red'),
              file=sys.stderr)
        print('Installation instructions: https://github.com/TonyBolger/HelixerPost, the lzf library is OPTIONAL',
              file=sys.stderr)
        print('Remember to add the compiled binary to a folder in your PATH variable.')
        sys.exit(1)
    else:
        run = subprocess.run([helixer_post_bin])
        # we should get the help function and return code 1 if helixer_post_bin is fully installed
        if run.returncode != 1:
            # if we get anything else, subprocess will have shown the error (I hope),
            # so just exit with the return code
            sys.exit(run.returncode)
    # second, are we able to write the (or rather a dummy) output file to the directory
    out_dir = os.path.dirname(gff_output_path)
    out_dir = os.path.join(out_dir, '.')  # dodges checking existence of directory '' for the current directory
    test_file = f'{gff_output_path}.{random.getrandbits(128)}'
    try:
        assert not os.path.exists(test_file), "this is a randomly generated test write, why is something here...?"
        with open(test_file, 'w'):
            pass
        os.remove(test_file)
    except Exception as e:
        print(colored(f'checking if a random test file ({test_file}) can be written in the output directory', 'yellow'))
        if not os.path.isdir(out_dir):
            # the 'file not found error' for the directory when the user is thinking
            # "of course it's not there, I want to crete it"
            # tends to confuse..., so make it obvious here
            print(colored(f'the output directory {out_dir}, needed to write the '
                  f'output file {gff_output_path}, is absent, inaccessible, or not a directory', 'red'))
        raise e

    print(colored('Helixer.py config loaded. Starting FASTA to H5 conversion.', 'green'))
    # generate the .h5 file in a temp dir, which is then deleted
    with tempfile.TemporaryDirectory(dir=temporary_dir) as tmp_dirname:
        print(f'storing temporary files under {tmp_dirname}')
        tmp_genome_zarr_path = os.path.join(tmp_dirname, f'tmp_species_{species}.zarr')
        tmp_pred_zarr_path = os.path.join(tmp_dirname, f'tmp_predictions_{species}.zarr')

        controller = HelixerFastaToZarrController(fasta_path, tmp_genome_zarr_path)
        # hard coded subsequence length due to how the models have been created
        controller.export_fasta_to_zarr(chunk_size=subsequence_length, multiprocess=not no_multiprocess,
                                        species=species, write_by=write_by)

        msg = 'with' if overlap else 'without'
        msg = 'FASTA to Zarr conversion done. Starting neural network prediction ' + msg + ' overlapping.'
        print(colored(msg, 'green'))

        hybrid_model_args = [
            '--verbose',
            '--load-model-path', model_filepath,
            '--test-data', tmp_genome_zarr_path,
            '--prediction-output-path', tmp_pred_zarr_path,
            '--val-test-batch-size', str(batch_size),
            '--overlap-offset', str(overlap_offset),
            '--core-length', str(overlap_core_length)
        ]
        if overlap:
            hybrid_model_append('--overlap')
        model = HybridModel(cli_args=hybrid_model_args)
        model.run()

        print(colored('Neural network prediction done. Starting post processing.', 'green'))

        # call to HelixerPost, has to be in PATH
        helixerpost_cmd = [helixer_post_bin, tmp_genome_zarr_path, tmp_pred_zarr_path]
        helixerpost_params = [window_size, edge_threshold, peak_threshold, min_coding_length]
        helixerpost_cmd += [str(e) for e in helixerpost_params] + [gff_output_path]

        helixerpost_out = subprocess.run(helixerpost_cmd)
        if helixerpost_out.returncode == 0:
            run_time = time.time() - start_time
            print(colored(f'\nHelixer successfully finished the annotation of {fasta_path} '
                          f'in {run_time / (60 * 60):.2f} hours. '
                          f'GFF file written to {gff_output_path}.', 'green'))
        else:
            print(colored('\nAn error occurred during post processing. Exiting.', 'red'))


if __name__ == '__main__':
    main()
