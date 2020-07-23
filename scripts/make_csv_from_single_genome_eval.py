#! /usr/bin/env python3

"""Extract the results from a single genome eval. Parses the file given by --log-file-name
for the F1 summary table, extracts the main information and outputs a .csv that can then
be imported elsewhere. Expects every eval to be in a seperate subfolder (like with
the 'trials' folder of nni)."""

import os
import h5py
import numpy as np
import argparse
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('-mf', '--main-folder', type=str, required=True)
parser.add_argument('-lfn', '--log-file-name', type=str, default='eval.log')
parser.add_argument('-i', '--ignore', action='append')
parser.add_argument('-er', '--error-rates', action='store_true')
parser.add_argument('-t', '--type', type=str, default='plants', help='Only used if --error-rates is set')
args = parser.parse_args()

header = ['genome', 'acc_overall', 'f1_ig', 'f1_utr', 'f1_exon', 'f1_intron', 'legacy_f1_cds',
          'sub_genic', 'f1_genic']
if args.error_rates:
    header += ['base_level_error_rate', 'padded_bases_rate', 'sequence_error_rate']
header += ['folder_name']
print(','.join(header))

for sub_folder in os.listdir(args.main_folder):
    sub_folder_path = os.path.join(args.main_folder, sub_folder)
    if not os.path.isdir(sub_folder_path):
        continue
    if args.ignore and sub_folder in args.ignore:
        continue
    log_file_path = os.path.join(sub_folder_path, args.log_file_name)
    if not os.path.exists(log_file_path) or not os.path.getsize(log_file_path) > 0:
        continue

    # get genome name
    parameters = eval(open(os.path.join(sub_folder_path, 'parameter.cfg')).read())
    path = parameters['parameters']['test_data']
    genome = path.split('/')[-2] # this may have to change depending on the folder structure

    if args.error_rates:
        # get sequence error rate
        f = h5py.File(f'/home/felix/Desktop/data/{args.type}/single_genomes/{genome}/h5_data_20k/test_data.h5', 'r')
        n_samples = f['/data/X'].shape[0]
        err = np.array(f['/data/err_samples'])
        n_err_samples = np.count_nonzero(err == True)
        sequence_error_rate = n_err_samples / n_samples

        # get base level error rate (including padding) iterativly to avoid running into memory issues
        sw_dset = f['/data/sample_weights']
        y_dset = f['/data/y']
        step_size = 1000
        n_error_bases = 0
        n_padded_bases = 0
        idxs = np.array_split(np.arange(len(sw_dset)), len(sw_dset) // step_size)
        for slice_idxs in idxs:
            sw_slice = sw_dset[list(slice_idxs)]
            y_slice = y_dset[list(slice_idxs)]
            n_error_bases += np.count_nonzero(sw_slice == 0)
            n_padded_bases += np.count_nonzero(np.all(y_slice == 0, axis=-1))
        base_level_error_rate = n_error_bases / sw_dset.size
        padded_bases_rate = n_padded_bases / sw_dset.size

    # parse metric table
    log_file = open(log_file_path)
    f1_scores = []
    for line in log_file:
        if 'Precision' in line:  # table start
            next(log_file)  # skip line
            for i in range(7):
                line = next(log_file)
                f1_scores.append(line.strip().split('|')[4].strip())
                if i == 3:
                    next(log_file)  # skip line
            break  # stop at the last line of the metric table

    # parse total accuracy
    next(log_file)
    line = next(log_file)
    acc_overall = line.strip().split(' ')[-1]

    # merge everything into one string
    str_rows = [genome, acc_overall] + f1_scores
    if args.error_rates:
        error_rates = [base_level_error_rate, padded_bases_rate, sequence_error_rate]
        str_rows += ['{:.4f}'.format(n) for n in error_rates]
    str_rows += [sub_folder]
    print(','.join(str_rows))
