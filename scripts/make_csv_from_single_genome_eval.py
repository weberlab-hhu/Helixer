#! /usr/bin/env python3
import os
import h5py
import numpy as np
import argparse
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', type=str, default='clc')
parser.add_argument('-nni', '--nni-id', type=str, required=True)
parser.add_argument('-i', '--ignore', action='append')
args = parser.parse_args()

assert args.server in ['clc', 'cluster']
assert len(args.nni_id) == 8

if args.server == 'clc':
    nni_base = '/mnt/data/experiments_backup/nni_clc_server/nni/experiments/'
else:
    nni_base = '/mnt/data/experiments_backup/nni_cluster/nni/experiments/'

trials_folder = '{}/{}/trials'.format(nni_base, args.nni_id)
print(','.join(['genome', 'loss', 'acc_overall', 'f1_ig', 'f1_utr', 'f1_exon', 'f1_intron',
                'f1_cds', 'f1_genic', 'old_f1_cds_1', 'base_level_error_rate', 'padded_bases_rate',
                'sequence_error_rate', 'nni_id']))
for folder in os.listdir(trials_folder):
    if folder in args.ignore:
        continue
    # get genome name
    parameters = eval(open('{}/{}/parameter.cfg'.format(trials_folder, folder)).read())
    path = parameters['parameters']['test_data']
    # genome = path.split('/')[5]  # when from cluster
    genome = path.split('/')[6]

    # get sequence error rate
    f = h5py.File('/home/felix/Desktop/data/single_genomes/' + genome + '/h5_data_20k/test_data.h5')
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

    log_file = open('{}/{}/trial.log'.format(trials_folder, folder))
    # get confusion matrix from log to calculate cds f1 that is the same as before
    for line in log_file:
        if 'array([[' in line:  # cm table start
            cm_str = line.split('array(')[1].strip()
            for i in range(3):
                cm_str += next(log_file).strip()
            cm_str = cm_str[:-1]  # remove last round bracket
            break
    cm = np.array(eval(cm_str))
    # change cm so we can get a comparable metric, merging intron and exon predictions
    cm[2, :] = cm[2, :] + cm[3, :]
    cm[:, 2] = cm[:, 2] + cm[:, 3]
    cm = cm[:3, :3]
    # make f1
    tp = cm[2, 2]
    fp = cm[0, 2] + cm[1, 2]
    fn = cm[2, 0] + cm[2, 1]
    _, _, old_f1_cds_1 = ConfusionMatrix._precision_recall_f1(tp, fp, fn)

    # parse metric table
    f1_scores = []
    for line in log_file:
        if 'Precision' in line:  # table start
            next(log_file)  # skip line
            for i in range(6):
                line = next(log_file)
                f1_scores.append(line.strip().split('|')[4].strip())
                if i == 3:
                    next(log_file)  # skip line

    # parse keras metrics
    keras_metrics = eval(''.join(line.strip().split(' ')[4:]))
    selected_keras_metrics = [keras_metrics['loss'], keras_metrics['acc']]
    str_rows = [genome] + ['{:.4f}'.format(n) for n in selected_keras_metrics] + f1_scores
    other_numbers = [old_f1_cds_1, base_level_error_rate, padded_bases_rate, sequence_error_rate]
    str_rows += ['{:.4f}'.format(n) for n in other_numbers] + [folder]
    print(','.join(str_rows))
