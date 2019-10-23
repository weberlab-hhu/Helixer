#! /usr/bin/env python3
import os
import numpy as np
import h5py
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix

nni_base = '/mnt/data/experiments_backup/nni_clc_server/nni/experiments/'
nni_id = 'XWPVurip'
trials_folder = '{}/{}/trials'.format(nni_base, nni_id)

print(','.join(['genome', 'loss', 'acc_overall', 'f1_ig', 'f1_utr', 'f1_exon', 'f1_intron',
                'f1_cds', 'f1_genic', 'old_f1_cds_1', 'error_rate', 'nni_id']))
for folder in os.listdir(trials_folder):
    if folder in ['g8EKR', 'rPxUg']:
        continue
    # get genome name
    parameters = eval(open('{}/{}/parameter.cfg'.format(trials_folder, folder)).read())
    path = parameters['parameters']['test_data']
    # genome = path.split('/')[5]  # when from cluster
    genome = path.split('/')[6]

    # get error rate
    f = h5py.File('/home/felix/Desktop/data/single_genomes/' + genome + '/h5_data_20k/test_data.h5')
    n_samples = f['/data/X'].shape[0]
    err = np.array(f['/data/err_samples'])
    n_err_samples = np.count_nonzero(err == True)
    error_rate = n_err_samples / n_samples

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
    str_rows += ['{:.4f}'.format(n) for n in [old_f1_cds_1, error_rate]] + [folder]
    print(','.join(str_rows))
