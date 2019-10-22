#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
parser.add_argument('-s', '--sample', type=int, default=None)
parser.add_argument('-res', '--resolution', type=int, default=1000)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

if args.sample:
    print('Sampling {} rows'.format(args.sample))
    a_sample = np.random.choice(h5_data['/data/y'].shape[0], size=[args.sample], replace=False)
    a_sample = list(np.sort(a_sample))
    y_true = h5_data['/data/y'][a_sample]
    y_pred = h5_pred['/predictions'][a_sample]
else:
    y_true = h5_data['/data/y']
    y_pred = h5_pred['/predictions']

# this can not be done at once if there is not enough memory
y_diff = np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)

table = [['index', 'overall acc']]
for i in range(0, y_true.shape[1], args.resolution):
    y_true_section = y_true[:, i:i+args.resolution]
    y_pred_section = y_pred[:, i:i+args.resolution]
    y_diff_section = y_diff[:, i:i+args.resolution]
    overall_acc = np.count_nonzero(y_diff_section) / np.product(y_diff_section.shape) * 100
    table.append(f'{i}\t{overall_acc:.4f}'.split('\t'))
    # cm
    print(f'\n{i}/{y_true.shape[1]}')
    cm = ConfusionMatrix(None, 4)
    cm._add_to_cm(y_true_section, y_pred_section)
    cm._print_results()
print('\n', AsciiTable(table).table, sep='')
