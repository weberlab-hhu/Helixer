#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
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

total_accs, genic_f1s, offsets = [], [], []
table = [['index', 'overall acc']]
for i in range(0, y_true.shape[1], args.resolution):
    y_true_section = y_true[:, i:i+args.resolution]
    y_pred_section = y_pred[:, i:i+args.resolution]
    y_diff_section = y_diff[:, i:i+args.resolution]
    overall_acc = np.count_nonzero(y_diff_section) / np.product(y_diff_section.shape) * 100
    table.append(f'{i}\t{overall_acc:.4f}'.split('\t'))
    total_accs.append(overall_acc / 100.0)
    # cm
    print(f'\n{i}/{y_true.shape[1]}')
    cm = ConfusionMatrix(None, 4)
    cm._add_to_cm(y_true_section, y_pred_section)
    genic_f1s.append(cm._print_results())
    offsets.append(i)
print('\n', AsciiTable(table).table, sep='')

plt.plot(offsets, total_accs, label='overall acc')
plt.plot(offsets, genic_f1s, label='genic f1')
plt.xlabel('length offset')
plt.legend()
plt.show()
