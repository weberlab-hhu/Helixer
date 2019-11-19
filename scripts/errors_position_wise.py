#! /usr/bin/env python3
import os
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
parser.add_argument('-o', '--output-folder', type=str, default='')
parser.add_argument('-res', '--resolution', type=int, default=1000)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')
genome = args.data.strip().split('/')[6]

if args.sample:
    print('Sampling {} rows'.format(args.sample))
    a_sample = np.random.choice(h5_data['/data/y'].shape[0], size=[args.sample], replace=False)
    a_sample = list(np.sort(a_sample))
    y_true = h5_data['/data/y'][a_sample]
    y_pred = h5_pred['/predictions'][a_sample]
else:
    y_true = h5_data['/data/y']
    y_pred = h5_pred['/predictions']

assert y_true.shape == y_pred.shape
sw = np.array(h5_data['/data/sample_weights']).astype(bool)

total_accs, genic_f1s, offsets = [], [], []
table = [['index', 'overall acc']]
for i in range(0, y_true.shape[1], args.resolution):
    y_true_section = y_true[:, i:i+args.resolution].reshape((-1, 4))
    y_pred_section = y_pred[:, i:i+args.resolution].reshape((-1, 4))
    y_diff_section = np.argmax(y_true_section, axis=-1) == np.argmax(y_pred_section, axis=-1)

    # apply sw
    sw_section = sw[:, i:i+args.resolution].ravel()
    y_diff_section = y_diff_section[sw_section]

    overall_acc = np.count_nonzero(y_diff_section) / len(y_diff_section) * 100
    table.append(f'{i}\t{overall_acc:.4f}'.split('\t'))
    total_accs.append(overall_acc / 100.0)
    # cm
    print(f'\n{i}/{y_true.shape[1]}')
    cm = ConfusionMatrix(None)
    cm._add_to_cm(y_true_section, y_pred_section, sw_section)
    genic_f1s.append(cm._print_results())
    offsets.append(i)
print('\n', AsciiTable(table).table, sep='')

plt.title(genome)
plt.plot(offsets, total_accs, label='overall acc')
plt.plot(offsets, genic_f1s, label='genic f1')
plt.ylim((0.0, 1.0))
plt.xlabel('length offset')
plt.legend()
plt.savefig(os.path.join(args.output_folder, genome + '.png'))
print(genome + '.png saved')
