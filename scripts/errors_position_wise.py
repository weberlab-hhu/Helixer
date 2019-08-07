#! /usr/bin/env python3
import h5py
import numpy as np
import argparse

def overall_acc_percent(y_true, y_pred):
    return np.sum(y_true == y_pred) / np.product(y_true.shape) * 100

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
parser.add_argument('-s', '--sample', type=int, default=None)
parser.add_argument('-res', '--resolution', type=int, default=100)
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

for i in range(0, y_true.shape[1], args.resolution):
    y_true_section = y_true[:, i:i+args.resolution, :]
    y_pred_section = y_pred[:, i:i+args.resolution, :]
    overall_acc = overall_acc_percent(y_true_section, np.round(y_pred_section))
    print(i, '{:.4f},'.format(overall_acc))
