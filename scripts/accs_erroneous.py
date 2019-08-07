#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable

def overall_acc_percent(y_true, y_pred):
    return np.count_nonzero(y_true == y_pred) / np.product(y_true.shape) * 100

def rowwise_acc_percent(y_true, y_pred):
    return np.count_nonzero((y_true == y_pred).all(axis=2)) / np.product(y_true.shape[:2]) * 100

def get_accs(y_true, y_pred, err_idx, acc_fn):
    clean_idx = np.logical_not(err_idx)
    total_acc = acc_fn(y_true, y_pred)
    error_acc = acc_fn(y_true[err_idx], y_pred[err_idx])
    clean_acc = acc_fn(y_true[clean_idx], y_pred[clean_idx])
    return ['{:.4f}'.format(a) for a in [total_acc, error_acc, clean_acc]]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

y_true = np.array(h5_data['/data/y'])
y_pred = np.round(h5_pred['/predictions'])
err_idx = np.array(h5_data['/data/err_samples'])

table = [['', 'acc total', 'acc error', 'acc clean']]
table.append(['pointwise acc'] + get_accs(y_true, y_pred, err_idx, overall_acc_percent))
table.append(['rowwise acc'] + get_accs(y_true, y_pred, err_idx, rowwise_acc_percent))
print('\n', AsciiTable(table).table, sep='')
