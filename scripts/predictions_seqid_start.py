#! /usr/bin/env python3
import os
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
parser.add_argument('-po', '--predictions-overlap', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')
h5_pred_overlap = h5py.File(args.predictions_overlap, 'r')

y_true = h5_data['/data/y']
y_pred = h5_pred['/predictions']
y_pred_overlap = h5_pred_overlap['/predictions']

assert y_true.shape == y_pred.shape
sw = np.array(h5_data['/data/sample_weights']).astype(bool)

seqids = np.array(h5_data['/data/seqids'])
idx_border = np.squeeze(np.argwhere(seqids[:-1] != seqids[1:]))
idx_border = list(np.add(idx_border, 1))
y_pred, y_pred_overlap = y_pred[idx_border], y_pred_overlap[idx_border]

for i in range(len(y_pred)):
    different_bases = np.where(np.argmax(y_pred[i, :5000], axis=-1) !=
                               np.argmax(y_pred_overlap[i, :5000], axis=-1))[0]
    beginning_close = np.allclose(y_pred[i, :5000] / 4, y_pred_overlap[i, :5000])
    print(i, beginning_close, different_bases)
    if i % 100 == 0:
        import pudb; pudb.set_trace()
        pass

