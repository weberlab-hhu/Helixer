#! /usr/bin/env python3
import h5py
import argparse
import numpy as np
from helixerprep.prediction.ConfusionMatrix import ConfusionMatrix as ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-p', '--predictions', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

y_true = h5_data['/data/y']
y_pred = h5_pred['/predictions']
sw = h5_data['/data/sample_weights']

assert y_true.shape == y_pred.shape
assert y_pred.shape[:-1] == sw.shape

# keep memory footprint the same no matter the seq length
# chunk_size should be 100 for 20k length seqs
chunk_size = 2 * 10 ** 6 // y_true.shape[1]
print(f'Using chunk size {chunk_size}')

n_seqs = int(np.ceil(y_true.shape[0] / chunk_size))
cm = ConfusionMatrix(None)
for i in range(n_seqs):
    print(i, '/', n_seqs, end='\r')
    cm._add_to_cm(y_true[i * chunk_size: (i + 1) * chunk_size],
                  y_pred[i * chunk_size: (i + 1) * chunk_size],
                  sw[i * chunk_size: (i + 1) * chunk_size])
cm.print_cm()
