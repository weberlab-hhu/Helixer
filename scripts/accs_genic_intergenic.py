#! /usr/bin/env python3
import h5py
import numpy as np
import argparse

"""Outputs the coding and intron accuracy within and outside of genes seperately.
Needs a data and a corresponding predictions file."""

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, required=True)
parser.add_argument('-predictions', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

# can be set for faster development
TRUNCATE = 100000

genic_mask = h5_data['/data/y'][:TRUNCATE, :, 0].astype(bool)
error_mask = h5_data['/data/sample_weights'][:TRUNCATE, :].astype(bool)
full_mask = np.logical_and(genic_mask, error_mask)
for col, name in [(1, 'coding'), (2, 'intron')]:
    for region in ['genic', 'intergenic']:
        if region == 'intergenic':
            current_mask = np.bitwise_not(full_mask)
        else:
            current_mask = full_mask
        y = h5_data['/data/y'][:TRUNCATE, :, col][current_mask]
        pred = h5_pred['/predictions'][:TRUNCATE, :, col][current_mask]

        total_correct = np.sum(y == np.round(pred))
        acc = total_correct / y.size
        print('{} {} acc: {:.4f}'.format(name, region, acc))
