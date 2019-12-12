#! /usr/bin/env python3
"""Outputs the raw error rate for each sample in --data. Includes 0-padding."""
import os
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
args = parser.parse_args()

f = h5py.File(args.data, 'r')

sw_dset = f['/data/sample_weights']
y_dset = f['/data/y']
for i in range(y_dset.shape[0]):
    y, sw = y_dset[i], sw_dset[i]
    n_error_bases = np.count_nonzero(sw == 0)
    ratio = n_error_bases / y_dset.shape[1]
    print(f'{ratio:.6f}')
