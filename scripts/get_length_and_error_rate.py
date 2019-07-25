#! /usr/bin/env python3
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True)
args = parser.parse_args()

f = h5py.File(args.data, 'r')
sw = f['/data/sample_weights']
tr = f['/data/X'][:, :, 0]

total_len = sw.size
total_intergenic = np.count_nonzero(tr == 0)
total_errors = np.count_nonzero(np.array(sw) == 0)

print(args.data)
print('Total len: {:.4f}Gb'.format(total_len / 10**9))
print('Total intergenic: {:.2f}% ({:.4f}Gb)'.format(total_intergenic / total_len * 100,
                                                    total_intergenic / 10**9))
print('Total errors: {:.2f}% ({:.4f}Gb)\n'.format(total_errors / total_len * 100,
                                                  total_errors / 10**9))
