#! /usr/bin/env python3

# Prints overall statistics like genome size, error rate and fragmentation for each
# genome in a subfolder of a given folder as well as the overall statistics for all
# genomes combined

import os
import h5py
import numpy as np
import argparse
from collections import defaultdict

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

parser = argparse.ArgumentParser()
parser.add_argument('main_folder', type=str)
args = parser.parse_args()

results = defaultdict(dict)
for folder in listdir_fullpath(args.main_folder):
    species = os.path.basename(folder)
    f = h5py.File(os.path.join(folder, 'test_data.h5'), 'r')
    sw = f['/data/sample_weights']
    # tr = f['/data/X'][:, :, 0]

    results[species]['total_len'] = sw.size
    """
    total_intergenic = np.count_nonzero(tr == 0)
    total_errors = np.count_nonzero(np.array(sw) == 0)

    print(args.data)
    print('Total len: {:.4f}Gb'.format(total_len / 10**9))
    print('Total intergenic: {:.2f}% ({:.4f}Gb)'.format(total_intergenic / total_len * 100,
                                                        total_intergenic / 10**9))
    print('Total errors: {:.2f}% ({:.4f}Gb)\n'.format(total_errors / total_len * 100,
                                                      total_errors / 10**9))
    """

print(','.join(['total_len']))
for species, stats in results.items():
    print(','.join(str(e) for e in [species, stats['total_len']]))

