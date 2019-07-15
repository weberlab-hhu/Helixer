#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from sklearn.metrics import precision_recall_fscore_support as f1_score

"""Outputs the coding and intron accuracy within and outside of genes seperately.
Needs a data and a corresponding predictions file."""

def append_f1_row(name, col, mask):
    y = h5_data['/data/y'][:TRUNCATE, :, col][mask]
    pred = h5_pred['/predictions'][:TRUNCATE, :, col][mask]

    row = [name]
    scores = f1_score(y, np.round(pred), average='binary', pos_label=1)[:3]
    row += ['{:.4f}'.format(s) for s in scores]
    table.append(row)


parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, required=True)
parser.add_argument('-predictions', type=str, required=True)
args = parser.parse_args()

h5_data = h5py.File(args.data, 'r')
h5_pred = h5py.File(args.predictions, 'r')

# can be set for faster development
# TRUNCATE = 10000000
TRUNCATE = 1000

genic_mask = h5_data['/data/y'][:TRUNCATE, :, 0].astype(bool)
error_mask = h5_data['/data/sample_weights'][:TRUNCATE, :].astype(bool)
full_mask = np.logical_and(genic_mask, error_mask)

# cds, intron cols in genic/intergenic
for region in ['Genic', 'Intergenic']:
    table = [['', 'Precision', 'Recall', 'F1-Score']]
    if region == 'intergenic':
        current_mask = np.bitwise_not(full_mask)
    else:
        current_mask = full_mask
    for col, name in [(1, 'coding'), (2, 'intron')]:
        append_f1_row(name, col, current_mask)
    print(AsciiTable(table, region).table)

# all cols for everything
table = [['', 'Precision', 'Recall', 'F1-Score']]
for col, name in [(0, 'transcript'), (1, 'coding'), (2, 'intron')]:
    append_f1_row(name, col, full_mask)
print(AsciiTable(table, 'Total').table)

# everything
# todo

