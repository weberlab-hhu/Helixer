#! /usr/bin/env python3
"""Does very simple evaluation of a prediction file. Makes sure that the maximum memory footprint
always stays the same no matter the length of the sequences."""

import h5py
import argparse
import numpy as np
import sys
from helixer.prediction.ConfusionMatrix import ConfusionMatrix as ConfusionMatrix


def main(h5_file,  preds_file=None, predictions_dataset='predictions', ground_truth_dataset='data/y'):
    h5_data = h5py.File(h5_file, 'r')
    if preds_file is not None:
        h5_pred = h5py.File(preds_file, 'r')
    else:
        h5_pred = h5_data

    y_true = h5_data[ground_truth_dataset]
    y_pred = h5_pred[predictions_dataset]
    sw = h5_data['/data/sample_weights']

    assert y_true.shape == y_pred.shape
    assert y_pred.shape[:-1] == sw.shape

    # keep memory footprint the same no matter the seq length
    # chunk_size should be 100 for 20k length seqs
    chunk_size = 2 * 10 ** 6 // y_true.shape[1]
    print(f'Using chunk size {chunk_size}', file=sys.stderr)

    n_seqs = int(np.ceil(y_true.shape[0] / chunk_size))
    cm = ConfusionMatrix(None)
    for i in range(n_seqs):
        print(i, '/', n_seqs, end='\r', file=sys.stderr)
        cm._add_to_cm(y_true[i * chunk_size: (i + 1) * chunk_size],
                      y_pred[i * chunk_size: (i + 1) * chunk_size],
                      sw[i * chunk_size: (i + 1) * chunk_size])
    cm.print_cm()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5-data', help='h5 data file (with data/{X, y, species, seqids, etc ...}'
                                                'and evaluation/{coverage, spliced_coverage}, as output by'
                                                'the rnaseq.py script)', required=True)
    parser.add_argument('-p', '--h5-predictions', help='set if the predictions data set is in a separate '
                                                       '(but sort matching!) h5 file')
    parser.add_argument('--ground-truth-dataset', default='data/y')
    parser.add_argument('--predictions-dataset', default='predictions')
    args = parser.parse_args()
    main(args.h5_data, args.h5_predictions, args.predictions_dataset, args.ground_truth_dataset)