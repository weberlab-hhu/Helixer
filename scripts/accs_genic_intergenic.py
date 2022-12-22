#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from helixer.prediction.Metrics import ConfusionMatrix as ConfusionMatrix
import re
import os

class H5Holder:
    def __init__(self, h5_data, h5_pred, h5_prediction_dataset):
        # find phase dataset based on prediction dataset
        if h5_prediction_dataset in ['predictions', '/predictions']:
            h5_phase_dataset = "predictions_phase"
        elif h5_prediction_dataset.endswith('/y'):
            h5_phase_dataset = re.sub('/y$', '/phases', args.h5_prediction_dataset)
        else:
            raise ValueError(f"do not know how to find phase dataset from {args.h5_prediction_dataset}")

        # get comparable subset of data
        # category
        self.data_y = h5_data['data/y']
        self.pred_y = h5_pred[h5_prediction_dataset]
        # phase
        self.data_phase = h5_data['data/phases']
        self.pred_phase = h5_pred[h5_phase_dataset]
        # sample weights (to mask both of the above)
        self.sample_weights = h5_data['data/sample_weights']


def main(args):

    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    h5h = H5Holder(h5_data, h5_pred, args.h5_prediction_dataset)

    # and score
    cm_calc = ConfusionMatrix(None)
    cm_phase = ConfusionMatrix(['no_phase', 'phase_0', 'phase_1', 'phase_2'])

    # sanity check
    assert h5h.data_y.shape == h5h.pred_y.shape

    # truncate (for devel efficiency, when we don't need the whole answer)
    if args.truncate is not None:
        for key, val in h5h.__dict__.items():
            h5h.__setattr__(key, val[:args.truncate])

    # random subset (for devel efficiency, or just if we don't care that much about the full accuracy
    if args.sample is not None:
        a_sample = np.random.choice(
            np.arange(h5h.data_y.shape[0]),
            size=[args.sample],
            replace=False
        )
        a_sample = np.sort(a_sample)
        for key, val in h5h.__dict__.items():
            h5h.__setattr__(key, val[a_sample])

    # break into chunks (keep mem usage minimal)
    i = 0
    size = 1000
    while i < h5h.data_y.shape[0]:
        cm_calc.count_and_calculate_one_batch(h5h.data_y[i:(i + size)],
                                              h5h.pred_y[i:(i + size)],
                                              h5h.sample_weights[i:(i + size)])
        cm_phase.count_and_calculate_one_batch(h5h.data_phase[i:(i + size)],
                                               h5h.pred_phase[i:(i + size)],
                                               h5h.sample_weights[i:(i + size)])
        i += size

    cm_calc.print_cm()
    cm_calc.export_to_csvs(os.path.join(args.stats_dir, 'genic_class'))

    cm_phase.print_cm()
    cm_phase.export_to_csvs(os.path.join(args.stats_dir, 'phase'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True,
                        help="predictions h5 file, sorting _must_ match that of --data!")
    parser.add_argument('--truncate', type=int, default=None, help="look at just the first N chunks of each sequence")
    parser.add_argument('--h5_prediction_dataset', type=str, default='/predictions',
                        help="dataset in predictions h5 file to compare with data's '/data/y', default='/predictions',"
                             "the other likely option is '/data/y'")
    parser.add_argument('--sample', type=int, default=None,
                        help="take a random sample of the data of this many chunks per sequence")
    parser.add_argument('--stats_dir', type=str,
                        help="export several csv files of the calculated stats in this directory")
    args = parser.parse_args()
    main(args)
