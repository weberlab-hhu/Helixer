#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from helixer.prediction.Metrics import ConfusionMatrix as ConfusionMatrix


def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    # and score
    cm_calc = ConfusionMatrix(None)

    # sanity check
    assert h5_data['data/y'].shape[0] == h5_pred[args.h5_prediction_dataset].shape[0]

    # get comparable subset of data
    h5_data_y = h5_data['data/y']
    h5_pred_y = h5_pred[args.h5_prediction_dataset]
    h5_sample_weights = h5_data['data/sample_weights']

    # truncate (for devel efficiency, when we don't need the whole answer)
    if args.truncate is not None:
        h5_data_y = h5_data_y[:args.truncate]
        h5_pred_y = h5_pred_y[:args.truncate]
        h5_sample_weights = h5_sample_weights[:args.truncate]

    # random subset (for devel efficiency, or just if we don't care that much about the full accuracy
    if args.sample is not None:
        a_sample = np.random.choice(
            np.arange(h5_data_y.shape[0]),
            size=[args.sample],
            replace=False
        )
        a_sample = np.sort(a_sample)
        h5_data_y = h5_data_y[a_sample]
        h5_pred_y = h5_pred_y[a_sample]
        h5_sample_weights = h5_sample_weights[a_sample]

    # break into chunks (keep mem usage minimal)
    i = 0
    size = 1000
    while i < h5_data_y.shape[0]:
        cm_calc.count_and_calculate_one_batch(h5_data_y[i:(i + size)],
                                              h5_pred_y[i:(i + size)],
                                              h5_sample_weights[i:(i + size)])
        i += size

    cm_calc.print_cm()
    cm_calc.export_to_csvs(args.stats_dir)


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
    parser.add_argument('--label_dim', type=int, default=4, help="number of classes, 4 (default) or 7")
    parser.add_argument('--stats_dir', type=str,
                        help="export several csv files of the calculated stats in this directory")
    args = parser.parse_args()
    main(args)
