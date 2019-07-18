#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from sklearn.metrics import precision_recall_fscore_support as f1_score
from helixerprep.prediction.F1Scores import F1Calculator

"""Outputs the coding and intron accuracy within and outside of genes seperately.
Needs a data and a corresponding predictions file."""


def append_f1_row(name, col, mask, h5_data_y, h5_pred_y, table):
    y = h5_data_y[:, :, col][mask]
    pred = h5_pred_y[:, :, col][mask]

    row = [name]
    scores = f1_score(y, np.round(pred), average='binary', pos_label=1)[:3]
    row += ['{:.4f}'.format(s) for s in scores]
    table.append(row)

def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    h5_data_y = h5_data['/data/y'][:args.truncate, :, :]
    h5_pred_y = h5_pred[args.h5_prediction_dataset][:args.truncate, :, :]
    f1_calc = F1Calculator(None, None)
    f1_calc.count_and_calculate_one_batch(h5_data_y, h5_pred_y)
    f1_calc.print_f1_scores()

    print_accuracy_counts(h5_data_y, h5_pred_y)

def acc_percent(y, preds):
    return np.sum(y == preds) / np.product(y.shape) * 100

def print_accuracy_counts(y, preds):
    print("overall accuracy: {:06.4f}%".format(acc_percent(y, preds)))
    print("transcriptional accuracy: {:06.4f}%".format(acc_percent(y[:, :, 0], preds[:, :, 0])))
    print("coding accuracy: {:06.4f}%".format(acc_percent(y[:, :, 1], preds[:, :, 1])))
    print("intron accuracy: {:06.4f}%".format(acc_percent(y[:, :, 2], preds[:, :, 2])))

def main_old(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    h5_data_y = h5_data['/data/y'][:args.truncate, :, :]
    h5_pred_y = h5_pred[args.h5_prediction_dataset][:args.truncate, :, :]

    genic_mask = h5_data['/data/y'][:args.truncate, :, 0].astype(bool)
    error_mask = h5_data['/data/sample_weights'][:args.truncate, :].astype(bool)
    genic_and_error_mask = np.logical_and(genic_mask, error_mask)

    # cds, intron cols in genic/intergenic
    for region in ['Genic', 'Intergenic']:
        table = [['', 'Precision', 'Recall', 'F1-Score']]
        if region == 'intergenic':
            current_mask = np.bitwise_not(genic_and_error_mask)
        else:
            current_mask = genic_and_error_mask
        for col, name in [(1, 'coding'), (2, 'intron')]:
            append_f1_row(name, col, current_mask, h5_data_y, h5_pred_y, table)
        print(AsciiTable(table, region).table)

    # all cols for everything
    table = [['', 'Precision', 'Recall', 'F1-Score']]
    for col, name in [(0, 'transcript'), (1, 'coding'), (2, 'intron')]:
        append_f1_row(name, col, error_mask, h5_data_y, h5_pred_y, table)
    print(AsciiTable(table, 'Total').table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--truncate', type=int, default=-1, help="can set to e.g. 1000 for development speed")
    parser.add_argument('--h5_prediction_dataset', type=str, default='/predictions',
                        help="dataset in predictions h5 file to compare with data's '/data/y', default='/predictions',"
                             "the other likely option is '/data/y'")
    main(parser.parse_args())
