#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from sklearn.metrics import precision_recall_fscore_support as f1_score
from helixerprep.prediction.F1Scores import F1Calculator

"""Outputs the coding and intron accuracy within and outside of genes seperately.
Needs a data and a corresponding predictions file."""


def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    # get comparable subset of data
    if not args.unsorted or all_coords_match(h5_data, h5_pred):
        h5_data_y = h5_data['/data/y']
        h5_pred_y = h5_pred[args.h5_prediction_dataset]
    else:
        h5_data_y, h5_pred_y = match_up(h5_data, h5_pred, args.h5_prediction_dataset)

    # truncate (for devel efficiency, when we don't need the whole answer)
    h5_data_y = h5_data_y[:args.truncate]
    h5_pred_y = h5_pred_y[:args.truncate]

    # and score
    f1_calc = F1Calculator(None, None)
    f1_calc.count_and_calculate_one_batch(h5_data_y, h5_pred_y)
    f1_calc.print_f1_scores()

    print_accuracy_counts(h5_data_y, h5_pred_y)


def acc_percent(y, preds):
    return np.sum(y == preds) / np.product(y.shape) * 100


def all_coords_match(h5_data, h5_pred):
    return list(mk_keys(h5_data)) == list(mk_keys(h5_pred))


def match_up(h5_data, h5_pred, h5_prediction_dataset):
    lab_keys = list(mk_keys(h5_data))
    pred_keys = list(mk_keys(h5_pred))

    shared = list(set(lab_keys).intersection(set(pred_keys)))
    lab_mask = [x in shared for x in lab_keys]
    pred_mask = [x in shared for x in pred_keys]

    # setup output arrays (with shared indexes)
    labs = np.array(h5_data['data/y'])[lab_mask]
    preds = np.array(h5_pred[h5_prediction_dataset])[pred_mask]

    # check if sorting matches
    shared_lab_keys = np.array(lab_keys)[lab_mask]
    shared_pred_keys = np.array(pred_keys)[pred_mask]
    sorting_matches = (shared_lab_keys == shared_pred_keys).all()
    # resort both if not
    if not sorting_matches:
        labs = labs[np.lexsort(np.flip(shared_lab_keys.T, axis=0))]
        preds = preds[np.lexsort(np.flip(shared_pred_keys.T, axis=0))]
    # todo, option to save as h5?
    return labs, preds


def mk_keys(h5):
    return zip(h5['data/species'],
               h5['data/seqids'],
               h5['data/start_ends'][:, 0],
               h5['data/start_ends'][:, 1])

def print_accuracy_counts(y, preds):
    print("overall accuracy: {:06.4f}%".format(acc_percent(y, preds)))
    print("transcriptional accuracy: {:06.4f}%".format(acc_percent(y[:, :, 0], preds[:, :, 0])))
    print("coding accuracy: {:06.4f}%".format(acc_percent(y[:, :, 1], preds[:, :, 1])))
    print("intron accuracy: {:06.4f}%".format(acc_percent(y[:, :, 2], preds[:, :, 2])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--truncate', type=int, default=-1, help="can set to e.g. 1000 for development speed")
    parser.add_argument('--h5_prediction_dataset', type=str, default='/predictions',
                        help="dataset in predictions h5 file to compare with data's '/data/y', default='/predictions',"
                             "the other likely option is '/data/y'")
    parser.add_argument('--unsorted', action='store_true',
                        help="don't assume coordinates match up but use the h5 datasets [species, seqids, start_ends]"
                             "to check order and reorder as necessary")
    main(parser.parse_args())
