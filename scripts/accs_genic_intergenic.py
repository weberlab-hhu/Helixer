#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from helixer.prediction.Metrics import ConfusionMatrix as ConfusionMatrix
import sys
from helixer.core.helpers import mk_keys, mk_seqonly_keys
import os
import csv


class Exporter(object):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'w')
        self.saved1 = False
        self.length_added = 0

    def add_data(self, h5_in, labs, preds, lab_mask, lab_lexsort, sample_weights, start, end):
        if not self.saved1:
            self.saved1 = True
            self.mk_datasets(h5_in, labs)

        length = labs.shape[0]
        for key in h5_in['data'].keys():
            fullkey = 'data/' + key
            dset = h5_in[fullkey][start:end]
            if key not in  ["y", 'sample_weights']:
                cleanup = self.cleanup(dset, lab_mask, lab_lexsort)
            elif key == "y":
                cleanup = labs
            else:
                cleanup = sample_weights
            self.append(fullkey, cleanup, length)

        self.append('predictions', preds, length)
        self.length_added += length

    @staticmethod
    def cleanup(dset, lab_mask, lab_lexsort):
        cleanup = np.array(dset)[lab_mask]
        return cleanup[lab_lexsort]

    def append(self, fullkey, data, length):
        assert data.shape[0] == length
        dset = self.h5_file[fullkey]
        old_len = dset.shape[0]
        dset.resize(old_len + length, axis=0)
        dset[old_len:] = data

    def mk_datasets(self, h5_in, labs):
        # setup datasets
        for key in h5_in['data'].keys():
            dset = h5_in['data/' + key]
            shape = list(dset.shape)
            shape[0] = 0
            self.h5_file.create_dataset('data/' + key,
                                        shape=shape,
                                        maxshape=[None] + list(shape[1:]),
                                        dtype=dset.dtype,
                                        compression="lzf")
        self.h5_file.create_dataset('predictions',
                                    shape=[0] + list(labs.shape[1:]),
                                    maxshape=[None] + list(labs.shape[1:]),
                                    dtype='f',
                                    compression='lzf')

    def close(self):
        self.h5_file.close()


def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    # and score
    cm_calc = ConfusionMatrix(None)
    # prep keys
    lab_keys = list(mk_keys(h5_data))
    pred_keys = list(mk_keys(h5_pred))

    # prep export
    if args.save_to is not None:
        exporter = Exporter(args.save_to)

    for d_start, d_end, p_start, p_end in chunk(h5_data, h5_pred):
        print('starting', h5_data['data/seqids'][d_start], file=sys.stderr)
        # get comparable subset of data
        if not args.unsorted or all_coords_match(h5_data, h5_pred, (d_start, d_end), (p_start, p_end)):
            length = d_end - d_start
            h5_data_y = np.array(h5_data['/data/y'][d_start:d_end])
            h5_pred_y = np.array(h5_pred[args.h5_prediction_dataset][p_start:p_end])
            lab_mask = [True] * length
            lab_lexsort = np.arange(length)
            h5_sample_weights = np.array(h5_data['/data/sample_weights'][d_start:d_end])
        else:
            h5_data_y, h5_pred_y, lab_mask, lab_lexsort, h5_sample_weights = match_up(h5_data, h5_pred,
                                                                                      lab_keys, pred_keys,
                                                                                      args.h5_prediction_dataset,
                                                                                      data_start_end=(d_start, d_end),
                                                                                      pred_start_end=(p_start, p_end))

        # truncate (for devel efficiency, when we don't need the whole answer)
        if args.truncate is not None:
            assert args.save_to is None, "truncate and save not implemented"
            h5_data_y = h5_data_y[:args.truncate]
            h5_pred_y = h5_pred_y[:args.truncate]
        # random subset (for devel efficiency, or just if we don't care that much about the full accuracy
        if args.sample is not None:
            assert args.save_to is None, "sample and save not implemented"
            a_sample = np.random.choice(
                np.arange(h5_data_y.shape[0]),
                size=[args.sample],
                replace=False
            )
            h5_data_y = h5_data_y[a_sample]
            h5_pred_y = h5_pred_y[a_sample]

        # export the cleaned up matched up everything
        if args.save_to is not None:
            exporter.add_data(h5_in=h5_data, labs=h5_data_y, preds=h5_pred_y,
                              lab_mask=lab_mask, lab_lexsort=lab_lexsort,
                              sample_weights=h5_sample_weights,
                              start=d_start, end=d_end)

        # for all subsequent analysis round predictions
        h5_pred_y = np.round(h5_pred_y)

        # break into chunks (keep mem usage minimal)
        i = 0
        size = 1000
        while i < h5_data_y.shape[0]:
            cm_calc.count_and_calculate_one_batch(h5_data_y[i:(i + size)],
                                                  h5_pred_y[i:(i + size)],
                                                  h5_sample_weights[i:(i + size)])
            i += size

    if args.save_to is not None:
        exporter.close()
    cm_calc.print_cm()
    cm_calc.export_to_csvs(args.stats_dir)


def all_coords_match(h5_data, h5_pred, data_start_end, pred_start_end):
    d_start, d_end = data_start_end
    p_start, p_end = pred_start_end
    # if the arrays aren't all the same size, they certainly won't all match
    if d_end - d_start != p_end - p_start:
        return False
    for key in ['species', 'seqids', 'start_ends']:
        if np.any(h5_data['data/' + key][data_start_end[0]:data_start_end[1]] !=
                  h5_pred['data/' + key][pred_start_end[0]:pred_start_end[1]]):
            return False
    return True


def match_up(h5_data, h5_pred, lab_keys, pred_keys, h5_prediction_dataset, data_start_end=None, pred_start_end=None):
    if data_start_end is None:
        data_start_end = (0, h5_data['data/X'].shape[0])
    else:
        data_start_end = [int(x) for x in data_start_end]
    if pred_start_end is None:
        pred_start_end = (0, h5_pred['data/X'].shape[0])
    else:
        pred_start_end = [int(x) for x in pred_start_end]

    lab_keys = lab_keys[data_start_end[0]:data_start_end[1]]
    pred_keys = pred_keys[pred_start_end[0]:pred_start_end[1]]

    shared = list(set(lab_keys).intersection(set(pred_keys)))
    lab_mask = [x in shared for x in lab_keys]
    pred_mask = [x in shared for x in pred_keys]

    # setup output arrays (with shared indexes)
    labs = np.array(h5_data['data/y'][data_start_end[0]:data_start_end[1]])[lab_mask]
    preds = np.array(h5_pred[h5_prediction_dataset][pred_start_end[0]:pred_start_end[1]])[pred_mask]
    sample_weights = np.array(h5_data['data/sample_weights'][data_start_end[0]:data_start_end[1]][lab_mask])
    # check if sorting matches
    shared_lab_keys = np.array(lab_keys)[lab_mask]
    shared_pred_keys = np.array(pred_keys)[pred_mask]
    sorting_matches = np.all(shared_lab_keys == shared_pred_keys)

    # resort both if not
    if not sorting_matches:
        lab_lexsort = np.lexsort(np.flip(shared_lab_keys.T, axis=0))
        labs = labs[lab_lexsort]
        sample_weights = sample_weights[lab_lexsort]
        preds = preds[np.lexsort(np.flip(shared_pred_keys.T, axis=0))]
    else:
        lab_lexsort = np.arange(labs.shape[0])
    return labs, preds, lab_mask, lab_lexsort, sample_weights


def np_unique_checksort(an_array):
    uniques, starts, counts = np.unique(an_array, return_index=True, return_counts=True)

    for i in range(uniques.shape[0]):
        dat = uniques[i]
        start = starts[i]
        end = start + counts[i]
        assert np.all(an_array[start:end] == dat)
    return uniques, starts, counts


def chunk(h5_data, h5_pred):
    # this assumes that all unique species,seqid sets
    # occur in blocks, and will fail (with error) if not
    data_array = np.array(mk_seqonly_keys(h5_data))
    pred_array = np.array(mk_seqonly_keys(h5_pred))
    d_unique, d_starts, d_counts = np_unique_checksort(data_array)
    p_unique, p_starts, p_counts = np_unique_checksort(pred_array)
    theintersect = np.intersect1d(d_unique, p_unique)
    d_mask = np.in1d(d_unique, theintersect)
    p_mask = np.in1d(p_unique, theintersect)
    d_unique, d_starts, d_counts = [x[d_mask] for x in [d_unique, d_starts, d_counts]]
    p_unique, p_starts, p_counts = [x[p_mask] for x in [p_unique, p_starts, p_counts]]
    # np.unique sorts it's output, so data and preds should now match
    out = np.empty(shape=[d_unique.shape[0], 4], dtype=int)
    # data start, data end, pred start, pred end
    out[:, 0] = d_starts
    out[:, 1] = d_starts + d_counts
    out[:, 2] = p_starts
    out[:, 3] = p_starts + p_counts
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--truncate', type=int, default=None, help="look at just the first N chunks of each sequence")
    parser.add_argument('--h5_prediction_dataset', type=str, default='/predictions',
                        help="dataset in predictions h5 file to compare with data's '/data/y', default='/predictions',"
                             "the other likely option is '/data/y'")
    parser.add_argument('--unsorted', action='store_true',
                        help="don't assume coordinates match up but use the h5 datasets [species, seqids, start_ends]"
                             "to check order and reorder as necessary")
    parser.add_argument('--sample', type=int, default=None,
                        help="take a random sample of the data of this many chunks per sequence")
    parser.add_argument('--save_to', type=str, help="set this to output the newly sorted matches to a h5 file")
    parser.add_argument('--label_dim', type=int, default=4, help="number of classes, 4 (default) or 7")
    parser.add_argument('--stats_dir', type=str,
                        help="export several csv files of the calculated stats in this directory")
    main(parser.parse_args())
