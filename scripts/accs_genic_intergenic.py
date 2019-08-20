#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from helixerprep.prediction.F1Scores import F1Calculator
import itertools


class AccuracyCalculator(object):
    def __init__(self, name):
        self.right = 0
        self.total = 0
        self.name = name

    def count_and_calculate_one_batch(self, y, preds):
        self.right += np.sum(y == preds)
        self.total += np.product(y.shape)

    def cal_accuracy(self):
        return self.right / self.total * 100


class AllAccuracyCalculator(object):
    DIMENSIONS = (0, 1, 2, (0, 1, 2))
    NAMES = ("tr", "cds", "intron", "total")

    def __init__(self):
        self.calculators = []
        for name in AllAccuracyCalculator.DIMENSIONS:
            self.calculators.append(AccuracyCalculator(name))

    def count_and_calculate_one_batch(self, y, preds):
        for i, dim in enumerate(AllAccuracyCalculator.DIMENSIONS):
            self.calculators[i].count_and_calculate_one_batch(y[:, :, dim], preds[:, :, dim])

    def print_accuracy(self):
        table_name = "Accuracy"
        table = [["region", "accuracy"]]
        for i, name in enumerate(AllAccuracyCalculator.NAMES):
            acc = self.calculators[i].cal_accuracy()
            table.append([name, '{:.4f}'.format(acc)])
        print('\n', AsciiTable(table, table_name).table, sep='')


def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    # and score
    f1_calc = F1Calculator(None)
    acc_calc = AllAccuracyCalculator()
    # prep keys
    lab_keys = list(mk_keys(h5_data))
    pred_keys = list(mk_keys(h5_pred))
    for d_start, d_end, p_start, p_end in chunk(h5_data, h5_pred):
        # get comparable subset of data
        if not args.unsorted or all_coords_match(h5_data, h5_pred):  # todo check subset only
            length = d_end - d_start
            h5_data_y = np.array(h5_data['/data/y'][d_start:d_end])
            h5_pred_y = np.array(h5_pred[args.h5_prediction_dataset][p_start:p_end])
            lab_mask = [True] * length
            lab_lexsort = np.arange(length)
        else:
            h5_data_y, h5_pred_y, lab_mask, lab_lexsort = match_up(h5_data, h5_pred,
                                                                   lab_keys, pred_keys,
                                                                   args.h5_prediction_dataset,
                                                                   data_start_end=(d_start, d_end),
                                                                   pred_start_end=(p_start, p_end))

        ## truncate (for devel efficiency, when we don't need the whole answer)
        #if args.truncate is not None:
        #    assert args.save_to is None, "truncate and save not implemented"
        #    h5_data_y = h5_data_y[:args.truncate]
        #    h5_pred_y = h5_pred_y[:args.truncate]
        ## random subset (for devel efficiency, or just if we don't care that much about the full accuracy
        #if args.sample is not None:
        #    assert args.save_to is None, "sample and save not implemented"
        #    a_sample = np.random.choice(
        #        np.arange(h5_data_y.shape[0]),
        #        size=[args.sample],
        #        replace=False
        #    )
        #    h5_data_y = h5_data_y[a_sample]
        #    h5_pred_y = h5_pred_y[a_sample]

        # export the cleaned up matched up everything
        #if args.save_to is not None:

        #    export(args.save_to, h5_in=h5_data,
        #           labs=h5_data_y, preds=h5_pred_y,
        #           lab_mask=lab_mask, lab_lexsort=lab_lexsort)

        # for all subsequent analysis round predictions
        h5_pred_y = np.round(h5_pred_y)


        # break into chunks (so as to not run out of memory)
        i = 0
        size = 1000
        while i < h5_data_y.shape[0]:
            f1_calc.count_and_calculate_one_batch(h5_data_y[i:(i + size)],
                                                  h5_pred_y[i:(i + size)])
            acc_calc.count_and_calculate_one_batch(h5_data_y[i:(i + size)],
                                                   h5_pred_y[i:(i + size)])
            i += size

    f1_calc.print_f1_scores()
    acc_calc.print_accuracy()


def acc_percent(y, preds):
    return np.sum(y == preds) / np.product(y.shape) * 100


def all_coords_match(h5_data, h5_pred):
    for data, pred in zip(mk_keys(h5_data), mk_keys(h5_pred)):
        if data != pred:
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
    print(data_start_end, pred_start_end)

    lab_keys = lab_keys[data_start_end[0]:data_start_end[1]]
    pred_keys = pred_keys[pred_start_end[0]:pred_start_end[1]]

    shared = list(set(lab_keys).intersection(set(pred_keys)))
    lab_mask = [x in shared for x in lab_keys]
    pred_mask = [x in shared for x in pred_keys]

    # setup output arrays (with shared indexes)
    labs = np.array(h5_data['data/y'][data_start_end[0]:data_start_end[1]])[lab_mask]
    preds = np.array(h5_pred[h5_prediction_dataset][pred_start_end[0]:pred_start_end[1]])[pred_mask]

    # check if sorting matches
    shared_lab_keys = np.array(lab_keys)[lab_mask]
    shared_pred_keys = np.array(pred_keys)[pred_mask]
    sorting_matches = np.all(shared_lab_keys == shared_pred_keys)

    # resort both if not
    if not sorting_matches:
        lab_lexsort = np.lexsort(np.flip(shared_lab_keys.T, axis=0))
        labs = labs[lab_lexsort]
        preds = preds[np.lexsort(np.flip(shared_pred_keys.T, axis=0))]
    else:
        lab_lexsort = np.arange(labs.shape[0])
    # todo, option to save as h5?
    return labs, preds, lab_mask, lab_lexsort


def export(h5_path, h5_in, labs, preds, lab_mask, lab_lexsort):
    h5_file = h5py.File(h5_path, 'w')
    # setup datasets
    for key in h5_in['data'].keys():
        dset = h5_in['data/' + key]
        shape = list(dset.shape)
        shape[0] = labs.shape[0]
        h5_file.create_dataset('data/' + key,
                               shape=shape,
                               dtype=dset.dtype,
                               compression="lzf"
                               )
        if key != "y":
            cleanup = np.array(dset)[lab_mask]
            cleanup = cleanup[lab_lexsort]
            h5_file['data/' + key][:] = cleanup
    h5_file['data/y'][:] = labs
    h5_file.create_dataset('predictions',
                           shape=preds.shape,
                           dtype='int8',
                           compression='lzf')
    h5_file['predictions'][:] = preds
    h5_file.close()


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


def mk_seqonly_keys(h5):
    return [a + b for a, b in zip(h5['data/species'],
                                  h5['data/seqids'])]


def mk_keys(h5):
    return zip(h5['data/species'],
               h5['data/seqids'],
               h5['data/start_ends'][:, 0],
               h5['data/start_ends'][:, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--truncate', type=int, default=None, help="can set to e.g. 1000 for development speed")
    parser.add_argument('--h5_prediction_dataset', type=str, default='/predictions',
                        help="dataset in predictions h5 file to compare with data's '/data/y', default='/predictions',"
                             "the other likely option is '/data/y'")
    parser.add_argument('--unsorted', action='store_true',
                        help="don't assume coordinates match up but use the h5 datasets [species, seqids, start_ends]"
                             "to check order and reorder as necessary")
    parser.add_argument('--sample', type=int, default=None,
                        help="take a random sample of the data of this many chunks")
    parser.add_argument('--save_to', type=str, help="set this to output the newly sorted matches to a h5 file")
    main(parser.parse_args())
