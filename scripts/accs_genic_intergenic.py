#! /usr/bin/env python3
import h5py
import numpy as np
import argparse
from terminaltables import AsciiTable
from sklearn.metrics import precision_recall_fscore_support as f1_score
from helixerprep.prediction.F1Scores import F1Calculator


def main(args):
    h5_data = h5py.File(args.data, 'r')
    h5_pred = h5py.File(args.predictions, 'r')

    # get comparable subset of data
    if not args.unsorted or all_coords_match(h5_data, h5_pred):
        h5_data_y = np.array(h5_data['/data/y'])
        h5_pred_y = np.array(h5_pred[args.h5_prediction_dataset])
        lab_mask = [True] * h5_data_y.shape[0]
        lab_lexsort = np.arange(h5_data_y.shape[0])
    else:
        h5_data_y, h5_pred_y, lab_mask, lab_lexsort = match_up(h5_data, h5_pred,
                                                               args.h5_prediction_dataset)

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

        export(args.save_to, h5_in=h5_data,
               labs=h5_data_y, preds=h5_pred_y,
               lab_mask=lab_mask, lab_lexsort=lab_lexsort)

    # for all subsequent analysis round predictions
    h5_pred_y = np.round(h5_pred_y)

    # and score
    f1_calc = F1Calculator(None, None)
    # break into chunks (so as to not run out of memory)
    i = 0
    size = 100
    while i < h5_data_y.shape[0]:
        f1_calc.count_and_calculate_one_batch(h5_data_y[i:(i + size)],
                                              h5_pred_y[i:(i + size)])
        i += size

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


def mk_keys(h5):
    return zip(h5['data/species'],
               h5['data/seqids'],
               h5['data/start_ends'][:, 0],
               h5['data/start_ends'][:, 1])


def print_accuracy_counts(y, preds):
    print("overall accuracy: {:06.4f}%".format(acc_percent(y[:], preds[:])))
    print("transcriptional accuracy: {:06.4f}%".format(acc_percent(y[:, :, 0], preds[:, :, 0])))
    print("coding accuracy: {:06.4f}%".format(acc_percent(y[:, :, 1], preds[:, :, 1])))
    print("intron accuracy: {:06.4f}%".format(acc_percent(y[:, :, 2], preds[:, :, 2])))


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
