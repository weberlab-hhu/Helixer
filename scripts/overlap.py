#! /usr/bin/env python3

"""Does the overlapping, that was previously done in HelixerModel.py"""

import os
import h5py
import numpy as np
import argparse


def _overlap_predictions(predictions):
    # some shortcut variables
    seq_overhang = int((chunk_size - args.core_length) / 2)
    n_overhang_seqs = seq_overhang // args.overlap_offset
    n_original_seqs = test_sequence._seqs_per_batch(batch_idx=batch_idx)
    n_overlapping_seqs = args.core_length // args.overlap_offset

    all_predictions = np.empty((0, ) + predictions.shape[1:])
    # get number of sequences for each seqid from border distance
    seqid_sizes = np.diff(np.array([0] + seqid_borders + [n_original_seqs]))
    print(seqid_sizes)
    pred_offset = 0
    for seqid_size in seqid_sizes:
        if seqid_size > 2:
            n_seqid_seqs = (seqid_size - 1) * chunk_size // args.overlap_offset + 1
        else:
            n_seqid_seqs = seqid_size
        predictions_seqid = predictions[pred_offset:pred_offset + n_seqid_seqs]
        pred_offset += n_seqid_seqs
        if seqid_size >= args.min_seqs_for_overlapping:
            # actual overlapping; save first and last sequence for special handling later
            first, last = predictions_seqid[0], predictions_seqid[-1]
            # cut to the core
            predictions_seqid = [s[seq_overhang:-seq_overhang] for s in predictions_seqid]
            # generate zero'd out filler sequences for the start and end
            filler_seqs = [np.zeros((args.core_length, 4))] * n_overhang_seqs
            predictions_seqid = filler_seqs + predictions_seqid + filler_seqs
            # stack eveything
            predictions_seqid = np.stack(predictions_seqid).astype(predictions.dtype)
            # add overhang edge data from first/last seq that can not be overlapped
            predictions_seqid[0, :seq_overhang] = first[:seq_overhang]
            predictions_seqid[-1, -seq_overhang:] = last[-seq_overhang:]

            # merge and stack efficiently so everything can be averaged
            n_predicted_bases = seqid_size * chunk_size
            stacked = np.zeros((n_overlapping_seqs, n_predicted_bases, 4), dtype=predictions.dtype)
            for j in range(n_overlapping_seqs):
                # get idx of every n_overlapping_seqs'th seq starting at j
                idx = list(range(j, predictions_seqid.shape[0], n_overlapping_seqs))
                seq = np.concatenate(predictions_seqid[idx], axis=0)
                start_base = j * args.overlap_offset
                stacked[j, start_base:start_base+seq.shape[0]] = seq

            # average individual softmax values
            # does change pseudo-probability dist at the edge but not the argmax afterwards
            # (causes values to be lower there)
            averages = np.mean(stacked, axis=0)
            predictions_seqid = np.stack(np.split(averages, seqid_size))
        all_predictions = np.concatenate([all_predictions, predictions_seqid], axis=0)
    assert all_predictions.shape[0] == n_original_seqs
    return all_predictions


def overlap(args):
    h5_data = h5py.File(args.data, 'r')
    h5_input_preds = h5py.File(args.input_pred_file, 'r')
    h5_output_preds = h5py.File(args.overlapped_output_file, 'w')
    assert h5_data['/data/X'].shape == h5_input_preds['/predictions'].shape

    input_preds = h5_input_preds['/predictions']
    h5_output.create_dataset('/predictions',
                             maxshape=(None,) + input_preds.shape[1:],
                             chunks=(1,) + input_preds.shape[1:],
                             dtype='float32',
                             compression='lzf',
                             shuffle=True)

    chunk_size = input_preds.shape[1]
    assert args.overlap_offset < args.core_length
    # check if everything divides evenly to avoid further head aches
    assert (chunk_size / args.overlap_offset).is_integer()
    assert (args.batch_size / (chunk_size / args.overlap_offset)).is_integer()
    assert ((chunk_size - args.core_length) / 2 / args.overlap_offset).is_integer()

    # get seqid borders
    seqids = h5_data['/data/seqids'][:]
    seqid_borders = np.argwhere(seqids[:-1] != seqids[1:])[:, 0]
    if len(seqid_borders) > 0:
        # if there are changes in seqid
        seqid_borders = np.add(seqid_borders, 1)  # add 1 for splitting with np.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--input-pred-file', type=str, required=True)
    parser.add_argument('--overlapped-output-file', type=str, required=True)
    parser.add_argument('--overlap-offset', type=int, default=2500)
    parser.add_argument('--core-length', type=int, default=10000)
    parser.add_argument('--min-seqs-for-overlapping', type=int, default=3)
    args = parser.parse_args()

    overlap(args)
