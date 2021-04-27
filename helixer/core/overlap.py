#! /usr/bin/env python3

"""Does the overlapping, that was previously done in HelixerModel.py"""

import h5py
import numpy as np
from .helpers import get_contiguous_ranges


def _n_ori_chunks_from_batch_chunks(max_batch_size=32, overlap_depth=4):
    """calculate max number of original (non overlapped) chunks that fit in overlapped batch_size (or remaining)"""
    # batch_size_out = 1 + (n_chunks - 1) * overlap_depth
    # max_batch_size > 1 + (n_chunks - 1) * overlap_depth
    # (max_batch_size - 1) / overlap_depth  + 1 > n_chunks
    max_n_chunks = (max_batch_size - 1) // overlap_depth + 1
    return max_n_chunks


def _n_batch_from_ori_chunks(n_chunks=8, overlap_depth=4):
    """calculates resulting batch size from given number of contiguous original sequence chunks"""
    # batch_size_out = 1 + (n_chunks - 1) * overlap_depth
    return 1 + (n_chunks - 1) * overlap_depth


class SubBatch:
    def __init__(self, indices, edge_handle_start=False, edge_handle_end=False, is_plus_strand=True,
                 overlap_depth=4):
        self.indices = indices
        # if not on a sequence edge start and end bits will be cropped
        # also, start on negative strand would ideally have special handling for weird padding
        self.edge_handle_start = edge_handle_start
        self.edge_handle_end = edge_handle_end
        self.is_plus_strand = is_plus_strand
        self.overlap_depth = overlap_depth

    def sub_batch_size(self):
        return _n_batch_from_ori_chunks(len(self.indices))

    def make_x(self):
        pass

    def overlap_preds(self, preds):
        pass


# places where overlap will affect core functionality of HelixerSequence
class OverlapSeqHelper(object):
    """manipulations of how data is fed in for predictions"""
    def __init__(self, test_h5):
        self.test_h5 = test_h5
        self.sliding_batches = self._mk_sliding_batches()

    def _mk_sliding_batches(self, max_batch_size=32, overlap_depth=4):
        max_n_chunks = _n_ori_chunks_from_batch_chunks(max_batch_size, overlap_depth)
        step = max_n_chunks - 2   # -2 bc ends will be cropped
        # most of these will effectively be final batches, but short seqs/ends may be grouped together (for efficiency)
        sub_batches = []
        contiguous_ranges = get_contiguous_ranges(self.test_h5)
        for crange in contiguous_ranges:
            # step through sequence so that non-edges can have 1-chunk cropped off start/end
            # and regenerate original sequence with a simple concatenation there after
            for i in range(crange['start_i'], crange['end_i'] - 1, step):
                sub_batch_start = max(i - 1, crange['start_i'])  # pad 1 left (except seq edge)
                sub_batch_end = min(i + step + 1, crange['end_i'])   # pad 1 right (except seq edge)
                indices = tuple(range(i, sub_batch_end))
                sub_batches.append(
                    SubBatch(indices, overlap_depth=overlap_depth, is_plus_strand=crange['is_plus_strand'],
                             edge_handle_start=sub_batch_start == crange['start_i'],
                             edge_handle_end=sub_batch_end == crange['end_i'])
                )

        # group into final batches, so as to keep total size <= max_batch_size
        # i.e. achieve consistent (& user adjustable) memory usage on graphics card
        sliding_batches = []
        batch = []
        batch_total_size = 0
        for sb in sub_batches:
            if batch_total_size + sb.sub_batch_size() <= max_batch_size:
                batch.append(sb)
                batch_total_size += sb.sub_batch_size()
            else:
                sliding_batches.append(batch)
                batch = [sb]
                batch_total_size = sb.sub_batch_size()
        sliding_batches.append(batch)
        return sliding_batches

    def adjusted_epoch_length(self):
        return len(self.sliding_batches)

    def make_x(self, idx):
        sub_batches = self.sliding_batches[idx]
        x_as_list = [sb.make_x() for sb in sub_batches]
        # todo, stack and return


def _get_batch_data(self, idx):
    usable_idx_batch = self._usable_idx_batch(idx)
    if self.overlap:
        X = self.x_dset[usable_idx_batch]
        seqid_borders = self._get_seqid_borders(idx)
        # split data along these borders
        X_by_seqid = np.array_split(X, seqid_borders)
        overlapping_X = []
        for seqid_x in X_by_seqid:
            if len(seqid_x) >= self.min_seqs_for_overlapping:
                seq = np.concatenate(seqid_x, axis=0)
                # apply sliding window
                overlapping_X += [seq[i:i+self.chunk_size]
                                  for i in range(0, len(seq) - self.chunk_size + 1,
                                                 self.overlap_offset)]
            else:
                # do not overlap short sequences
                overlapping_X += [seqid_x[i] for i in range(len(seqid_x))]


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
