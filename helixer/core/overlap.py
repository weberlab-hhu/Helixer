#! /usr/bin/env python3

"""handler code for sliding window predictions and re-overlapping things back to original sequences"""

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
    YDIM = 4

    def __init__(self, indices, x_dset, edge_handle_start=False, edge_handle_end=False, is_plus_strand=True,
                 overlap_depth=4, overlap_offset=5000, chunk_size=20000):
        self.indices = indices
        self.x_dset = x_dset
        # if not on a sequence edge start and end bits will be cropped
        # also, start on negative strand would ideally have special handling for weird padding
        self.edge_handle_start = edge_handle_start
        self.edge_handle_end = edge_handle_end
        self.is_plus_strand = is_plus_strand
        self.overlap_depth = overlap_depth
        self.overlap_offset = overlap_offset
        self.chunk_size = chunk_size

    @property
    def seq_length(self):
        return self.chunk_size * len(self.indices)

    @property
    def sub_batch_size(self):
        return _n_batch_from_ori_chunks(len(self.indices), self.overlap_depth)

    def sliding_coordinates(self):
        for i in range(0, self.seq_length - self.chunk_size + 1, self.overlap_offset):
            yield i, i + self.chunk_size

    def make_x(self):
        """generate sliding window X, to feed into network for predicting"""
        x = self.x_dset[self.indices]
        seq = x.reshape((-1, self.YDIM))
        # apply sliding window
        overlapping_x = [seq[start:end] for start, end in self.sliding_coordinates()]
        return overlapping_x

    def _overlap_preds(self, preds, core_length=10000):
        """take sliding-window predictions, and overlap (w/end clipping) to generate original coordinate predictions"""
        trim_by = (self.chunk_size - core_length) / 2
        preds_out = np.full(shape=(self.seq_length, self.YDIM), fill_value=0)
        counts = np.full(shape=(self.seq_length,), fill_value=0)

        len_preds = len(preds)
        for i, chunk, start_end in zip(range(len_preds), preds, self.sliding_coordinates()):
            start, end = start_end
            # cut to core, (but not sequence ends)
            if i > 0:  # all except first seq
                start += trim_by
            if i < len_preds - 1:  # all except last seq
                end -= trim_by
            sub_counts = counts[start:end]
            # average weighted by number of predictions counted at position so far
            preds_out[start:end] = (preds_out[start:end] * sub_counts + chunk) / (sub_counts + 1)
            # increment counted so far
            counts[start:end] += 1
        preds_out = preds_out.reshape(shape=(len(self.indices), self.chunk_size, self.YDIM))
        return preds_out

    def overlap_and_edge_handle_preds(self, preds, core_length=10000):
        clean_preds = self._overlap_preds(preds, core_length)
        if not self.edge_handle_start:
            clean_preds = clean_preds[1:]
        if not self.edge_handle_end:
            clean_preds = clean_preds[:-1]
        return clean_preds


# places where overlap will affect core functionality of HelixerSequence
class OverlapSeqHelper(object):
    """manipulations of how data is fed in for predictions"""
    def __init__(self, test_h5, x_dset):
        self.test_h5 = test_h5
        self.x_dset = x_dset
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
                             x_dset=self.x_dset, edge_handle_start=sub_batch_start == crange['start_i'],
                             edge_handle_end=sub_batch_end == crange['end_i'])
                )

        # group into final batches, so as to keep total size <= max_batch_size
        # i.e. achieve consistent (& user adjustable) memory usage on graphics card
        sliding_batches = []
        batch = []
        batch_total_size = 0
        for sb in sub_batches:
            if batch_total_size + sb.sub_batch_size <= max_batch_size:
                batch.append(sb)
                batch_total_size += sb.sub_batch_size
            else:
                sliding_batches.append(batch)
                batch = [sb]
                batch_total_size = sb.sub_batch_size
        sliding_batches.append(batch)
        return sliding_batches

    def adjusted_epoch_length(self):
        return len(self.sliding_batches)

    def make_x(self, idx):
        sub_batches = self.sliding_batches[idx]
        x_as_list = [sb.make_x() for sb in sub_batches]
        x = np.concatenate(x_as_list)
        return x

    def overlap_predictions(self, idx, predictions, core_length=10000):
        sub_batches = self.sliding_batches[idx]
        sub_batch_lengths = [sb.sub_batch_size for sb in sub_batches]
        sub_batch_starts = np.cumsum(sub_batch_lengths) - sub_batch_lengths
        out = []
        for start, length, sb in zip(sub_batch_starts, sub_batch_lengths, sub_batches):
            preds = predictions[start:(start + length)]
            out.append(sb.overlap_and_edge_handle_preds(preds, core_length))
        return np.concatenate(out)


