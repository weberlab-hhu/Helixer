#! /usr/bin/env python3

"""handler code for sliding window predictions and re-overlapping things back to original sequences"""

import numpy as np
import math


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

    def __init__(self, h5_indices, edge_handle_start=False, edge_handle_end=False, is_plus_strand=True,
                 overlap_offset=5000, chunk_size=20000, keep_start=None, keep_end=None):
        self.h5_indices = h5_indices
        # the following parameters are primarily for debugging
        self.keep_start = keep_start
        self.keep_end = keep_end
        # if not on a sequence edge start and end bits will be cropped
        # also, start on negative strand would ideally have special handling for weird padding
        self.edge_handle_start = edge_handle_start
        self.edge_handle_end = edge_handle_end
        self.is_plus_strand = is_plus_strand
        self.overlap_depth = math.ceil(chunk_size / overlap_offset)  # todo, test user settings once, not here
        self.overlap_offset = overlap_offset
        self.chunk_size = chunk_size

    def __repr__(self):
        sstr, estr = '(', ')'

        if self.edge_handle_end:
            estr = ']'
        if self.edge_handle_start:
            sstr = '['

        return 'SubBatch, {}edges{}, h5 indices: {}'.format(sstr, estr, self.h5_indices)

    @property
    def seq_length(self):
        return self.chunk_size * len(self.h5_indices)

    @property
    def sub_batch_size(self):
        return _n_batch_from_ori_chunks(len(self.h5_indices), self.overlap_depth)

    def sliding_coordinates(self):
        for i in range(0, self.seq_length - self.chunk_size + 1, self.overlap_offset):
            yield i, i + self.chunk_size

    def mk_sliding_overlaps_for_data_sub_batch(self, data_sub_batch):
        """makes sliding window of input data (x, or coverage data)"""
        # combine first 2 dimensions (i.e. merge chunks)
        dat = data_sub_batch.reshape([np.prod(data_sub_batch.shape[:2])] + list(data_sub_batch.shape[2:]))
        sliding_dat = [dat[start:end] for start, end in self.sliding_coordinates()]
        return sliding_dat

    def _overlap_preds(self, preds, core_length=10000):
        """take sliding-window predictions, and overlap (w/end clipping) to generate original coordinate predictions"""
        trim_by = (self.chunk_size - core_length) // 2
        ydim = preds[0].shape[-1]
        if ydim == self.chunk_size:
            ydim = 1
        preds_out = np.zeros(shape=(self.seq_length, ydim))
        counts = np.zeros(shape=(self.seq_length, 1))

        len_preds = len(preds)
        for i, chunk, start_end in zip(range(len_preds), preds, self.sliding_coordinates()):
            start, end = start_end
            #print('pre trim chunk shape is ', chunk.shape, )
            # cut to core, (but not sequence ends)
            if trim_by > 0:
                if i > 0:  # all except first seq
                    start += trim_by
                    chunk = chunk[trim_by:]
                if i < len_preds - 1:  # all except last seq
                    end -= trim_by
                    chunk = chunk[:-trim_by]
            elif trim_by < 0:  # sanity check only
                raise ValueError('invalid trim value: {}. Maybe core_length {} > chunk_size {}?'.format(
                    trim_by, core_length, chunk.shape[0]))
            sub_counts = counts[start:end]
            # average weighted by number of predictions counted at position so far
            #print(start, end, sub_counts.shape, chunk.shape, trim_by, f"counts shape is {counts.shape}")
            preds_out[start:end] = (preds_out[start:end] * sub_counts + chunk) / (sub_counts + 1)
            # increment counted so far
            counts[start:end] += 1
        preds_out = preds_out.reshape((len(self.h5_indices), self.chunk_size, ydim))
        return preds_out

    def overlap_and_edge_handle_preds(self, preds, core_length=10000):
        """overlaps sliding predictions, then crops as necessary on edges"""
        # the final sequences for what is cropped will come from previous/next batch instead
        # i.e. this should produce identical output regardless of batch size
        # as avoidable batch/sub-batch edge effects will be complete cropped here.
        clean_preds = self._overlap_preds(preds, core_length)
        clean_preds = self.edge_handle(clean_preds)
        return clean_preds

    def edge_handle(self, dat):
        """crops first and second array from output unless at sequence edge"""
        if not self.edge_handle_start:
            dat = dat[1:]
        if not self.edge_handle_end:
            dat = dat[:-1]
        return dat


# places where overlap will affect core functionality of HelixerSequence
class OverlapSeqHelper(object):
    """handles overlap-ready batching, as well as overlap-prep and overlapping there-of"""
    def __init__(self, contiguous_ranges, chunk_size=20000, max_batch_size=32, overlap_offset=5000, core_length=10000):
        # check validity of settings
        self.max_batch_size = max_batch_size
        self.core_length = core_length
        assert not chunk_size % overlap_offset, "chunk size must be divisible by overlap_offset"
        self.overlap_depth = chunk_size // overlap_offset
        min_functional_bs = 2 * self.overlap_depth + 1
        assert max_batch_size >= min_functional_bs, "batch_size is set too small to functionally overlap, " \
                                                    "set to at least {}, increase overlap_offset, or don't overlap" \
                                                    "".format(min_functional_bs)
        assert overlap_offset <= chunk_size - core_length, "change settings to over-, not under-lap"
        assert core_length > 0
        assert overlap_offset > 0

        # contiguous ranges should be created by .helpers.get_contiguous_ranges
        self.sliding_batches = self._mk_sliding_batches(contiguous_ranges=contiguous_ranges, chunk_size=chunk_size, overlap_offset=overlap_offset)

    def _mk_sliding_batches(self, contiguous_ranges, chunk_size, overlap_offset):
        max_n_chunks = _n_ori_chunks_from_batch_chunks(self.max_batch_size, self.overlap_depth)
        step = max_n_chunks - 2   # -2 bc ends will be cropped
        # most of these will effectively be final batches, but short seqs/ends may be grouped together (for efficiency)
        sub_batches = []

        for crange in contiguous_ranges:
            # step through sequence so that non-edges can have 1-chunk cropped off start/end
            # and regenerate original sequence with a simple concatenation there after
            for i in range(crange['start_i'], crange['end_i'], step):
                sub_batch_start = max(i - 1, crange['start_i'])  # pad 1 left (except seq edge)
                keep_start = max(i, crange['start_i'])
                keep_end = min(i + step, crange['end_i'])
                sub_batch_end = min(i + step + 1, crange['end_i'])   # pad 1 right (except seq edge)
                h5_indices = tuple(range(sub_batch_start, sub_batch_end))
                sub_batches.append(
                    SubBatch(h5_indices, is_plus_strand=crange['is_plus_strand'],
                             edge_handle_start=sub_batch_start == crange['start_i'],
                             edge_handle_end=i + step + 1 > crange['end_i'],
                             keep_start=keep_start,
                             keep_end=keep_end,
                             overlap_offset=overlap_offset, chunk_size=chunk_size)
                )

        # group into final batches, so as to keep total size <= max_batch_size
        # i.e. achieve consistent (& user adjustable) memory usage on graphics card
        sliding_batches = []
        batch = []
        batch_total_size = 0
        for sb in sub_batches:
            if batch_total_size + sb.sub_batch_size <= self.max_batch_size:
                batch.append(sb)
                batch_total_size += sb.sub_batch_size
            else:
                sliding_batches.append(batch)
                batch = [sb]
                batch_total_size = sb.sub_batch_size
        sliding_batches.append(batch)
        return sliding_batches

    def adjusted_epoch_length(self):
        """number of batches per epoch (given that we're overlapping)"""
        return len(self.sliding_batches)

    def h5_indices_of_batch(self, batch_idx):
        """concatenate indices from sub batches to give all indices for the batch at {batch_idx}"""
        sub_batches = self.sliding_batches[batch_idx]
        h5_indices = []
        for sb in sub_batches:
            h5_indices += sb.h5_indices
        return np.array(h5_indices)

    def make_input(self, batch_idx, data_batch):
        """make sliding input for prediction and overlapping (i.e. for X, maybe also for coverage)"""
        sub_batches = self.sliding_batches[batch_idx]
        sb_input_lengths = [len(sb.h5_indices) for sb in sub_batches]
        sb_input_starts = np.cumsum(sb_input_lengths) - sb_input_lengths
        x_as_list = []
        for start, length, sb in zip(sb_input_starts, sb_input_lengths, sub_batches):
            x_as_list.append(sb.mk_sliding_overlaps_for_data_sub_batch(data_batch[start:(start + length)]))
        sliding_input = np.concatenate(x_as_list)
        return sliding_input

    def overlap_predictions(self, batch_idx, predictions):
        """overlapping of sliding predictions to regenerate original dimensions"""
        sub_batches = self.sliding_batches[batch_idx]
        sub_batch_lengths = [sb.sub_batch_size for sb in sub_batches]
        sub_batch_starts = np.cumsum(sub_batch_lengths) - sub_batch_lengths
        out = []
        for start, length, sb in zip(sub_batch_starts, sub_batch_lengths, sub_batches):
            preds = predictions[start:(start + length)]
            out.append(sb.overlap_and_edge_handle_preds(preds, self.core_length))
        out = np.concatenate(out)
        n_expect = np.sum([sb.keep_end - sb.keep_start for sb in sub_batches])
        assert out.shape[0] == n_expect
        return out

    def subset_input(self, batch_idx, y_true_or_sw):
        """generate subset from data corresponding to _final_ predictions, i.e. to run y_true through during eval"""
        sub_batches = self.sliding_batches[batch_idx]
        sb_input_lengths = [len(sb.h5_indices) for sb in sub_batches]
        sb_input_starts = np.cumsum(sb_input_lengths) - sb_input_lengths
        dat_as_list = []
        for start, length, sb in zip(sb_input_starts, sb_input_lengths, sub_batches):
            dat_as_list.append(sb.edge_handle(y_true_or_sw[start:(start + length)]))
        return np.concatenate(dat_as_list)
