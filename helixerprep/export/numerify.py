"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import copy
from abc import ABC, abstractmethod

import geenuff
from geenuff.base.orm import Coordinate
from geenuff.base.transcript_interp import TranscriptInterpBase
from geenuff.base.handlers import SuperLocusHandlerBase, TranscribedHandlerBase
from ..core import handlers, helpers


# for now collapse everything to one vector (with or without pre-selection of primary transcript)
# 1x coding, utr, intron, intergenic (precedence on collapse and/or multi label)
# 1x TSS, TTS, status-transcribed, start, stop, status-translated, don-splice, acc-splice, status intron (")
# both of the above + trans-splice separate from splicing

# general structuring
# class defining data manipulation functions (Numerifier)
#   takes a coord & returns a matrix of values;
#   and can transform matrix <-> flat;
#   provides name
#
# class defining examples (ExampleMaker)
#   makes x, y pairs of data (as .dict)
#   handles processing of said data via calls to appropriate Numerifier


AMBIGUITY_DECODE = {
    'C': [1., 0., 0., 0.],
    'A': [0., 1., 0., 0.],
    'T': [0., 0., 1., 0.],
    'G': [0., 0., 0., 1.],
    'Y': [0.5, 0., 0.5, 0.],
    'R': [0., 0.5, 0., 0.5],
    'W': [0., 0.5, 0.5, 0.],
    'S': [0.5, 0., 0., 0.5],
    'K': [0., 0., 0.5, 0.5],
    'M': [0.5, 0.5, 0., 0.],
    'D': [0., 0.33, 0.33, 0.33],
    'V': [0.33, 0.33, 0., 0.33],
    'H': [0.33, 0.33, 0.33, 0.],
    'B': [0.33, 0., 0.33, 0.33],
    'N': [0.25, 0.25, 0.25, 0.25]
}

class DataInterpretationError(Exception):
    pass


class Stepper(object):
    def __init__(self, end, by):
        self.at = 0
        self.end = end
        self.by = by

    def step(self):
        prev = self.at
        # fits twice or more, just step
        if prev + self.by * 2 <= self.end:
            new = prev + self.by
        # fits less than twice, take half way point (to avoid an end of just a few bp)
        elif prev + self.by < self.end:
            new = prev + (self.end - prev) // 2
        # doesn't fit at all
        else:
            new = self.end
        self.at = new
        return prev, new

    def step_to_end(self):
        while self.at < self.end:
            yield self.step()


class Numerifier(ABC):
    def __init__(self, n_cols, coord_handler, is_plus_strand, max_len, dtype=np.float32):
        assert isinstance(n_cols, int)
        self.n_cols = n_cols
        self.coord_handler = coord_handler
        self.is_plus_strand = is_plus_strand
        self.max_len = max_len
        self.dtype = dtype
        self.matrix = None
        self.error_mask = None
        self._gen_steps()  # sets self.paired_steps
        super().__init__()

    @property
    def coordinate(self):
        return self.coord_handler.data  # todo, clean up

    @abstractmethod
    def _unflipped_coord_to_matrix(self):
        pass

    def _gen_steps(self):
        partitioner = Stepper(end=len(self.coordinate.sequence), by=self.max_len)
        self.paired_steps = list(partitioner.step_to_end())
        if not self.is_plus_strand:
            self.paired_steps.reverse()

    def coord_to_matrices(self):
        """This works only when each bp has it's own annotation In other cases
        (e.g. where transitions are encoded) this method must be overwritten
        """
        self._unflipped_coord_to_matrix()
        data = []
        error_masks = []
        for prev, current in self.paired_steps:
            data_slice = self.matrix[prev:current]
            error_mask_slice = self.error_mask[prev:current]
            if not self.is_plus_strand:
                data_slice = np.flip(data_slice, axis=0)  # invert direction
                error_mask_slice = np.flip(error_mask_slice, axis=0)
            data.append(data_slice)
            error_masks.append(error_mask_slice)
        return data, error_masks

    def _zero_matrix(self):
        length = len(self.coordinate.sequence)
        self.matrix = np.zeros((length, self.n_cols,), self.dtype)
        self.error_mask = np.zeros((length,), np.int8)


class SequenceNumerifier(Numerifier):
    def __init__(self, coord_handler, is_plus_strand, max_len):
        super().__init__(n_cols=4, coord_handler=coord_handler,
                         is_plus_strand=is_plus_strand, max_len=max_len)

    def _unflipped_coord_to_matrix(self):
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        self._zero_matrix()
        for i, bp in enumerate(self.coordinate.sequence):
            self.matrix[i] = AMBIGUITY_DECODE[bp]
        if not self.is_plus_strand:
            self.matrix = np.flip(self.matrix, axis=1)  # invert base
        return self.matrix


class AnnotationNumerifier(Numerifier, ABC):
    """Base class for numerification of the labels. Outputs a matrix that
    fits the sequence length of the coordinate but only for the provided features.
    This is done to support alternative splicing in the future.
    """
    def __init__(self, n_cols, coord_handler, features, is_plus_strand, max_len):
        Numerifier.__init__(self, n_cols=n_cols, coord_handler=coord_handler,
                            is_plus_strand=is_plus_strand, max_len=max_len)
        ABC.__init__(self)
        self.features = features

    # todo integrate into self.transcribeds_with_handlers and maybe use an orm core query
    def _get_transcribed_pieces(self):
        pieces = set()
        for feature in self.features:
            if feature.is_plus_strand == self.is_plus_strand:
                for piece in feature.transcribed_pieces:
                    pieces.add(piece)
        return pieces

    def _unflipped_coord_to_matrix(self):
        self._zero_matrix()
        for piece, transcribed_handler, super_locus_handler in self.transcribeds_with_handlers():
            t_interp = TranscriptLocalReader(transcribed_handler, super_locus=super_locus_handler)
            self.update_matrix(piece, t_interp)
            self.update_error_mask(piece, t_interp)

    def transcribeds_with_handlers(self):
        for piece in self._get_transcribed_pieces():
            transcribed_handler = TranscribedHandlerBase(piece.transcribed)
            super_locus_handler = SuperLocusHandlerBase(piece.transcribed.super_locus)
            yield piece, transcribed_handler, super_locus_handler

    def update_error_mask(self, piece, t_interp):
        errors = t_interp.error_ranges()
        piece_errors = t_interp.filter_to_piece(piece=piece, transcript_coordinates=errors)
        for p in piece_errors:
            self.error_mask[p.start:p.end] = 1

    @abstractmethod
    def update_matrix(self, transcribed_piece, transcript_interpreter):
        pass


class TranscriptLocalReader(TranscriptInterpBase):
    @staticmethod
    def filter_to_piece(piece, transcript_coordinates):
        # where transcript_coordinates should be a list of TranscriptCoordinate instances
        for item in transcript_coordinates:
            if item.piece_position == piece.position:
                yield item


# todo, break or mask on errors
class BasePairAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, coord_handler, features, is_plus_strand, max_len):
        super().__init__(n_cols=3, coord_handler=coord_handler, features=features,
                         is_plus_strand=is_plus_strand, max_len=max_len)

    @staticmethod
    def class_labels(status):
        labels = (
            status.genic,
            status.in_translated_region,
            status.in_intron or status.in_trans_intron,
        )
        return [float(x) for x in labels]

    def update_matrix(self, transcribed_piece, transcript_interpreter):
        for i_col, fn in self.col_fns(transcript_interpreter):
            ranges = transcript_interpreter.filter_to_piece(transcript_coordinates=fn(),
                                                            piece=transcribed_piece)
            self._update_row(i_col, ranges)

    def _update_row(self, i_col, ranges):
        shift_by = self.coordinate.start
        for a_range in ranges:  # todo, how to handle - strand??
            start = a_range.start - shift_by
            end = a_range.end - shift_by
            if not self.is_plus_strand:
                start, end = end + 1, start + 1
            self.matrix[start:end, i_col] = 1

    def col_fns(self, transcript_interpreter):
        assert isinstance(transcript_interpreter, TranscriptInterpBase)
        return [(0, transcript_interpreter.transcribed_ranges),
                (1, transcript_interpreter.translated_ranges),
                (2, transcript_interpreter.intronic_ranges)]



class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters.
    Currently just selects all Features of the given Coordinate.
    """
    def __init__(self, coord_handler, is_plus_strand, max_len):
        assert isinstance(coord_handler, handlers.CoordinateHandler)
        assert isinstance(is_plus_strand, bool)
        assert isinstance(max_len, int) and max_len > 0
        self.anno_numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                                            features=coord_handler.data.features,
                                                            is_plus_strand=is_plus_strand,
                                                            max_len=max_len)
        self.seq_numerifier = SequenceNumerifier(coord_handler=coord_handler,
                                                 is_plus_strand=is_plus_strand,
                                                 max_len=max_len)

    def numerify(self):
        inputs, input_masks = self.seq_numerifier.coord_to_matrices()
        labels, label_masks = self.anno_numerifier.coord_to_matrices()

        # do not output the input_masks yet as it is not used for anything
        out = {
            'inputs': inputs,
            'label_masks': label_masks,
            'labels': labels,
        }
        return out
