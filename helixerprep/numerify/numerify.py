"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import copy

import geenuff
from geenuff.base.transcript_interp import TranscriptInterpBase
from helixerprep.datas.annotations import slicer
from ..core import partitions
from ..core import helpers


# for now collapse everything to one vector (with or without pre-selection of primary transcript)
# 1x coding, utr, intron, intergenic (precedence on collapse and/or multi label)
# 1x TSS, TTS, status-transcribed, start, stop, status-translated, don-splice, acc-splice, status intron (")
# both of the above + trans-splice separate from splicing

# general structuring
# class defining data manipulation functions (Numerifier)
#   takes a slice & returns a matrix of values;
#   and can transform matrix <-> flat;
#   provides name
#
# class defining examples (ExampleMaker)
#   makes x, y pairs of data (as .dict)
#   handles processing of said data via calls to appropriate Numerifier


class Numerifier(object):
    def __init__(self, shape, is_plus_strand, dtype=np.float):
        self.shape = shape
        self.dtype = dtype
        self.matrix = None
        self.is_plus_strand = is_plus_strand

    def slice_to_matrix(self, *args, **kwargs):
        matrix = self.unflipped_slice_to_matrix(*args, **kwargs)
        if not self.is_plus_strand:
            matrix = np.flip(matrix, axis=1)
        return matrix

    def unflipped_slice_to_matrix(self, *arkgs, **kwargs):
        raise NotImplementedError

    # this works only when each bp has it's own annotation; In other cases (e.g. where transitions are encoded)
    # this method must be overwritten
    def slice_to_matrices(self, max_len, *args, **kwargs):
        matrix = self.unflipped_slice_to_matrix(*args, **kwargs)
        partitioner = partitions.Stepper(end=matrix.shape[0], by=max_len)
        paired_steps = list(partitioner.step_to_end())
        if not self.is_plus_strand:
            paired_steps.reverse()
        for prev, current in paired_steps:
            to_yield = matrix[prev:current]
            if self.is_plus_strand:
                yield to_yield
            else:
                yield np.flip(to_yield, axis=1)

    def flatten_matrix(self):
        assert isinstance(self.matrix, np.ndarray)
        return self.matrix.flatten()

    def deflatten_matrix(self, flattened):
        return np.reshape(flattened, [-1] + self.shape)

    def _zeros(self, length):
        return np.zeros([length] + self.shape, self.dtype)


AMBIGUITY_DECODE = {'c': [1., 0., 0., 0.],
                    'a': [0., 1., 0., 0.],
                    't': [0., 0., 1., 0.],
                    'g': [0., 0., 0., 1.],
                    'y': [0.5, 0., 0.5, 0.],
                    'r': [0., 0.5, 0., 0.5],
                    'w': [0., 0.5, 0.5, 0.],
                    's': [0.5, 0., 0., 0.5],
                    'k': [0., 0., 0.5, 0.5],
                    'm': [0.5, 0.5, 0., 0.],
                    'd': [0., 0.33, 0.33, 0.33],
                    'v': [0.33, 0.33, 0., 0.33],
                    'h': [0.33, 0.33, 0.33, 0.],
                    'b': [0.33, 0., 0.33, 0.33],
                    'n': [0.25, 0.25, 0.25, 0.25]}


class SequenceNumerifier(Numerifier):
    def __init__(self, is_plus_strand):
        super().__init__([4], is_plus_strand=is_plus_strand)
        # todo, add data_slice as attribute for consistency & ease of use

    def unflipped_slice_to_matrix(self, data_slice, *args, **kwargs):
        assert isinstance(data_slice, sequences.SequenceSlice)
        length = data_slice.end - data_slice.start
        matrix = self._zeros(length)
        i = 0
        for subseq in data_slice.sequence:
            for bp in subseq:
                matrix[i] = AMBIGUITY_DECODE[bp.lower()]
                i += 1
        assert i == length, 'sequence length {} does not match expected {}'.format(i, length)
        if not self.is_plus_strand:
            matrix = np.flip(matrix, axis=1)
        return matrix


class AnnotationFoo(object):

    def __init__(self, data_slice, is_plus_strand):
        assert isinstance(data_slice, slicer.CoordinateHandler)
        self.data_slice = data_slice
        self.is_plus_strand = is_plus_strand
        self.coordinates = self._get_coordinates()
        self.transcribed_pieces = self._get_pieces()

    def _get_coordinates(self):
        return self.data_slice.data  # todo, clean up

    def _get_pieces(self):
        pieces = set()
        for feature in self.coordinates.features:
            if feature.is_plus_strand == self.is_plus_strand:
                for piece in feature.transcribed_pieces:
                    pieces.add(piece)

        return pieces


class AnnotationNumerifier(Numerifier, AnnotationFoo):
    def __init__(self, data_slice, shape, is_plus_strand, **kwargs):
        assert isinstance(is_plus_strand, bool)
        Numerifier.__init__(self, shape, is_plus_strand)
        AnnotationFoo.__init__(self, data_slice, is_plus_strand)

    def unflipped_slice_to_matrix(self, *arkgs, **kwargs):
        length = self.coordinates.end - self.coordinates.start
        matrix = self._zeros(length)
        for piece, transcribed_handler, super_locus_handler in self.transcribeds_with_handlers():
            t_interp = TranscriptLocalReader(transcribed_handler, super_locus=super_locus_handler)
            self.update_matrix(matrix, piece, t_interp)
        return matrix

    def transcribeds_with_handlers(self):
        for piece in self.transcribed_pieces:
            transcribed_handler = geenuff.handlers.TranscribedHandlerBase()
            transcribed_handler.add_data(piece.transcribed)

            super_locus_handler = geenuff.handlers.SuperLocusHandlerBase()
            super_locus_handler.add_data(piece.transcribed.super_locus)
            yield piece, transcribed_handler, super_locus_handler

    @staticmethod
    def select_transcripts(super_locus):
        for transcribed in super_locus.data.transcribeds:
            yield transcribed

    def update_matrix(self, matrix, transcribed_piece, transcript_interpreter):
        raise NotImplementedError



class DataInterpretationError(Exception):
    pass


# TODO, break or mask on errors
class BasePairAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, data_slice, is_plus_strand):
        super().__init__(data_slice, self._shape, is_plus_strand=is_plus_strand)

    @property
    def _shape(self):
        return [3]

    @staticmethod
    def class_labels(status):
        labs = (status.genic, status.in_translated_region, status.in_intron or status.in_trans_intron)
        return [float(x) for x in labs]

    def update_matrix(self, matrix, transcribed_piece, transcript_interpreter):
        transcript_interpreter.check_no_errors(piece=transcribed_piece)

        for i_col, fn in self.col_fns(transcript_interpreter):
            ranges = transcript_interpreter.filter_to_piece(transcript_coordinates=fn(),
                                                            piece=transcribed_piece)
            self._update_row(matrix, i_col, ranges)

    def _update_row(self, matrix, i_col, ranges):
        shift_by = self.data_slice.data.start
        for a_range in ranges:  # todo, how to handle - strand??
            start = a_range.start - shift_by
            end = a_range.end - shift_by
            if not self.is_plus_strand:
                start, end = end + 1, start + 1
            matrix[start:end, i_col] = 1

    def col_fns(self, transcript_interpreter):
        assert isinstance(transcript_interpreter, TranscriptInterpBase)
        return [(0, transcript_interpreter.transcribed_ranges),
                (1, transcript_interpreter.translated_ranges),
                (2, transcript_interpreter.intronic_ranges)]


class TransitionAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, data_slice, is_plus_strand):
        super().__init__(data_slice, [12], is_plus_strand=is_plus_strand)
    def update_matrix(self, matrix, transcribed_piece, transcript_interpreter):
        transcript_interpreter.check_no_errors(transcribed_piece)

        for i_col, transitions in self.col_transitions(transcript_interpreter):
            transitions = transcript_interpreter.filter_to_piece(transcript_coordinates=transitions,
                                                                 piece=transcribed_piece)
            self._update_row(matrix, i_col, transitions)

    def _update_row(self, matrix, i_col, transitions):
        shift_by = self.data_slice.data.start
        for transition in transitions:  # todo, how to handle - strand?
            position = transition.start - shift_by
            matrix[position, i_col] = 1

    def col_transitions(self, transcript_interpreter):
        assert isinstance(transcript_interpreter, TranscriptInterpBase)
        #                transcribed, coding, intron, trans_intron
        # real start     0            4       8       8
        # open status    1            5       9       9
        # real end       2            6       10      10
        # close status   3            7       11      11
        out = []
        targ_types = [(0, geenuff.types.TRANSCRIBED),
                      (4, geenuff.types.CODING),
                      (8, geenuff.types.INTRON),
                      (8, geenuff.types.TRANS_INTRON)]

        for i in range(4):
            offset, targ_type = targ_types[i]
            j = 0
            for is_biological in [True, False]:
                for is_start_not_end in [True, False]:
                    out.append((offset + j, transcript_interpreter.get_by_type_and_bearing(
                        target_type=targ_type,
                        target_start_not_end=is_start_not_end,
                        target_is_biological=is_biological
                    )))
                    j += 1
        return out

    def slice_to_matrices(self, data_slice, max_len, *args, **kwargs):
        raise NotImplementedError  # todo!


#class StepHolder(object):
#    def __init__(self, features=None, status=None, at=None):
#        # todo, check not everything is none
#        self._at = at
#        self.features = features
#        self.status = status
#
#    @property
#    def a_feature(self):
#        return self.features[0]
#
#    @property
#    def at(self):
#        if self._at is not None:
#            return self._at
#        elif self.features is not None:
#            return self.a_feature.position
#        else:
#            raise ValueError
#
#    def py_range(self, previous):
#        current_at = self.at
#        if previous.features is not None:
#            previous_at = previous.at
#            if not self.a_feature.is_plus_strand:
#                # + 1 to move from - strand coordinates (,] to pythonic coordinates [,)
#                current_at += 1
#                previous_at += 1
#        else:
#            if self.a_feature.is_plus_strand:
#                previous_at = self.a_feature.coordinates.start
#            else:
#                previous_at = self.a_feature.coordinates.end
#                current_at += 1  # + 1 to move from - strand coordinate ( to pythonic coordinate [
#
#        return helpers.min_max(previous_at, current_at)
#
#    def any_erroneous_features(self):
#        #errors = [x.value for x in type_enums.ErrorFeature]
#        return any([x.type.value == geenuff.types.ERROR for x in self.features])


class TranscriptLocalReader(TranscriptInterpBase):

    @staticmethod  # todo, rather move to Handler?
    def coordinate_id_from_piece(piece):
        coord_ids = [f.coordinate_id for f in piece.features]
        assert all([x == coord_ids[0] for x in coord_ids]), \
            "not all coordinates match for features from piece: {}".format(coord_ids)
        return coord_ids[0]

    def pieces_on_coordinate(self, coordinate_id):
        out = []
        for piece in self.transcript.data.transcribed_pieces:
            if self.coordinate_id_from_piece(piece) == coordinate_id:
                out.append(piece)
        return out

    @staticmethod
    def filter_to_piece(piece, transcript_coordinates):
        # where transcript_coordinates should be a list of TranscriptCoordinate instances
        for item in transcript_coordinates:
            if item.piece_position == piece.position:
                yield item

    def check_no_errors(self, piece):
        errors = self.error_ranges()
        errors = self.filter_to_piece(piece=piece,
                                      transcript_coordinates=errors)
        errors = list(errors)
        if errors:
            raise DataInterpretationError


#### and now on to actual example gen
class ExampleMakerSeqMetaBP(object):
    def examples_from_slice(self, anno_slice, seq_slice, structured_genome, is_plus_strand, max_len):
        assert isinstance(seq_slice, sequences.SequenceSlice)
        assert isinstance(structured_genome, sequences.StructuredGenome)
        anno_nummerifier = BasePairAnnotationNumerifier(anno_slice, is_plus_strand=is_plus_strand)
        anno_gen = anno_nummerifier.slice_to_matrices(max_len=max_len)
        seq_nummerifier = SequenceNumerifier(is_plus_strand=is_plus_strand)
        seq_gen = seq_nummerifier.slice_to_matrices(data_slice=seq_slice, max_len=max_len)
        total_Gbp = structured_genome.meta_info.total_bp / 10**9
        gc = structured_genome.meta_info.gc_content
        for anno in anno_gen:
            seq = next(seq_gen)
            out = {'labels': anno.flatten(), 'input': seq.flatten(), 'meta_Gbp': [total_Gbp], 'meta_gc': [gc]}
            yield out
