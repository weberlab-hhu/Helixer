"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import copy

import geenuff
from helixerprep.datas.annotations import slicer
from ..core import partitions
from ..datas import sequences
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
    def __init__(self, shape, dtype=np.float):
        self.shape = shape
        self.dtype = dtype
        self.matrix = None

    def slice_to_matrix(self, data_slice, *args, **kwargs):
        raise NotImplementedError

    # this works only when each bp has it's own annotation; In other cases (e.g. where transitions are encoded)
    # this method must be overwritten
    def slice_to_matrices(self, data_slice, is_plus_strand, max_len, *args, **kwargs):
        matrix = self.slice_to_matrix(data_slice, is_plus_strand)
        partitioner = partitions.Stepper(end=matrix.shape[0], by=max_len)
        for prev, current in partitioner.step_to_end():
            yield matrix[prev:current]

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
    def __init__(self):
        super().__init__([4])

    def slice_to_matrix(self, data_slice, is_plus_strand=None, *args, **kwargs):
        assert is_plus_strand is not None  # todo, can I make this standard part of sig?
        assert isinstance(data_slice, sequences.SequenceSlice)
        length = data_slice.end - data_slice.start
        matrix = self._zeros(length)
        i = 0
        for subseq in data_slice.sequence:
            for bp in subseq:
                matrix[i] = AMBIGUITY_DECODE[bp.lower()]
                i += 1
        assert i == length, 'sequence length {} does not match expected {}'.format(i, length)
        return matrix


class AnnotationFoo(object):

    def __init__(self, data_slice):
        assert isinstance(data_slice, slicer.CoordinateHandler)
        self.data_slice = data_slice
        self.coordinates = self._get_coordinates()
        self.super_loci = self._get_super_loci()

    def _get_coordinates(self):
        return self.data_slice.data  # todo, clean up

    def _get_super_loci(self):
        super_loci = set()
        for feature in self.coordinates.features:
            for piece in feature.transcribed_pieces:
                super_loci.add(piece.transcribed.super_locus)
        out = []
        for sl in super_loci:
            handler = slicer.SuperLocusHandler()
            handler.add_data(sl)
            out.append(handler)
        return out


class AnnotationNumerifier(Numerifier, AnnotationFoo):
    def __init__(self, data_slice, shape, **kwargs):
        Numerifier.__init__(self, shape)
        AnnotationFoo.__init__(self, data_slice)

    def slice_to_matrix(self, data_slice, is_plus_strand=None, *args, **kwargs):
        assert is_plus_strand is not None
        length = self.coordinates.end - self.coordinates.start
        matrix = self._zeros(length)
        for transcript, super_locus in self.transcribeds_to_use():
            t_interp = TranscriptLocalReader(transcript, super_locus=super_locus)
            self.update_matrix(matrix, t_interp)
        return matrix
        # todo, setup transcript local reader for each transcript.
        #   maybe grab primary transcript only
        #   transition setting numbers into matrix

    def transcribeds_to_use(self):
        for super_locus in self.super_loci:
            super_locus.make_all_handlers()
            for transcript in self.select_transcripts(super_locus):
                yield transcript.handler, super_locus

    @staticmethod
    def select_transcripts(super_locus):
        for transcribed in super_locus.data.transcribeds:
            yield transcribed

    def update_matrix(self, matrix, transcript_interpreter):
        raise NotImplementedError


class DataInterpretationError(Exception):
    pass


# TODO, break or mask on errors
class BasePairAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, data_slice):
        super().__init__(data_slice, self._shape)

    @property
    def _shape(self):
        return [3]

    @staticmethod
    def class_labels(status):
        labs = (status.genic, status.in_translated_region, status.in_intron or status.in_trans_intron)
        return [float(x) for x in labs]

    def update_matrix(self, matrix, transcript_interpreter):
        errors = transcript_interpreter.error_ranges()
        errors = transcript_interpreter.filter_to_coordinate(coordinate_id=self.data_slice.data.id,
                                                             transcript_coordinates=errors)
        errors = list(errors)
        if errors:
            print(errors)
            raise DataInterpretationError

        for i_col, fn in self.row_fns(transcript_interpreter):
            self._update_row(matrix, i_col, fn())

    def _update_row(self, matrix, i_col, ranges):
        shift_by = self.data_slice.data.start
        for a_range in ranges:
            start = a_range.start - shift_by
            end = a_range.end - shift_by
            if not a_range.is_plus_strand:
                start, end = end + 1, start + 1
            matrix[start:end, i_col] = 1

    def row_fns(self, transcript_interpreter):
        assert isinstance(transcript_interpreter, geenuff.api.TranscriptInterpBase)
        return [(0, transcript_interpreter.transcribed_ranges),
                (1, transcript_interpreter.translated_ranges),
                (2, transcript_interpreter.intronic_ranges)]


class TransitionAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, data_slice):
        super().__init__(data_slice, [12])

    types = {geenuff.types.TRANSCRIBED: 0,
             geenuff.types.CODING: 4,
             geenuff.types.INTRON: 8,
             geenuff.types.TRANS_INTRON: 8}

    bearings = {geenuff.types.START: 0,
                geenuff.types.END: 1,
                geenuff.types.OPEN_STATUS: 2,
                geenuff.types.CLOSE_STATUS: 3}

    @staticmethod
    def class_labels(aligned_features):
        # ordered [start, end, status for types in transcribed, translated, any_intron]
        labs = [False] * 12
        for feature in aligned_features:
            i = TransitionAnnotationNumerifier.types[feature.type.value]
            i += TransitionAnnotationNumerifier.bearings[feature.bearing.value]
            labs[i] = True
        return labs

    def update_matrix(self, matrix, prev_step, step):
        if step.any_erroneous_features():
            raise DataInterpretationError
        py_start = step.a_feature.position - step.a_feature.coordinates.start
        labels = self.class_labels(step.features)
        matrix[py_start, :] = np.logical_or(
            matrix[py_start, :],
            labels)
        print('labelling {} as {}'.format(py_start, labels))

    def slice_to_matrices(self, data_slice, is_plus_strand, max_len, *args, **kwargs):
        raise NotImplementedError  # todo!


class StepHolder(object):
    def __init__(self, features=None, status=None, at=None):
        # todo, check not everything is none
        self._at = at
        self.features = features
        self.status = status

    @property
    def a_feature(self):
        return self.features[0]

    @property
    def at(self):
        if self._at is not None:
            return self._at
        elif self.features is not None:
            return self.a_feature.position
        else:
            raise ValueError

    def py_range(self, previous):
        current_at = self.at
        if previous.features is not None:
            previous_at = previous.at
            if not self.a_feature.is_plus_strand:
                # + 1 to move from - strand coordinates (,] to pythonic coordinates [,)
                current_at += 1
                previous_at += 1
        else:
            if self.a_feature.is_plus_strand:
                previous_at = self.a_feature.coordinates.start
            else:
                previous_at = self.a_feature.coordinates.end
                current_at += 1  # + 1 to move from - strand coordinate ( to pythonic coordinate [

        return helpers.min_max(previous_at, current_at)

    def any_erroneous_features(self):
        #errors = [x.value for x in type_enums.ErrorFeature]
        return any([x.type.value == geenuff.types.ERROR for x in self.features])


class TranscriptLocalReader(geenuff.api.TranscriptInterpBase):
    def sort_features(self, coords, is_plus_strand):
        # features from transcript
        features = []
        for piece in self.transcript.data.transcribed_pieces:
            for feature in piece.features:
                if feature.coordinates is coords and feature.is_plus_strand == is_plus_strand:
                    features.append(feature)
        # sort
        features = sorted(features, key=lambda x: x.pos_cmp_key())
        if not is_plus_strand:
            features.reverse()
        return features

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

    def filter_to_coordinate(self, coordinate_id, transcript_coordinates):
        # where transcript_coordinates should be a list of geenuff.api.TranscriptCoordinate instances
        matching_pieces = self.pieces_on_coordinate(coordinate_id)
        matching_piece_positions = [p.position for p in matching_pieces]
        for item in transcript_coordinates:
            if item.piece_position in matching_piece_positions:
                yield item

    #def local_transition_5p_to_3p(self, coords, is_plus_strand):
    #    status = geenuff.api.TranscriptStatusBase()
    #    for aligned_features in self.stack_matches(self.sort_features(coords, is_plus_strand)):
    #        self.update_status(status, aligned_features)
    #        yield aligned_features, copy.deepcopy(status)

    def transition_5p_to_3p_paired_steps(self, coords, is_plus_strand):
        prev_step = StepHolder()
        for aligned_features, status in self.local_transition_5p_to_3p(coords, is_plus_strand):
            step = StepHolder(aligned_features, status)
            yield prev_step, step
            prev_step = step


#### and now on to actual example gen
class ExampleMakerSeqMetaBP(object):
    def examples_from_slice(self, anno_slice, seq_slice, structured_genome, is_plus_strand, max_len):
        assert isinstance(seq_slice, sequences.SequenceSlice)
        assert isinstance(structured_genome, sequences.StructuredGenome)
        anno_nummerifier = BasePairAnnotationNumerifier(anno_slice)
        anno_gen = anno_nummerifier.slice_to_matrices(anno_slice, is_plus_strand=is_plus_strand, max_len=max_len)
        seq_nummerifier = SequenceNumerifier()
        seq_gen = seq_nummerifier.slice_to_matrices(seq_slice, is_plus_strand=is_plus_strand, max_len=max_len)
        total_Gbp = structured_genome.meta_info.total_bp / 10**9
        gc = structured_genome.meta_info.gc_content
        for anno in anno_gen:
            seq = next(seq_gen)
            out = {'labels': anno.flatten(), 'input': seq.flatten(), 'meta_Gbp': [total_Gbp], 'meta_gc': [gc]}
            yield out
