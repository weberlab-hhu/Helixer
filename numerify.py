"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import copy

import annotations
import type_enums
import gff_2_annotations
import slicer
import partitions
import sequences


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
        assert isinstance(data_slice, slicer.SequenceInfoHandler)
        self.data_slice = data_slice
        self.coordinates = self._get_coordinates()
        self.super_loci = self._get_super_loci()

    def _get_coordinates(self):
        coords = self.data_slice.data.coordinates
        assert len(coords) == 1, '{} != {}'.format(len(coords), 1)
        coords = coords[0]
        return coords

    def _get_super_loci(self):
        super_loci = set()
        for feature in self.coordinates.features:
            super_loci.add(feature.super_locus)
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
        for transcript in self.transcribeds_to_use():
            t_interp = TranscriptLocalReader(transcript)
            for prev_step, step in t_interp.transition_5p_to_3p_paired_steps(self.coordinates, is_plus_strand):
                self.update_matrix(matrix, prev_step, step)
        return matrix
        # todo, setup transcript local reader for each transcript.
        #   maybe grab primary transcript only
        #   transition setting numbers into matrix

    def transcribeds_to_use(self):
        for super_locus in self.super_loci:
            super_locus.make_all_handlers()
            for transcript in self.select_transcripts(super_locus):
                yield transcript.handler

    @staticmethod
    def select_transcripts(super_locus):
        for transcribed in super_locus.data.transcribeds:
            yield transcribed

    def update_matrix(self, matrix, prev_step, step):
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

    def update_matrix(self, matrix, prev_step, step):
        if prev_step.features is None:
            pass
        else:
            if step.any_erroneous_features():
                raise DataInterpretationError
            py_start, py_end = step.py_range(prev_step)
            py_start -= step.a_feature.coordinates.start
            py_end -= step.a_feature.coordinates.start
            labels = self.class_labels(prev_step.status)
            matrix[py_start:py_end, :] = np.logical_or(matrix[py_start:py_end, :], labels)
            print('labelling {}-{} as {}'.format(py_start, py_end, labels))



class TransitionAnnotationNumerifier(AnnotationNumerifier):
    def __init__(self, data_slice):
        super().__init__(data_slice, [12])

    types = {type_enums.TRANSCRIBED: 0,
             type_enums.CODING: 4,
             type_enums.INTRON: 8,
             type_enums.TRANS_INTRON: 8}

    bearings = {type_enums.START: 0,
                type_enums.END: 1,
                type_enums.OPEN_STATUS: 2,
                type_enums.CLOSE_STATUS: 3}

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
        py_start = step.a_feature.start - step.a_feature.coordinates.start
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
            return self.a_feature.start
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

        return gff_2_annotations.min_max(previous_at, current_at)

    def any_erroneous_features(self):
        #errors = [x.value for x in type_enums.ErrorFeature]
        return any([x.type.value == type_enums.ERROR for x in self.features])


class TranscriptLocalReader(annotations.TranscriptInterpBase):
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

    def local_transition_5p_to_3p(self, coords, is_plus_strand):
        status = annotations.TranscriptStatus()
        for aligned_features in self.stack_matches(self.sort_features(coords, is_plus_strand)):
            self.update_status(status, aligned_features)
            yield aligned_features, copy.deepcopy(status)

    def transition_5p_to_3p_paired_steps(self, coords, is_plus_strand):
        prev_step = StepHolder()
        for aligned_features, status in self.local_transition_5p_to_3p(coords, is_plus_strand):
            step = StepHolder(aligned_features, status)
            yield prev_step, step
            prev_step = step

