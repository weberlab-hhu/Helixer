"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import logging
from abc import ABC, abstractmethod

from geenuff.base import types


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
    def __init__(self, n_cols, coord, is_plus_strand, max_len, dtype=np.float32):
        assert isinstance(n_cols, int)
        self.n_cols = n_cols
        self.coord = coord
        self.is_plus_strand = is_plus_strand
        self.max_len = max_len
        self.dtype = dtype
        self.matrix = None
        self.error_mask = None
        self.paired_steps = None
        self._gen_steps()  # sets self.paired_steps
        super().__init__()

    @abstractmethod
    def _unflipped_coord_to_matrix(self):
        pass

    def _gen_steps(self):
        partitioner = Stepper(end=self.coord.length, by=self.max_len)
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
                # invert directions
                data_slice = np.flip(data_slice, axis=0)
                error_mask_slice = np.flip(error_mask_slice, axis=0)
            data.append(data_slice)
            error_masks.append(error_mask_slice)
        return data, error_masks

    def _zero_matrix(self):
        length = len(self.coord.sequence)
        self.matrix = np.zeros((length, self.n_cols,), self.dtype)
        # 0 means error so this can be used directly as sample weight later on
        self.error_mask = np.ones((length,), np.int8)


class SequenceNumerifier(Numerifier):
    def __init__(self, coord, is_plus_strand, max_len):
        super().__init__(n_cols=4, coord=coord, is_plus_strand=is_plus_strand,
                         max_len=max_len, dtype=np.float16)

    def _unflipped_coord_to_matrix(self):
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        self._zero_matrix()
        for i, bp in enumerate(self.coord.sequence):
            self.matrix[i] = AMBIGUITY_DECODE[bp]
        if not self.is_plus_strand:
            self.matrix = np.flip(self.matrix, axis=1)  # invert base
        return self.matrix


class AnnotationNumerifier(Numerifier, ABC):
    """Base class for numerification of the labels. Outputs a matrix that
    fits the sequence length of the coordinate but only for the provided features.
    This is done to support alternative splicing in the future.
    """
    def __init__(self, n_cols, coord, features, is_plus_strand, max_len):
        Numerifier.__init__(self, n_cols=n_cols, coord=coord, is_plus_strand=is_plus_strand,
                            max_len=max_len, dtype=np.int8)
        ABC.__init__(self)
        self.features = features

    def _unflipped_coord_to_matrix(self):
        self._zero_matrix()
        self.update_matrix_and_error_mask()

    @abstractmethod
    def update_matrix_and_error_mask(self):
        pass


class BasePairAnnotationNumerifier(AnnotationNumerifier):
    feature_to_col = {
        types.GeenuffFeature.geenuff_transcript: 0,
        types.GeenuffFeature.geenuff_cds: 1,
        types.GeenuffFeature.geenuff_intron: 2,
     }
    error_type_values = [t.value for t in types.Errors]

    def __init__(self, coord, features, is_plus_strand, max_len):
        super().__init__(n_cols=3, coord=coord, features=features,
                         is_plus_strand=is_plus_strand, max_len=max_len)

    def update_matrix_and_error_mask(self):
        for feature in self.features:
            start = feature.start
            end = feature.end
            if not self.is_plus_strand:
                start, end = end + 1, start + 1
            if feature.type in BasePairAnnotationNumerifier.feature_to_col.keys():
                col = BasePairAnnotationNumerifier.feature_to_col[feature.type]
                self.matrix[start:end, col] = 1
            elif feature.type.value in BasePairAnnotationNumerifier.error_type_values:
                self.error_mask[start:end] = 0
            else:
                raise ValueError('Unknown feature type found: {}'.format(feature.type.value))


class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters.
    Currently just selects all Features of the given Coordinate.
    """
    def __init__(self, coord, is_plus_strand, max_len):
        assert isinstance(is_plus_strand, bool)
        assert isinstance(max_len, int) and max_len > 0
        if not coord.features:
            logging.warning('Sequence {} has no annoations'.format(coord.seqid))

        self.anno_numerifier = BasePairAnnotationNumerifier(coord=coord,
                                                            features=coord.features,
                                                            is_plus_strand=is_plus_strand,
                                                            max_len=max_len)
        self.seq_numerifier = SequenceNumerifier(coord=coord,
                                                 is_plus_strand=is_plus_strand,
                                                 max_len=max_len)

    def numerify(self):
        inputs, input_masks = self.seq_numerifier.coord_to_matrices()
        labels, label_masks = self.anno_numerifier.coord_to_matrices()

        coord = self.anno_numerifier.coord
        # flip the start ends back for - strand
        if self.anno_numerifier.is_plus_strand:
            start_ends = self.anno_numerifier.paired_steps
        else:
            start_ends = [(x[1], x[0]) for x in self.anno_numerifier.paired_steps]

        # do not output the input_masks yet as it is not used for anything
        out = {
            'inputs': inputs,
            'label_masks': label_masks,
            'labels': labels,
            'species': coord.genome.species.encode('ASCII'),
            'seqids': coord.seqid.encode('ASCII'),
            'start_ends': start_ends
        }
        return out
