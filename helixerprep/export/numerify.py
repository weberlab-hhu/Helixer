"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from sqlalchemy.orm.exc import NoResultFound

from geenuff.base import types
from geenuff.base.orm import Coordinate, Genome
from ..core.orm import Mer


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
        self.transitions_matrix = None

        super().__init__()

    @abstractmethod
    def coord_to_matrices(self):
        pass

    def _gen_steps(self):
        partitioner = Stepper(end=self.coord.length, by=self.max_len)
        self.paired_steps = list(partitioner.step_to_end())
        if not self.is_plus_strand:
            self.paired_steps.reverse()

    def _slice_matrix(self, matrix, error_mask):
        data = []
        error_masks = []
        for prev, current in self.paired_steps:
            data_slice = matrix[prev:current]
            error_mask_slice = error_mask[prev:current]
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

    def coord_to_matrices(self):
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        self._zero_matrix()
        for i, bp in enumerate(self.coord.sequence):
            self.matrix[i] = AMBIGUITY_DECODE[bp]
        if not self.is_plus_strand:
            self.matrix = np.flip(self.matrix, axis=1)  # invert base

        data, error_mask = self._slice_matrix(self.matrix, self.error_mask)
        return data, error_mask


class AnnotationNumerifier(Numerifier, ABC):
    """Base class for numerification of the labels. Outputs a matrix that
    fits the sequence length of the coordinate but only for the provided features.
    This is done to support alternative splicing in the future.
    """
    feature_to_col = {
        types.GeenuffFeature.geenuff_transcript: 0,
        types.GeenuffFeature.geenuff_cds: 1,
        types.GeenuffFeature.geenuff_intron: 2,
    }
    error_type_values = [t.value for t in types.Errors]

    def __init__(self, coord, features, is_plus_strand, max_len, one_hot_transitions):
        Numerifier.__init__(self, n_cols=3, coord=coord, is_plus_strand=is_plus_strand,
                            max_len=max_len, dtype=np.int8)
        ABC.__init__(self)
        self.features = features
        self.onehot4_matrix = None
        self.onehot7_matrix = None

    def coord_to_matrices(self):
        self._zero_matrix()
        self._update_matrix_and_error_mask()
        self._encode_onehot4()
        self._encode_onehot7()
        onehot4_chunks, error_mask_chunks = self._slice_matrix(self.onehot4_matrix, self.error_mask)
        onehot7_chunks, _ = self._slice_matrix(self.onehot7_matrix, self.error_mask)
        return onehot4_chunks, onehot7_chunks, error_mask_chunks

    def _update_matrix_and_error_mask(self):
        for feature in self.features:
            # don't include features from the other strand
            if not feature.is_plus_strand == self.is_plus_strand:
                continue
            start = feature.start
            end = feature.end
            if not self.is_plus_strand:
                start, end = end + 1, start + 1
            if feature.type in AnnotationNumerifier.feature_to_col.keys():
                col = AnnotationNumerifier.feature_to_col[feature.type]
                self.matrix[start:end, col] = 1
            elif feature.type.value in AnnotationNumerifier.error_type_values:
                self.error_mask[start:end] = 0
            else:
                raise ValueError('Unknown feature type found: {}'.format(feature.type.value))

    def _encode_onehot4(self):
        # Class order: Intergenic, UTR, CDS, (non-coding Intron), Intron
        # This could be done in a more efficient way, but this way we may catch bugs
        # where non-standard classes are output in the multiclass output
        one_hot_matrix = np.zeros((self.matrix.shape[0], 4), dtype=bool)
        col_0, col_1, col_2 = self.matrix[:, 0], self.matrix[:, 1], self.matrix[:, 2]
        # Intergenic
        one_hot_matrix[:, 0] = np.logical_not(col_0)
        # UTR
        genic_non_coding = np.logical_and(col_0, np.logical_not(col_1))
        one_hot_matrix[:, 1] = np.logical_and(genic_non_coding, np.logical_not(col_2))
        # CDS
        one_hot_matrix[:, 2] = np.logical_and(np.logical_and(col_0, col_1), np.logical_not(col_2))
        # Introns
        one_hot_matrix[:, 3] = np.logical_and(col_0, col_2)
        self.onehot4_matrix = one_hot_matrix.astype(np.int8)
        assert np.all(np.count_nonzero(one_hot_matrix, axis=1) == 1)

    def _encode_onehot7(self):
        add = np.array([[0, 0, 0]])
        shifted_feature_matrix = np.vstack((self.matrix[1:], add))

        y_isTransition = np.logical_xor(self.matrix[:-1], shifted_feature_matrix[:-1]).astype(np.int8)
        y_direction_zero_to_one = np.logical_and(y_isTransition, self.matrix[1:]).astype(np.int8)
        y_direction_one_to_zero = np.logical_and(y_isTransition, self.matrix[:-1]).astype(np.int8)
        stack = np.hstack((y_direction_zero_to_one, y_direction_one_to_zero))
        
        add2 = np.array([[0, 0, 0, 0, 0, 0]]) 
        shape_stack = np.insert(stack, 0, add2, axis=0).astype(np.int8)
        shape_end_stack = np.insert(stack, len(stack), add2, axis=0).astype(np.int8)
        self.onehot7_matrix = np.logical_or(shape_stack, shape_end_stack).astype(np.int8)
        # ToDO maybe re-implement None-Transition if needed
        #y_is_no_Transition = np.all(np.logical_not(merged_stacks), axis=-1).astype(np.int8)

        #self.onehot7_matrix = np.concatenate((merged_stacks, y_is_no_Transition[:, None]), axis=1)


class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters.
    Currently just selects all Features of the given Coordinate.
    """

    def __init__(self, geenuff_exporter, coord, coord_features, is_plus_strand, max_len,
                 one_hot_transitions):
        assert isinstance(is_plus_strand, bool)
        assert isinstance(max_len, int) and max_len > 0
        self.geenuff_exporter = geenuff_exporter
        self.coord = coord
        self.max_len = max_len

        if not coord_features:
            logging.warning('Sequence {} has no annoations'.format(self.coord.seqid))

        self.anno_numerifier = AnnotationNumerifier(coord=self.coord,
                                                      features=self.coord.features,
                                                      is_plus_strand=is_plus_strand,
                                                      max_len=max_len,
                                                      one_hot_transitions=one_hot_transitions)

        self.seq_numerifier = SequenceNumerifier(coord=self.coord,
                                                 is_plus_strand=is_plus_strand,
                                                 max_len=max_len)

    def numerify(self):
        inputs, input_masks = self.seq_numerifier.coord_to_matrices()
        onehot4_labels, onehot7_labels, label_masks = self.anno_numerifier.coord_to_matrices()

        # flip the start ends back for - strand
        if self.anno_numerifier.is_plus_strand:
            start_ends = self.anno_numerifier.paired_steps
        else:
            start_ends = [(x[1], x[0]) for x in self.anno_numerifier.paired_steps]

        try:
            # need to hijack the session from geenuff_exporter as the Mer table does not exist there
            gc_content = (self.geenuff_exporter.session.query(Mer.count)
                .filter(Mer.coordinate == self.coord)
                .filter(Mer.mer_sequence == 'C')
                .one()[0])
        except NoResultFound:
            gc_content = 0
            logging.warning('No gc_content found for coord {}, set to 0 in the data'
                                 .format(self.coord.seqid))
        # do not output the input_masks as it is not used for anything
        out = {
            'inputs': inputs,
            'labels': onehot4_labels,
            'transitions': onehot7_labels,
            'label_masks': label_masks,
            'gc_contents': gc_content,
            'coord_lengths': self.coord.length,
            'species': self.coord.genome.species.encode('ASCII'),
            'seqids': self.coord.seqid.encode('ASCII'),
            'start_ends': start_ends,
        }
        return out
