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
        if prev + self.by < self.end:
            new = prev + self.by
        else:
            new = self.end
        self.at = new
        return prev, new

    def step_to_end(self):
        while self.at < self.end:
            yield self.step()


class Numerifier(ABC):
    def __init__(self, n_cols, coord, max_len, dtype=np.float32):
        assert isinstance(n_cols, int)
        self.n_cols = n_cols
        self.coord = coord
        self.max_len = max_len
        self.dtype = dtype
        self.matrix = None
        self.error_mask = None
        # set paired steps
        partitioner = Stepper(end=self.coord.length, by=self.max_len)
        self.paired_steps = list(partitioner.step_to_end())

        super().__init__()

    @abstractmethod
    def coord_to_matrices(self):
        """Method to be called from outside. Numerifies both strands."""
        pass

    def _slice_matrices(self, is_plus_strand, *argv):
        """Slices (potentially) multiple matrices in the same way according to self.paired_steps"""
        assert len(argv) > 0, 'Need a matrix to slice'
        all_slices = [[] for _ in range(len(argv))]
        # reverse steps on minus strand
        steps = self.paired_steps if is_plus_strand else self.paired_steps[::-1]
        for prev, current in steps:
            for matrix, slices in zip(argv, all_slices):
                data_slice = matrix[prev:current]
                if not is_plus_strand:
                    # invert directions
                    data_slice = np.flip(data_slice, axis=0)
                slices.append(data_slice)
        return all_slices

    def _zero_matrix(self):
        length = len(self.coord.sequence)
        self.matrix = np.zeros((length, self.n_cols,), self.dtype)
        # 0 means error so this can be used directly as sample weight later on
        self.error_mask = np.ones((length,), np.int8)


class SequenceNumerifier(Numerifier):
    def __init__(self, coord, max_len):
        super().__init__(n_cols=4, coord=coord, max_len=max_len, dtype=np.float16)

    def coord_to_matrices(self):
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        self._zero_matrix()
        # plus strand, actual numerification of the sequence
        for i, bp in enumerate(self.coord.sequence):
            self.matrix[i] = AMBIGUITY_DECODE[bp]
        # very important to copy here
        data_plus, error_mask_plus = self._slice_matrices(True,
                                                          np.copy(self.matrix),
                                                          np.copy(self.error_mask))

        # minus strand, just flip
        self.matrix = np.flip(self.matrix, axis=1)  # invert base
        data_minus, error_mask_minus = self._slice_matrices(False,
                                                            self.matrix,
                                                            self.error_mask)
        # put everything together
        data = data_plus + data_minus
        error_masks = error_mask_plus + error_mask_minus
        return data, error_masks


class AnnotationNumerifier(Numerifier):
    """Class for the numerification of the labels. Outputs a matrix that
    fits the sequence length of the coordinate but only for the provided features.
    This is done to support alternative splicing in the future.
    """
    feature_to_col = {
        types.GeenuffFeature.geenuff_transcript: 0,
        types.GeenuffFeature.geenuff_cds: 1,
        types.GeenuffFeature.geenuff_intron: 2,
     }

    def __init__(self, coord, features, max_len, one_hot=True):
        Numerifier.__init__(self, n_cols=3, coord=coord, max_len=max_len, dtype=np.int8)
        self.features = features
        self.one_hot = one_hot
        self.coord = coord
        self._zero_gene_lengths()

    def _zero_gene_lengths(self):
        self.gene_lengths = np.zeros((len(self.coord.sequence),), dtype=np.uint32)

    def coord_to_matrices(self):
        """Always numerifies both strands one after the other."""
        plus_strand = self._encode_strand(True)
        minus_strand = self._encode_strand(False)

        # put everything together
        combined_data = tuple(plus_strand[i] + minus_strand[i] for i in range(len(plus_strand)))
        return combined_data

    def _encode_strand(self, is_plus_strand):
        self._zero_matrix()
        self._zero_gene_lengths()
        self._update_matrix_and_error_mask(is_plus_strand=is_plus_strand)

        # encoding of transitions
        binary_transition_matrix = self._encode_transitions()

        # encoding of the actual labels and slicing; generation of error mask and gene length array
        if self.one_hot:
            label_matrix = self._encode_onehot4()

        else:
            label_matrix = self.matrix
        matrices = self._slice_matrices(is_plus_strand,
                                        label_matrix,
                                        self.error_mask,
                                        self.gene_lengths,
                                        binary_transition_matrix)
        return matrices

    def _update_matrix_and_error_mask(self, is_plus_strand):
        for feature in self.features:
            # don't include features from the other strand
            if not feature.is_plus_strand == is_plus_strand:
                continue
            start = feature.start
            end = feature.end
            if not is_plus_strand:
                start, end = end + 1, start + 1
            if feature.type in AnnotationNumerifier.feature_to_col.keys():
                col = AnnotationNumerifier.feature_to_col[feature.type]
                self.matrix[start:end, col] = 1
            elif feature.type.value in types.geenuff_error_type_values:
                self.error_mask[start:end] = 0
            else:
                raise ValueError('Unknown feature type found: {}'.format(feature.type.value))
            # also fill self.gene_lengths
            # give precedence for the longer transcript if present
            if feature.type.value == types.GEENUFF_TRANSCRIPT:
                length_arr = np.full((end - start,), end - start)
                self.gene_lengths[start:end] = np.maximum(self.gene_lengths[start:end], length_arr)


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
        assert np.all(np.count_nonzero(one_hot_matrix, axis=1) == 1)

        one_hot4_matrix = one_hot_matrix.astype(np.int8)
        return one_hot4_matrix

    def _encode_transitions(self):
        add = np.array([[0, 0, 0]])
        shifted_feature_matrix = np.vstack((self.matrix[1:], add))

        y_isTransition = np.logical_xor(self.matrix[:-1], shifted_feature_matrix[:-1]).astype(np.int8)
        y_direction_zero_to_one = np.logical_and(y_isTransition, self.matrix[1:]).astype(np.int8)
        y_direction_one_to_zero = np.logical_and(y_isTransition, self.matrix[:-1]).astype(np.int8)
        stack = np.hstack((y_direction_zero_to_one, y_direction_one_to_zero))

        add2 = np.array([[0, 0, 0, 0, 0, 0]])
        shape_stack = np.insert(stack, 0, add2, axis=0).astype(np.int8)
        shape_end_stack = np.insert(stack, len(stack), add2, axis=0).astype(np.int8)
        binary_transitions  = np.logical_or(shape_stack, shape_end_stack).astype(np.int8)
        return binary_transitions # 6 columns, one for each switch (+TR, +CDS, +In, -TR, -CDS, -In)

class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters. Selects all Features of the given Coordinate.
    """
    @staticmethod
    def numerify(geenuff_exporter, coord, coord_features, max_len, one_hot=True):
        assert isinstance(max_len, int) and max_len > 0
        if not coord_features:
            logging.warning('Sequence {} has no annoations'.format(coord.seqid))

        anno_numerifier = AnnotationNumerifier(coord=coord, features=coord_features, max_len=max_len,
                                               one_hot=one_hot)
        seq_numerifier = SequenceNumerifier(coord=coord, max_len=max_len)

        # returns results for both strands, with the plus strand first in the list
        inputs, input_masks = seq_numerifier.coord_to_matrices()
        labels, label_masks, gene_lengths, transitions = anno_numerifier.coord_to_matrices()

        start_ends = anno_numerifier.paired_steps
        # flip the start ends back for - strand and append
        start_ends += [(x[1], x[0]) for x in anno_numerifier.paired_steps[::-1]]

        # do not output the input_masks as it is not used for anything
        out = {
            'inputs': inputs,
            'labels': labels,
            'label_masks': label_masks,
            'gene_lengths': gene_lengths,
            'transitions': transitions,
            'species': [coord.genome.species.encode('ASCII')] * len(inputs),
            'seqids': [coord.seqid.encode('ASCII')] * len(inputs),
            'start_ends': start_ends,
        }
        return out
