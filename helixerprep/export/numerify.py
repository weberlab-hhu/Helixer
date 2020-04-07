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
    def __init__(self, n_cols, coord, max_len, dtype=np.float32, start=0, end=None):
        if end is None:
            end = coord.length
        assert isinstance(n_cols, int)
        self.n_cols = n_cols
        self.coord = coord
        self.max_len = max_len
        self.dtype = dtype
        self.matrix = None
        self.error_mask = None
        self.start = start
        self.end = end
        # set paired steps
        partitioner = Stepper(end=self.end, by=self.max_len)
        partitioner.at = self.start
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
        length = self.end - self.start
        self.matrix = np.zeros((length, self.n_cols,), self.dtype)
        # 0 means error so this can be used directly as sample weight later on
        self.error_mask = np.ones((length,), np.int8)


class SequenceNumerifier(Numerifier):
    def __init__(self, coord, max_len, start, end):
        super().__init__(n_cols=4, coord=coord, max_len=max_len, dtype=np.float16, start=start, end=end)

    def coord_to_matrices(self):
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        self._zero_matrix()
        # plus strand, actual numerification of the sequence
        for i, bp in enumerate(self.coord.sequence[self.start:self.end]):
            self.matrix[i] = AMBIGUITY_DECODE[bp]
        # very important to copy here
        data_plus = self._slice_matrices(True,
                                         np.copy(self.matrix))

        # minus strand
        self.matrix = np.flip(self.matrix, axis=1)  # complementary base
        data_minus = self._slice_matrices(False,  # slice matrix will reverse direction
                                          self.matrix)

        # put everything together
        data = {'plus': data_plus, 'minus': data_minus}
        return data


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
    # todo, major refactor so that everything is handled in a symmetric fashion, and so that it's possible
    #  to skip onehot, sample_weights, gene_length, or transitions without a maze of if/else statements
    #  maybe have first pass & second pass matrix gen functions, and loop through those that exist at each step??
    #  Second pass could also be written to h5 in a second round to reduce mem usage if need be. Or first pass is
    #  a generator that autodetects splittable intergenic regions every 10mb or so.
    
    def __init__(self, coord, features, max_len, one_hot=True, start=0, end=None):
        Numerifier.__init__(self, n_cols=3, coord=coord, max_len=max_len, dtype=np.int8, start=start, end=end)
        self.features = features
        self.one_hot = one_hot
        self.start = start
        self.end = end
        self.coord = coord
        self._zero_gene_lengths()

    def _zero_gene_lengths(self):
        self.gene_lengths = np.zeros((self.end - self.start,), dtype=np.uint32)

    def coord_to_matrices(self):
        """Always numerifies both strands one after the other."""
        plus_strand = self._encode_strand(True)
        minus_strand = self._encode_strand(False)

        # put everything together
        combined_data = tuple(({'plus': plus_strand[i], 'minus': minus_strand[i]} for i in range(len(plus_strand))))
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
            start = feature.start - self.start  # self.start used as offset for writing chunk
            end = feature.end - self.start
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

        y_is_transition = np.logical_xor(self.matrix[:-1], shifted_feature_matrix[:-1]).astype(np.int8)
        y_direction_zero_to_one = np.logical_and(y_is_transition, self.matrix[1:]).astype(np.int8)
        y_direction_one_to_zero = np.logical_and(y_is_transition, self.matrix[:-1]).astype(np.int8)
        stack = np.hstack((y_direction_zero_to_one, y_direction_one_to_zero))

        add2 = np.array([[0, 0, 0, 0, 0, 0]])
        shape_stack = np.insert(stack, 0, add2, axis=0).astype(np.int8)
        shape_end_stack = np.insert(stack, len(stack), add2, axis=0).astype(np.int8)
        binary_transitions = np.logical_or(shape_stack, shape_end_stack).astype(np.int8)
        return binary_transitions  # 6 columns, one for each switch (+TR, +CDS, +In, -TR, -CDS, -In)


class MatAndInfo:
    """organizes data and meta info for post-processing and saving a matrix"""
    def __init__(self, key, matrix, dtype):
        self.key = key
        self.matrix = matrix
        self.dtype = dtype

    def __repr__(self):
        return "key: {}, matrix shape: {}, matrix dtype {}: target dtype {}".format(self.key, self.matrix.shape,
                                                                                    self.matrix.dtype, self.dtype)


class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters. Selects all Features of the given Coordinate.
    """

    @staticmethod
    def pad(d, chunk_size):
        n_seqs = len(d)
        # insert all the sequences so that 0-padding is added if needed
        shape = tuple([n_seqs, chunk_size] + list(d[0].shape[1:]))
        padded_d = np.zeros(shape, dtype=d[0].dtype)
        for j in range(n_seqs):
            padded_d[j, :len(d[j])] = d[j]
        return padded_d

    @staticmethod
    def numerify(coord, coord_features, max_len, one_hot=True, mode=('X', 'y', 'anno_meta', 'transitions'),
                 write_by=5000000):
        assert isinstance(max_len, int) and max_len > 0, 'what is {} of type {}'.format(max_len, type(max_len))
        coord_features = sorted(coord_features, key=lambda f: min(f.start, f.end))  # sort by ~ +strand start
        split_finder = SplitFinder(features=coord_features, write_by=write_by, coord_length=coord.len,
                                   chunk_size=max_len)

        for f_set, bp_coord, h5_coord in \
                zip(split_finder.split_features(), split_finder.coords, split_finder.relative_h5_coords):
            start, end = bp_coord

            # todo, run for just start-end and f_set
            anno_numerifier = AnnotationNumerifier(coord=coord, features=f_set, max_len=max_len,
                                                   one_hot=one_hot, start=start, end=end)
            seq_numerifier = SequenceNumerifier(coord=coord, max_len=max_len, start=start, end=end)

            # returns results for both strands, with the plus strand first in the list
            # todo, this needs to change so that + & - strand are returned and handled as two arrays
            xb = seq_numerifier.coord_to_matrices()
            yb, sample_weightsb, gene_lengthsb, transitionsb = anno_numerifier.coord_to_matrices()

            for strand in ['plus', 'minus']:
                x, y, sample_weights, gene_lengths, transitions = \
                    (CoordNumerifier.pad(x, max_len) for x in [xb[strand],
                                                               yb[strand],
                                                               sample_weightsb[strand],
                                                               gene_lengthsb[strand],
                                                               transitionsb[strand]])

                # todo, move to be part of anno numerifier??
                if strand == 'plus':
                    start_ends = anno_numerifier.paired_steps
                else:
                    # flip the start ends back for - strand
                    start_ends = [(x[1], x[0]) for x in anno_numerifier.paired_steps[::-1]]
                start_ends = np.array(start_ends, dtype=np.int64)

                # mark examples from featureless coordinate / assume there is no trustworthy annotation
                if not coord_features:
                    logging.warning('Sequence {} has no annotations'.format(coord.seqid))
                    is_annotated = [0] * len(x)
                else:
                    is_annotated = [1] * len(x)
                is_annotated = np.array(is_annotated, dtype=np.bool)

                # additional derived matrices
                err_samples = np.any(sample_weights == 0, axis=1)
                # just one entry per chunk
                if one_hot:
                    fully_intergenic_samples = np.all(y[:, :, 0] == 0, axis=1)
                else:
                    fully_intergenic_samples = np.all(y[:, :, 0] == 1, axis=1)

                # do not output the input_masks as it is not used for anything
                out = (MatAndInfo('y', y, 'int8'),  # y should always be first (bc currently we always want it)
                       MatAndInfo('X', x, 'float16'),
                       MatAndInfo('sample_weights', sample_weights, 'int8'),
                       MatAndInfo('gene_lengths', gene_lengths, 'uint32'),
                       MatAndInfo('transitions', transitions, 'int8'),
                       MatAndInfo('err_samples', err_samples, 'bool'),
                       MatAndInfo('fully_intergenic_samples', fully_intergenic_samples,  'bool'),
                       MatAndInfo('species', np.array([coord.genome.species.encode('ASCII')] * len(x)), 'S25'),
                       MatAndInfo('seqids', np.array([coord.seqid.encode('ASCII')] * len(x)), 'S50'),
                       MatAndInfo('start_ends', start_ends, 'int64'),
                       MatAndInfo('is_annotated', is_annotated, 'bool'))
                yield out, h5_coord[strand]  # todo, yield will of course make more sense once we chunk up the data some, and once the
                #             other numerifiers also yield...


class SplitFinder:
    # todo, tryout and test
    def __init__(self, features, write_by, coord_length, chunk_size):
        assert not write_by % chunk_size, "number of bp to write at once 'write_by' must be a multiple of 'chunk_size'"
        self.features = features
        self.write_by = write_by  # target writing this many bp to the h5 file at once
        self.coord_length = coord_length
        self.chunk_size = chunk_size
        self.splits = tuple(self._find_splits())
        self.relative_h5_coords = tuple(self._get_rel_h5_coords_for_splits())

    @property
    def coords(self):
        starts = [0] + list(self.splits)[:-1]
        return tuple(zip(starts, self.splits))

    def split_features(self):
        """get all features from start of list that aren't passed _to_"""
        i = 0
        in_split = []
        in_next_split = []
        for end in self.splits:
            feature = self.features[i]
            while self._feature_not_past(feature, end):
                in_split.append(feature)
                if self._feature_ends_after(feature, end):  # basically this is 'if overlaps end' in context
                    in_next_split.append(feature)
                i += 1
                feature = self.features[i]
            yield in_split
            in_split = in_next_split
            in_next_split = []

    def _get_rel_h5_coords_for_splits(self):
        """calculates where to write the +/- strand super-chunk splits in the h5 file"""
        # calculate the positive strand first
        postive_h5_ends = []
        postive_h5_starts = [0]
        for end in self.splits:
            p_h5 = end // self.chunk_size
            if end % self.chunk_size:  # end of seq, will be padded to full chunk size
                p_h5 += 1
            else:
                postive_h5_starts.append(p_h5)
            postive_h5_ends.append(p_h5)

        postive_h5s = zip(postive_h5_starts, postive_h5_ends)
        # calculate the negative strand as positive strand, but backwards from end
        h5_end = postive_h5_ends[-1] * 2
        negative_h5s = [(h5_end - x[1], h5_end - x[0]) for x in postive_h5s]
        return ({'plus': x[0], 'minus': x[1]} for x in zip(postive_h5s, negative_h5s))

    @staticmethod
    def _feature_not_past(feature, to):
        """whether feature is before or overlaps end by any measure"""
        if feature.is_plus_strand:
            return feature.start < to
        else:
            # although the end is exclusive and this gets one position where the feature itself has no overlap;
            # for transitions themselves we potentially want to make sure the feature.end is included
            return feature.end < to

    @staticmethod
    def _feature_ends_after(feature, to):
        """combines with _feature_not_past to define overlaps of trailing edge 'to'"""
        # overlapping features will be saved and numerified with the next write_by split as well
        if feature.is_plus_strand:
            # include as soon as end is after, despite exclusive end, to include the transition just in case
            return feature.end >= to
        else:
            return feature.start >= to

    def _find_splits(self):
        """yields splits of ~write_by size that can be safely split at"""
        tr_mask = self._transition_mask()
        for i in range(self.coord_length, self.coord_length, self.write_by):
            if i not in tr_mask:
                yield i
            else:
                for i_fudge in range(i, i + self.write_by, self.chunk_size):
                    if i_fudge not in tr_mask:
                        yield i_fudge
                        break
        yield self.coord_length

    def _transition_mask(self):
        """mark all possible splits where there is a transition, so splitting there would change the numerify results"""
        tr_mask = set()
        for feature in self.features:
            for tr in self._plus_strand_transitions(feature):
                if not tr % self.chunk_size:
                    tr_mask.add(tr)
        return tr_mask

    @staticmethod
    def _plus_strand_transitions(feature):
        if feature.is_plus_strand:
            return feature.start, feature.end
        else:
            return feature.start - 1, feature.end - 1