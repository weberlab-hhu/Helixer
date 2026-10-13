"""convert cleaned-db schema to numeric values describing gene structure"""
import time
import logging
import geenuff.base.types
import numpy as np
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator

from geenuff.base import types

logger = logging.getLogger('HelixerLogger')

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
# AMBIGUITY_DECODE as a table indexed by a base's byte value, to encode a whole sequence at once
SEQUENCE_ENCODING = np.zeros((256, 4), np.float16)
SEQUENCE_ENCODING[[ord(base) for base in AMBIGUITY_DECODE]] = list(AMBIGUITY_DECODE.values())
KNOWN_BASES = np.zeros(256, bool)
KNOWN_BASES[[ord(base) for base in AMBIGUITY_DECODE]] = True


class Stepper(object):
    def __init__(self, end: int, by: int) -> None:
        self.at = 0
        self.end = end
        self.by = by

    def step(self) -> tuple[int, int]:
        prev = self.at
        if prev + self.by < self.end:
            new = prev + self.by
        else:
            new = self.end
        self.at = new
        return prev, new

    def step_to_end(self) -> Iterator[tuple[int, int]]:
        while self.at < self.end:
            yield self.step()


class Numerifier(ABC):
    def __init__(self, n_cols: int, coord: object, max_len: int, dtype: np.dtype = np.float32,
                 start: int = 0, end: int | None = None) -> None:
        if end is None:
            end = coord.length
        assert isinstance(n_cols, int)
        self.n_cols = n_cols
        self.coord = coord
        self.max_len = max_len
        self.dtype = dtype
        self.matrix = None
        self.start = start
        self.end = end
        self.length = self.end - self.start
        # set paired steps
        partitioner = Stepper(end=self.end - self.start, by=self.max_len)
        self.paired_steps = list(partitioner.step_to_end())
        super().__init__()

    @abstractmethod
    def coord_to_matrices(self) -> dict | tuple:
        """Method to be called from outside. Numerifies both strands."""
        pass

    def _slice_matrices(self, is_plus_strand: bool, *argv: np.ndarray) -> list[list[np.ndarray]]:
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

    def _zero_matrix(self) -> None:
        self.matrix = np.zeros((self.length, self.n_cols,), self.dtype)

def seq_numerify(seq_part: str) -> np.ndarray:
    """Encodes a sequence as one AMBIGUITY_DECODE row per base, by indexing SEQUENCE_ENCODING with its
    bytes. Raises KeyError for a character AMBIGUITY_DECODE does not know."""
    codes = np.frombuffer(seq_part.encode('ascii'), np.uint8)
    unknown = ~KNOWN_BASES[codes]
    if unknown.any():
        raise KeyError(seq_part[int(np.argmax(unknown))])
    return SEQUENCE_ENCODING[codes]


class SequenceNumerifier(Numerifier):
    def __init__(self, coord: object, max_len: int, start: int = 0, end: int | None = None) -> None:
        super().__init__(n_cols=4, coord=coord, max_len=max_len, dtype=np.float16, start=start, end=end)

    def coord_to_matrices(self) -> dict[str, list[np.ndarray]]:
        """Does not alter the error mask unlike in AnnotationNumerifier"""
        start_time = time.time()
        encoded = seq_numerify(self.coord.sequence[self.start:self.end])
        # both strands' slices are views of it, so it is made read-only: changing it in place would
        # silently change what was sliced before, and now raises an error instead
        encoded.flags.writeable = False
        data_plus = self._slice_matrices(True, encoded)[0]

        # minus strand: complementary base by flipping the columns, _slice_matrices reverses direction
        self.matrix = np.flip(encoded, axis=1)
        data_minus = self._slice_matrices(False, self.matrix)[0]

        # put everything together
        data = {'plus': data_plus, 'minus': data_minus}
        logger.info(f'Numerification of {self.start}-{self.end} of the sequence of {self.coord.seqid} '
              f'took {time.time() - start_time:.2f} secs')
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

    def __init__(self, coord: object, features: list, max_len: int, one_hot: bool = True,
                 start: int = 0, end: int | None = None) -> None:
        super().__init__(n_cols=3, coord=coord, max_len=max_len, dtype=np.int8, start=start, end=end)
        self.features = features
        self.one_hot = one_hot
        self.coord = coord
        self.error_mask = None

    def _zero_matrix(self) -> None:
        super()._zero_matrix()
        # 0 means error so this can be used directly as sample weight later on
        self.error_mask = np.ones((self.length,), np.int8)

    def _init_additional_data(self) -> None:
        self.gene_lengths = np.zeros(self.length, dtype=np.uint32)
        self.phases = np.zeros((self.length, 4), dtype=np.int8)
        self.phases[:, 0] = 1  # set no phase encoding as default

    def coord_to_matrices(self) -> tuple[dict[str, list[np.ndarray]], ...]:
        """Always numerifies both strands one after the other."""
        plus_strand = self._encode_strand(True)
        minus_strand = self._encode_strand(False)

        # put everything together
        combined_data = tuple(({'plus': plus_strand[i], 'minus': minus_strand[i]} for i in range(len(plus_strand))))
        return combined_data

    def _encode_strand(self, is_plus_strand: bool) -> list[list[np.ndarray]]:
        self._zero_matrix()
        self._init_additional_data()
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
                                        self.phases,
                                        binary_transition_matrix)
        return matrices

    def _update_matrix_and_error_mask(self, is_plus_strand: bool) -> None:
        def start_end_of_feature(feature):
            start = feature.start - self.start  # self.start used as offset for writing chunk
            end = feature.end - self.start
            if not is_plus_strand:
                start, end = end + 1, start + 1
            # save ori length and crop to size
            gene_length = end - start
            start = max(start, 0)
            end = min(end, self.matrix.shape[0])
            return start, end, gene_length

        for feature in self.features:
            # don't include features from the other strand
            if not feature.is_plus_strand == is_plus_strand:
                continue
            start, end, gene_length = start_end_of_feature(feature)
            # regular feature encoding with 3 columns
            if feature.type in AnnotationNumerifier.feature_to_col.keys():
                col = AnnotationNumerifier.feature_to_col[feature.type]
                self.matrix[start:end, col] = 1
            elif feature.type.value == types.GEENUFF_MASK:
                self.error_mask[start:end] = 0
            else:
                raise ValueError('Unknown feature type found: {}'.format(feature.type.value))
            # also fill self.gene_lengths
            # give precedence for the longer transcript if present
            if feature.type.value == types.GEENUFF_TRANSCRIPT:
                length_arr = np.full(end - start, gene_length)
                self.gene_lengths[start:end] = np.maximum(self.gene_lengths[start:end], length_arr)

        # figure out phases of cds regions after everything has, ignoring the introns
        # directly writes the one hot encoding for the 4 phase classes: -1, 0, 1, 2 (in that order)
        # expects only one transcript per gene, this will not work for all_transcripts=True in
        # HelixerExportController.export()
        cds_features = [f for f in self.features if f.type.value == types.GEENUFF_CDS]
        for cds_feature in cds_features:
            if not cds_feature.is_plus_strand == is_plus_strand:
                continue
            start, end, _ = start_end_of_feature(cds_feature)
            # get the region where there is cds but no intron
            cds_idx = np.logical_not(self.matrix[start:end, 2])
            cds_len = np.count_nonzero(cds_idx)
            # generate for full codons, truncate later if mismatched_ending_phase error
            phase_base = np.array([1, 3, 2], dtype=np.int8)
            if cds_feature.phase != 0:
                phase_base = np.roll(phase_base, cds_feature.phase)
            cds_phase = np.tile(phase_base, int(np.ceil(cds_len / 3)))
            if not is_plus_strand:
                cds_phase = cds_phase[::-1]
            cds_phase_one_hot = np.squeeze(np.eye(4, dtype=np.int8)[cds_phase])
            # need to potentially truncate from different side if phase error present (start with 0)
            overhang = len(cds_phase) - cds_len
            if overhang > 0:
                if is_plus_strand:
                    cds_phase_one_hot = cds_phase_one_hot[:-overhang]
                else:
                    cds_phase_one_hot = cds_phase_one_hot[overhang:]
            self.phases[start:end][cds_idx] = cds_phase_one_hot

    def _encode_onehot4(self) -> np.ndarray:
        # Class order: Intergenic, UTR, CDS, Intron. Each base gets the index of its one class, from
        # the binary [transcript, CDS, intron] encoding: outside a transcript intergenic (0), else an
        # intron (3) over CDS (2) over UTR (1), so exactly one class is set per base by construction
        col_0, col_1, col_2 = (self.matrix[:, i].astype(np.int8) for i in range(3))
        classes = col_0 * (1 + col_1 * (1 - col_2) + 2 * col_2)
        return (classes[:, None] == np.arange(4, dtype=np.int8)).astype(np.int8)

    def _encode_transitions(self) -> np.ndarray:
        # self.matrix has base-wise binary encoding of [geenuff_transcript, geenuff_cds, geenuff_intron];
        # a feature starts before a position where the value rises (-1 difference to the next base)
        # and ends before one where it falls (1)
        diffs = self.matrix[:-1] - self.matrix[1:]
        stack = np.zeros((self.matrix.shape[0], 6), np.int8)
        stack[:-1, :3] = diffs < 0
        stack[:-1, 3:] = diffs > 0
        # mark both the last bp of the preceding class and the first bp of the new class, so the
        # transition weight lies on both sides; the last row of stack is always 0, so nothing wraps
        binary_transitions = stack.copy()
        binary_transitions[1:] += stack[:-1]
        return binary_transitions  # 6 columns, one for each switch (+TR, +CDS, +In, -TR, -CDS, -In)


class MatAndInfo:
    """organizes data and meta info for post-processing and saving a matrix"""
    def __init__(self, key: str, matrix: np.ndarray, dtype: str) -> None:
        self.key = key
        self.matrix = matrix
        self.dtype = dtype

    def __repr__(self) -> str:
        return "key: {}, matrix shape: {}, matrix dtype {}: target dtype {}".format(self.key, self.matrix.shape,
                                                                                    self.matrix.dtype, self.dtype)


class CoordNumerifier(object):
    """Combines the different Numerifiers which need to operate on the same Coordinate
    to ensure consistent parameters. Selects all Features of the given Coordinate.
    """

    @staticmethod
    def pad(d: list[np.ndarray], chunk_size: int) -> np.ndarray:
        n_seqs = len(d)
        # insert all the sequences so that 0-padding is added if needed
        shape = tuple([n_seqs, chunk_size] + list(d[0].shape[1:]))
        padded_d = np.zeros(shape, dtype=d[0].dtype)
        for j in range(n_seqs):
            padded_d[j, :len(d[j])] = d[j]
        return padded_d

    @staticmethod
    def start_ends(numerifier: Numerifier, strand: str) -> np.ndarray:
        assert isinstance(numerifier, Numerifier)
        if strand == 'plus':
            start_ends = numerifier.paired_steps
        else:
            # flip the start ends back for - strand
            start_ends = [(x[1], x[0]) for x in numerifier.paired_steps[::-1]]
        start_ends = np.array(start_ends, dtype=np.int64)
        return start_ends

    @staticmethod
    def seq_matinfos(coord: object, genome: str, start_ends: np.ndarray,
                     length: int) -> list[MatAndInfo]:
        res = [MatAndInfo('species', np.array([genome.encode('ASCII')] * length), 'S25'),
               MatAndInfo('seqids', np.array([coord.seqid.encode('ASCII')] * length), 'S50'),
               MatAndInfo('start_ends', start_ends, 'int64')]
        return res

    @staticmethod
    def numerify_only_fasta(coord: object, max_len: int, genome: str, write_by: int,
                            one_hot: bool = True) -> Iterator[tuple[tuple[MatAndInfo, ...], tuple[int, int]]]:
        """export the FASTA sequence only"""
        # passing empty features causes SplitFinder to consider noting more than splitting
        # to max length of write_by and end of sequence handling.
        split_finder = SplitFinder(features=(), write_by=write_by, coord_length=coord.length,
                                   chunk_size=max_len)
        for _, bp_coord, h5_coord in split_finder.feature_n_coord_gen():
            start, end = bp_coord
            seq_numerifier = SequenceNumerifier(coord=coord, max_len=max_len, start=start, end=end)
            xb = seq_numerifier.coord_to_matrices()
            for strand in ['plus', 'minus']:
                x = CoordNumerifier.pad(xb[strand], max_len)
                start_ends = CoordNumerifier.start_ends(seq_numerifier, strand)
                start_ends += start
                out = [MatAndInfo('X', x, 'float16')]
                out.extend(CoordNumerifier.seq_matinfos(coord, genome, start_ends, len(x)))
                yield tuple(out), h5_coord[strand]

    @staticmethod
    def numerify(coord: object, coord_features: list, max_len: int, write_by: int,
                 one_hot: bool = True,
                 mode: tuple[str, ...] = ('X', 'y', 'anno_meta', 'transitions')
                 ) -> Iterator[tuple[tuple[MatAndInfo, ...], tuple[int, int]]]:
        assert isinstance(max_len, int) and max_len > 0, 'what is {} of type {}'.format(max_len, type(max_len))
        coord_features = sorted(coord_features, key=lambda f: min(f.start, f.end))  # sort by ~ +strand start
        split_finder = SplitFinder(features=coord_features, write_by=write_by, coord_length=coord.length,
                                   chunk_size=max_len)
        for f_set, bp_coord, h5_coord in split_finder.feature_n_coord_gen():
            for strand_res in CoordNumerifier._numerify_super_write_chunk(f_set, bp_coord, h5_coord, coord, max_len,
                                                                          one_hot, coord_features, mode):
                yield strand_res

    @staticmethod
    def _numerify_super_write_chunk(f_set: list, bp_coord: tuple[int, int],
                                    h5_coord: dict[str, tuple[int, int]], coord: object,
                                    max_len: int, one_hot: bool, coord_features: list,
                                    mode: tuple[str, ...]) -> Iterator[tuple[tuple[MatAndInfo, ...], tuple[int, int]]]:
        export_x = 'X' in mode
        start, end = bp_coord

        anno_numerifier = AnnotationNumerifier(coord=coord, features=f_set, max_len=max_len,
                                               one_hot=one_hot, start=start, end=end)
        seq_numerifier = SequenceNumerifier(coord=coord, max_len=max_len, start=start, end=end)

        # everything with _b below is for "both strands" and is {"plus": +_np_array, "minus": -_np_array }
        # todo, make mode more elegant / extensible
        if export_x:
            xb = seq_numerifier.coord_to_matrices()
        yb, sample_weightsb, gene_lengthsb, phasessb, transitionsb = anno_numerifier.coord_to_matrices()
        for strand in ['plus', 'minus']:
            if export_x:
                x = CoordNumerifier.pad(xb[strand], max_len)
            y, sample_weights, gene_lengths, phases, transitions = \
                (CoordNumerifier.pad(x, max_len) for x in [yb[strand],
                                                           sample_weightsb[strand],
                                                           gene_lengthsb[strand],
                                                           phasessb[strand],
                                                           transitionsb[strand]])
            start_ends = CoordNumerifier.start_ends(anno_numerifier, strand)
            start_ends += start

            # mark examples from featureless coordinate / assume there is no trustworthy annotation
            if not coord_features:
                logger.warning('Sequence {} has no annotations'.format(coord.seqid))
                is_annotated = [0] * len(y)
            else:
                is_annotated = [1] * len(y)
            is_annotated = np.array(is_annotated, dtype=bool)

            # additional derived matrices
            err_samples = np.any(sample_weights, axis=1)
            # just one entry per chunk
            if one_hot:  # question: when isn't it one hot encoded?
                is_first_class_position_intergenic = 1
                fully_intergenic_samples = np.all(y[:, :, 0] == is_first_class_position_intergenic, axis=1)
            else:  # will never be executed with the current setup
                fully_intergenic_samples = np.all(y[:, :, 0] == 1, axis=1)  # todo: might be wrong, but is never called

            # do not output the input_masks as it is not used for anything
            out = [MatAndInfo('y', y, 'int8'),  # y should always be first (bc currently we always want it)
                   MatAndInfo('sample_weights', sample_weights, 'int8'),
                   MatAndInfo('gene_lengths', gene_lengths, 'uint32'),
                   MatAndInfo('phases', phases, 'int8'),
                   MatAndInfo('transitions', transitions, 'int8'),
                   MatAndInfo('err_samples', err_samples, 'bool'),
                   MatAndInfo('fully_intergenic_samples', fully_intergenic_samples,  'bool'),
                   MatAndInfo('is_annotated', is_annotated, 'bool')]
            out.extend(CoordNumerifier.seq_matinfos(coord, coord.genome.species, start_ends, len(y)))
            if export_x:
                out.append(MatAndInfo('X', x, 'float16'))
            out = tuple(out)
            yield out, h5_coord[strand]


# todo, consider moving to separate splitting file or exporter...?
class SplitFinder:
    def __init__(self, features: list | tuple, write_by: int, coord_length: int, chunk_size: int) -> None:
        if write_by % chunk_size:
            old_write_by = write_by
            write_by = chunk_size * (write_by // chunk_size)
            logger.info(f'parameter "write_by" changed from {old_write_by} to {write_by} to be a multiple'
                        f'of "subsequence length" {chunk_size}')
        self.features = features
        self.write_by = write_by  # target writing this many bp to the h5 file at once
        self.coord_length = coord_length
        self.chunk_size = chunk_size
        self.splits = tuple(self._find_splits())
        logger.info(f'{len(self.splits)} expected num of chunks to write in {self.write_by} bases to hdf5')
        self.relative_h5_coords = tuple(self._get_rel_h5_coords_for_splits())

    @property
    def coords(self) -> tuple[tuple[int, int], ...]:
        starts = [0] + list(self.splits)[:-1]
        return tuple(zip(starts, self.splits))

    def feature_n_coord_gen(self) -> Iterator[tuple]:
        """generator for feature subset, bp coordinates, and h5 coordinates (for +/-)"""
        return zip(self.split_features(), self.coords, self.relative_h5_coords)

    def split_features(self) -> Iterator[list]:
        """get all features from start of list that aren't passed _to_"""
        i = 0
        in_split = []
        for end in self.splits:
            feature = self._ith_feature_or_none(i)
            while self._feature_not_past(feature, end):
                in_split.append(feature)
                i += 1
                feature = self._ith_feature_or_none(i)
            yield in_split
            # every feature still reaching past this split is carried into the next one, the
            # ones carried here included, so that a feature spanning any number of splits is
            # given to each of them and gets written over its whole length
            in_split = [f for f in in_split if self._feature_ends_after(f, end)]

    def _ith_feature_or_none(self, i: int) -> object | None:
        try:
            return self.features[i]
        except IndexError:
            return None

    def _get_rel_h5_coords_for_splits(self) -> Iterator[dict[str, tuple[int, int]]]:
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

        postive_h5s = list(zip(postive_h5_starts, postive_h5_ends))
        # calculate the negative strand as positive strand, but backwards from end
        h5_end = postive_h5_ends[-1] * 2
        negative_h5s = [(h5_end - x[1], h5_end - x[0]) for x in postive_h5s]
        return ({'plus': x[0], 'minus': x[1]} for x in zip(postive_h5s, negative_h5s))

    @staticmethod
    def _feature_not_past(feature: object | None, to: int) -> bool:
        """whether feature is before or overlaps end by any measure"""
        if feature is None:
            return False  # force break of while loop when out of features
        if feature.is_plus_strand:
            return feature.start < to
        else:
            # although the end is exclusive and this gets one position where the feature itself has no overlap;
            # for transitions themselves we potentially want to make sure the feature.end is included
            return feature.end < to

    @staticmethod
    def _feature_ends_after(feature: object, to: int) -> bool:
        """combines with _feature_not_past to define overlaps of trailing edge 'to'"""
        # overlapping features will be saved and numerified with the next write_by split as well
        if feature.is_plus_strand:
            # include as soon as end is after, despite exclusive end, to include the transition just in case
            return feature.end >= to
        else:
            return feature.start >= to

    def _find_splits(self) -> Iterator[int]:
        """yields splits of ~write_by size that can be safely split at"""
        tr_mask = self._transition_and_split_cds_mask()
        for i in range(self.write_by, self.coord_length, self.write_by):
            if i not in tr_mask:
                yield i
            else:
                for i_fudge in range(i, min(i + self.write_by, self.coord_length), self.chunk_size):
                    if i_fudge not in tr_mask:
                        yield i_fudge
                        break
        yield self.coord_length

    def _transition_and_split_cds_mask(self) -> set[int]:
        """mark all possible splits where there is a transition, so splitting there would change the numerify results"""
        tr_mask = set()
        for feature in self.features:
            # avoid splitting exactly at transitions, as transitions are detected by
            # state change (of binary encoding) and you can't detect state-change
            # if you split at it
            f_start, f_end = self._plus_strand_transitions(feature)
            for tr in (f_start, f_end):
                if not tr % self.chunk_size:
                    tr_mask.add(tr)
            # also avoid splitting in the middle of a CDS, this is necessary to
            # simplify the encoding of phase / avoid painful edge cases
            # this should add all chunk ends with a CDS to the mask
            cs = self.chunk_size
            if feature.type.value == geenuff.base.types.GEENUFF_CDS:
                round_start = math.ceil(f_start / cs)
                round_end = math.ceil(f_end / cs)
                for chunk_end in range(round_start * cs, round_end * cs, cs):
                    tr_mask.add(chunk_end)

        return tr_mask

    @staticmethod
    def _plus_strand_transitions(feature: object) -> tuple[int, int]:
        if feature.is_plus_strand:
            return feature.start, feature.end
        else:
            return feature.end - 1, feature.start - 1
