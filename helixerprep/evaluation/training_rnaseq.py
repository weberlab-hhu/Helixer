import sys
import argparse
import HTSeq
import h5py
import numpy as np
import random
import os
import shutil
from helixerprep.evaluation import rnaseq
import copy
import logging


def add_empty_eval_datasets(h5):
    length = h5['data/X'].shape[0]
    chunk_len = h5['data/X'].shape[1]

    h5.create_group('evaluation')
    for key in rnaseq.COVERAGE_SETS:
        h5.create_dataset('evaluation/' + key,
                          shape=(length, chunk_len),
                          maxshape=(None, chunk_len),
                          dtype="int",
                          compression="lzf",
                          fillvalue=-1)


def add_empty_score_datasets(h5):
    length = h5['data/X'].shape[0]
    chunk_size = h5['data/X'].shape[1]
    h5.create_group('scores')
    shapes = [(length, chunk_size), (length, 4), (length, 4), (length, ), (length, )]
    max_shapes = [(None, chunk_size), (None, 4), (None, 4), (None, ), (None, )]
    for i, key in enumerate(['by_bp', 'four', 'four_centered', 'one', 'one_centered']):
        h5.create_dataset('scores/' + key,
                          shape=shapes[i],
                          maxshape=max_shapes[i],
                          dtype="float",
                          compression="lzf",
                          fillvalue=-1)


def get_bool_stretches(alist):
    targ = alist[0]
    while alist:
        try:
            i = alist.index(not targ)
        except ValueError:
            i = len(alist)
            yield targ, i
            return
        yield targ, i
        targ = not targ
        alist = alist[i:]


def species_range(h5, species):
    mask = np.array(h5['/data/species'][:] == species.encode('utf-8'))
    stretches = list(get_bool_stretches(mask.tolist()))  # [(False, count), (True, Count), (False, Count)]
    print(stretches)
    i_of_true = [i for i in range(len(stretches)) if stretches[i][0]]
    assert len(i_of_true) == 1, "not contiguous or missing species ({}) in h5???".format(species)
    iot = i_of_true[0]
    if iot == 0:
        return 0, stretches[0][1]
    elif iot == 1:
        start = stretches[0][1]
        length = stretches[1][1]
        return start, start + length
    else:
        raise ValueError("should never be reached, maybe h5 sorting something or failed bool comparisons (None or so?)")


class Scorer:
    def __init__(self, median_cov, column):
        self.scale_to = 2.6
        self.column = column
        self.median_cov = median_cov
        self.final_scale = self.scale_pos_half

    @staticmethod
    def scale_pos_half(score):
        # starts (0.5, 1), target (0, 1)
        return (score - 0.5) * 2

    @staticmethod
    def scale_neg_half(score):
        # starts (0, 0.5), target (0, 1)
        return score * 2

    def score(self, datay, coverage, spliced_coverage):
        mask = datay[:, self.column] == 1
        cov = coverage[mask]
        sc = spliced_coverage[mask]
        if cov.shape[0]:
            pre_score = self._pre_score(cov, sc)
            score = self.sigmoid(pre_score)
            score = self.final_scale(score)
        else:
            score = np.array([])

        return score, mask  # todo, maybe full score can be used _with_ and mask not returned??

    def _pre_score(self, cov, sc):
        # calculates a proxy score for how well the reference is supported by RNAseq data
        # exact equation depends on annotation (subclass)
        raise NotImplementedError

    @staticmethod
    def sigmoid(x):
        # avoid overflow
        x[x > 30] = 30
        x[x < -30] = -30
        # sigmoid
        return 1 / (1 + np.exp(-x))


class ScorerIntergenic(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column=column)
        self.final_scale = self.scale_neg_half

    def _pre_score(self, cov, sc):
        x = np.log(cov + sc + 1)
        # normalize (different .bams have different coverage)
        x = x * self.scale_to / np.log(self.median_cov + 1) * -1
        return x


class ScorerExon(Scorer):
    def _pre_score(self, cov, sc):
        effective_cov = cov - sc
        # trim, so scores below 0 are
        effective_cov[effective_cov < 0] = 0
        x = np.log(effective_cov + 1)
        return x * self.scale_to / np.log(self.median_cov + 1)


class ScorerIntron(Scorer):
    def _pre_score(self, cov, sc):
        effective_cov = sc - cov
        effective_cov[effective_cov < 0] = 0
        x = np.log(effective_cov + 1)
        return x * self.scale_to / np.log(self.median_cov + 1)


def get_median_expected_coverage(h5, max_expected=1000):
    # this will fail unless max_expected is higher than the median
    # but since we're expecting a median around say 5-20... should be OK
    bins = list(range(max_expected)) + [np.inf]
    by = 1000
    histo = np.zeros(shape=[len(bins) - 1])
    for i in range(1, h5['data/y'].shape[0], by):
        counts, _ = histo_expected_coverage(h5, i, by, bins)
        histo += counts

    half_total_bp = np.sum(histo) / 2
    cumulative = 0
    for bi in range(len(histo)):
        cumulative += histo[bi]
        if cumulative == half_total_bp:
            # if we're exactly half way, _and_ between bins, then the median is the average of the middle two
            median = bi + 0.5
            break
        elif cumulative > half_total_bp:
            # median was in this bin
            median = bi
            break
    if max_expected == median + 1:
        logging.warning('median is in last bin [{}, {}] and therefore undefined'.format(bins[-2], bins[-1]))
    if median == 0:
        logging.warning('median is 0, ignoring reality and setting to 1 to avoid dividing by 0 later')
        median = 1
    return median


def histo_expected_coverage(h5, i, by, bins):
    masked_cov = h5['evaluation/coverage'][i:(i + by)][
        np.logical_or(h5['data/y'][i:(i + by), :, 1], h5['data/y'][i:(i + by), :, 2])
    ]
    histo = np.histogram(masked_cov, bins=bins)
    return histo


def main(species, bam, h5_data, d_utp, dont_score):

    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation, score, & metadata placeholders if they don't exist
    # (evaluation/coverage, "/spliced_coverage, scores/*, meta/*)

    try:
        h5['evaluation/coverage']
    except KeyError:
        add_empty_eval_datasets(h5)

    try:
        h5['scores/by_bp']
    except KeyError:
        add_empty_score_datasets(h5)

    try:
        h5['meta']
    except KeyError:
        rnaseq.add_meta(h5)

    # add remaining grp not currently part of rnaseq as it's unneeded there
    try:
        h5['meta/median_expected_coverage']
    except KeyError:
        h5['meta'].create_group('median_expected_coverage')

    # identify regions in h5 corresponding to species
    species_start, species_end = species_range(h5, species)
    # save range to h5 meta
    h5['meta/start_end_i'].attrs.create(name=species, data=(species_start, species_end))

    # insert coverage into said regions
    coords = rnaseq.gen_coords(h5, species_start, species_end)
    print('start, end', species_start, species_end, file=sys.stderr)
    cov_counts = copy.deepcopy(rnaseq.COVERAGE_COUNTS)  # tracks number reads, bp coverage, bp spliced coverage
    if bam is not None:
        chunk_size = h5['evaluation/coverage'].shape[1]
        # setup dir for memmap array (AKA, don't try and store the whole chromosome in RAM
        memmap_dirs = ["memmap_dir_{}".format(random.getrandbits(128)),
                       "memmap_dir_{}".format(random.getrandbits(128))]
        for d in memmap_dirs:
            if not os.path.exists(d):
                os.mkdir(d)
        # open bam (read alignment file)
        htseqbam = HTSeq.BAM_Reader(bam)
        for coord in coords:
            print(coord, file=sys.stderr)
            coord_cov_counts = rnaseq.coverage_from_coord_to_h5(
                coord, h5, bam=htseqbam, d_utp=d_utp,
                chunk_size=chunk_size, memmap_dirs=memmap_dirs)
            for key in coord_cov_counts:
                cov_counts[key] += coord_cov_counts[key]

        for d in memmap_dirs:
            shutil.rmtree(d)  # rm -r

        # add bam related metadata
        h5['meta/bamfile'].attrs.create(name=species, data=bam.encode('utf-8'))
        for key in cov_counts:
            h5['meta/total_' + key].attrs.create(name=species, data=cov_counts[key])

        # add median coverage in regions annotated as UTR/CDS for slightly more robust scaling
        median_coverage = get_median_expected_coverage(h5)
        h5['meta/median_expected_coverage'].attrs.create(name=species, data=median_coverage)

    if not dont_score:
        # calculate coverage score  (0 - 2)
        # setup scorers
        mec = int(h5['meta/median_expected_coverage'].attrs[species])
        ig_scorer = ScorerIntergenic(column=0, median_cov=mec)
        utr_scorer = ScorerExon(column=1, median_cov=mec)
        cds_scorer = ScorerExon(column=2, median_cov=mec)
        intron_scorer = ScorerIntron(column=3, median_cov=mec)
        scorers = [ig_scorer, utr_scorer, cds_scorer, intron_scorer]

        # todo, this is really naive and probably terribly slow... fix
        counts = np.zeros(shape=(species_end - species_start, 4))
        print("scoring {}-{}".format(species_start, species_end), file=sys.stderr)
        by = 500
        for i in range(species_start, species_end, by):
            if i + by > species_end:
                by_out = species_end - i
            else:
                by_out = by
            i_rel = i - species_start
            y = h5['data/y'][i:(i + by_out)]
            datay = y.reshape([-1, 4])
            _, chunk_size, n_cats = y.shape
            coverage = h5['evaluation/coverage'][i:(i + by_out)].ravel()
            spliced_coverage = h5['evaluation/spliced_coverage'][i:(i + by_out)].ravel()
            by_bp = np.full(fill_value=-1., shape=[by_out * chunk_size])
            for scorer in scorers:
                raw_score, mask = scorer.score(datay=datay, coverage=coverage, spliced_coverage=spliced_coverage)
                if raw_score.size > 0:
                    by_bp[mask] = raw_score
            del coverage, spliced_coverage
            by_bp = by_bp.reshape([by_out, chunk_size])
            h5['scores/by_bp'][i:(i + by)] = by_bp

            current_counts = np.sum(h5['data/y'][i:(i + by)], axis=1)
            scores_four = np.full(fill_value=-1., shape=[by_out, 4])
            scores_one = np.full(fill_value=-1., shape=[by_out])
            for j in range(by_out):  # todo, can I replace this with some sort of apply?
                for col in range(4):
                    mask = y[j, :, col].astype(np.bool)
                    if np.any(mask):
                        scores_four[j, col] = np.mean(by_bp[j][y[j, :, col].astype(np.bool)])
                    else:
                        scores_four[j, col] = 0.
                scores_one[j] = np.sum(current_counts[j] * scores_four[j] / np.sum(current_counts[j]))
            counts[i_rel:(i_rel + by)] = current_counts
            h5['scores/four'][i:(i + by)] = scores_four
            h5['scores/one'][i:(i + by)] = scores_one
            if not i % 2000:
                print('reached i={}'.format(i))
        print('fin, i={}'.format(i), file=sys.stderr)
        # normalize coverage score by species and category
        raw_four = h5['scores/four'][species_start:species_end]
        centers = np.sum(raw_four * counts, axis=0) / np.sum(counts, axis=0)
        centered_four = raw_four - centers
        # cat_counts = np.sum(counts, axis=0)
        # weighted_four = centered_four / np.sum(cat_counts) * cat_counts ### make sure works along last axis
        centered_one = np.sum(centered_four * counts, axis=1) / np.sum(counts, axis=1)
        h5['scores/four_centered'][species_start:species_end] = centered_four
        h5['scores/one_centered'][species_start:species_end] = centered_one
    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', help="species name, matching geenuff db and h5 files", required=True)
    parser.add_argument('-d', '--h5_data', help='h5 data file (with /data/{X, y, species, seqids, etc...}) '
                                                'to which evaluation coverage will be ADDED!',
                        required=True)
    parser.add_argument('-b', '--bam', help='sorted (and indexed) bam file. Omit to only score existing coverage.',
                        default=None)
    parser.add_argument('-x', '--not_dUTP', help='bam does not contain stranded (from typical dUTP protocol) reads',
                        action='store_true')
    parser.add_argument('-r', '--skip_scoring', action="store_true",
                        help="set this to add coverage to the bam file, but not 'score' it (raw cov. only)")
    args = parser.parse_args()
    main(args.species,
         args.bam,
         args.h5_data,
         not args.not_dUTP,
         args.skip_scoring)
