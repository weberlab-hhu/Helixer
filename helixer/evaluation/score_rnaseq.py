import sys
import argparse
import h5py
import numpy as np
import logging
from add_ngs_coverage import add_empty_cov_meta


def add_empty_score_datasets(h5):
    length = h5['data/X'].shape[0]
    chunk_size = h5['data/X'].shape[1]
    h5.create_group(SCORE_STR)
    shapes = [(length, chunk_size), (length, 4), (length, 4), (length, ), (length, ), (length, chunk_size, 2)]
    max_shapes = [(None, chunk_size), (None, 4), (None, 4), (None, ), (None, ), (None, chunk_size, 2)]
    for i, key in enumerate(['by_bp', 'four', 'four_centered', 'one', 'one_centered', 'norm_cov_by_bp']):
        h5.create_dataset(f'{SCORE_STR}/{key}',
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


class Scorer:
    def __init__(self, median_cov, column, sigmoid=True):
        self.scale_to = 2.6
        self.column = column
        self.median_cov = median_cov
        self.final_scale = self.scale_pos_half
        self.run_sigmoid = sigmoid

    @staticmethod
    def scale_pos_half(score):
        # starts (0.5, 1), target (0, 1)
        return (score - 0.5) * 2

    @staticmethod
    def scale_neg_half(score):
        # starts (0, 0.5), target (0, 1)
        return score * 2

    @staticmethod
    def scale_nada(score):
        return score

    def score(self, datay, coverage, spliced_coverage):
        if self.column is not None:
            mask = datay[:, self.column].astype(bool)
        else:
            # this will effectively mask padding only
            mask = np.sum(datay, axis=1).astype(bool)
        cov = coverage[mask]
        sc = spliced_coverage[mask]
        if cov.shape[0]:
            pre_score = self._pre_score(cov, sc)
            if self.run_sigmoid:
                score = self.sigmoid(pre_score)
            else:
                score = pre_score
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


class NormScorer(Scorer):
    def __init__(self, median_cov, column):
        super().__init__(median_cov, column, sigmoid=False)
        self.final_scale = self.scale_nada


class NormScoreCoverage(NormScorer):
    def _pre_score(self, cov, sc):
        x = np.log10(cov + 1)
        return x / np.log10(self.median_cov + 1)


class NormScoreSplicedCoverage(NormScorer):
    def _pre_score(self, cov, sc):
        x = np.log10(sc + 1)
        return x / np.log10(self.median_cov + 1)


def get_median_expected_coverage(h5, max_expected=1000):
    # this will fail unless max_expected is higher than the median
    # but since we're expecting a median around say 5-20... should be OK
    start, end = 0, h5['data/X'].shape[0]
    bins = list(range(max_expected)) + [np.inf]
    by = 1000
    histo = np.zeros(shape=[len(bins) - 1])
    for i in range(start, end, by):
        counts, _ = histo_expected_coverage(h5, start_i=i, end_i=min(i + by, end), bins=bins)
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


def histo_expected_coverage(h5, start_i, end_i, bins):
    masked_cov = h5[f'evaluation/{COV_STR}'][start_i:end_i][
        np.logical_or(h5['data/y'][start_i:end_i, :, 1], h5['data/y'][start_i:end_i, :, 2])
    ]
    histo = np.histogram(masked_cov, bins=bins)
    return histo


def sum_last_and_flatten(x):
    return np.sum(x, axis=-1).ravel()


def main(species, h5_data):

    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation, score, & metadata placeholders if they don't exist
    # (evaluation/coverage, "/spliced_coverage, scores/*, meta/*)

    try:
        h5[f'evaluation/{COV_STR}']
    except KeyError as e:
        print('ATTENTION: coverage must now be added via the "add_ngs_coverage.py" script, before calling this one.'
              '--dataset-prefix must match for both scripts (scripts have different defaults!)',
              file=sys.stderr)
        raise e
    try:
        h5[f'{SCORE_STR}/by_bp']
    except KeyError:
        add_empty_score_datasets(h5)

    try:
        h5[META_STR]
    except KeyError:
        pass
        # todo match to add_ngs_coverage
        #rnaseq.add_meta(h5)

    # add remaining grp not currently part of rnaseq as it's unneeded there
    try:
        h5[f'{META_STR}/median_expected_coverage']
    except KeyError:
        h5[META_STR].create_group('median_expected_coverage')

    try:
        h5[f'{META_STR}/max_normalized_cov_sc']
    except KeyError:
        h5[META_STR].create_group('max_normalized_cov_sc')

    # add median coverage in regions annotated as UTR/CDS for slightly more robust scaling
    median_coverage = get_median_expected_coverage(h5)
    h5[f'{META_STR}/median_expected_coverage'].attrs.create(name=species, data=median_coverage)

    start = 0
    end = h5['data/X'].shape[0]

    # calculate coverage score  (0 - 2)
    # setup scorers
    mec = int(h5[f'{META_STR}/median_expected_coverage'].attrs[species])
    ig_scorer = ScorerIntergenic(column=0, median_cov=mec)
    utr_scorer = ScorerExon(column=1, median_cov=mec)
    cds_scorer = ScorerExon(column=2, median_cov=mec)
    intron_scorer = ScorerIntron(column=3, median_cov=mec)
    scorers = [ig_scorer, utr_scorer, cds_scorer, intron_scorer]
    norm_cov_scorer = NormScoreCoverage(column=None, median_cov=mec)
    norm_sc_scorer = NormScoreSplicedCoverage(column=None, median_cov=mec)
    norm_scorers = [norm_cov_scorer, norm_sc_scorer]  # calculate normalized coverage more than "scoring" persay
    max_norm_cov = 0
    max_norm_sc = 0
    counts = np.zeros(shape=(end - start, 4))
    print("scoring {}-{}".format(start, end), file=sys.stderr)
    by = 500
    for i in range(start, end, by):
        if i + by > end:
            by_out = end - i
        else:
            by_out = by
        i_rel = i - start
        y = h5['data/y'][i:(i + by_out)]
        datay = y.reshape([-1, 4])
        _, chunk_size, n_cats = y.shape
        coverage = h5[f'evaluation/{COV_STR}'][i:(i + by_out)]
        spliced_coverage = h5[f'evaluation/{SC_STR}'][i:(i + by_out)]
        # coverage and spliced_coverage will initially have dimensions [n_chunks, chunk_size, n_bams]
        # but scoring assumes shape [n_basepairs], so they must be summed and flattened
        coverage, spliced_coverage = sum_last_and_flatten(coverage), sum_last_and_flatten(spliced_coverage)

        by_bp = np.full(fill_value=-1., shape=[by_out * chunk_size])
        norm_cov_by_bp = np.full(fill_value=-1., shape=[by_out * chunk_size, 2])  # 2 for [cov, sc]
        for scorer in scorers:
            raw_score, mask = scorer.score(datay=datay, coverage=coverage, spliced_coverage=spliced_coverage)
            if raw_score.size > 0:
                by_bp[mask] = raw_score
        for index_asif, norm_scorer in enumerate(norm_scorers):
            raw_score, mask = norm_scorer.score(datay=datay, coverage=coverage,
                                                spliced_coverage=spliced_coverage)
            if raw_score.size > 0:
                norm_cov_by_bp[mask, index_asif] = raw_score
        del coverage, spliced_coverage
        by_bp = by_bp.reshape([by_out, chunk_size])
        norm_cov_by_bp = norm_cov_by_bp.reshape([by_out, chunk_size, 2])
        h5[f'{SCORE_STR}/by_bp'][i:(i + by_out)] = by_bp
        h5[f'{SCORE_STR}/norm_cov_by_bp'][i:(i + by_out)] = norm_cov_by_bp
        max_norm_cov = max(max_norm_cov, np.max(norm_cov_by_bp[:, 0]))
        max_norm_sc = max(max_norm_sc, np.max(norm_cov_by_bp[:, 1]))

        current_counts = np.sum(h5['data/y'][i:(i + by_out)], axis=1)
        scores_four = np.full(fill_value=-1., shape=[by_out, 4])
        scores_one = np.full(fill_value=-1., shape=[by_out])
        for j in range(by_out):  # todo, can I replace this with some sort of apply?
            for col in range(4):
                mask = y[j, :, col].astype(bool)
                if np.any(mask):
                    scores_four[j, col] = np.mean(by_bp[j][y[j, :, col].astype(bool)])
                else:
                    scores_four[j, col] = 0.
            scores_one[j] = np.sum(current_counts[j] * scores_four[j] / np.sum(current_counts[j]))
        counts[i_rel:(i_rel + by_out)] = current_counts
        h5[f'{SCORE_STR}/four'][i:(i + by_out)] = scores_four
        h5[f'{SCORE_STR}/one'][i:(i + by_out)] = scores_one
        if not i % 2000:
            print('reached i={}'.format(i))
    print('fin, {}'.format(i + by_out), file=sys.stderr)
    # save maximums as attributes
    h5[f'{META_STR}/max_normalized_cov_sc'].attrs.create(name=species, data=(max_norm_cov, max_norm_sc))
    # normalize coverage score by species and category
    raw_four = h5[f'{SCORE_STR}/four'][start:end]
    centers = np.sum(raw_four * counts, axis=0) / np.sum(counts, axis=0)
    centered_four = raw_four - centers
    # cat_counts = np.sum(counts, axis=0)
    # weighted_four = centered_four / np.sum(cat_counts) * cat_counts ### make sure works along last axis
    centered_one = np.sum(centered_four * counts, axis=1) / np.sum(counts, axis=1)
    h5[f'{SCORE_STR}/four_centered'][start:end] = centered_four
    h5[f'{SCORE_STR}/one_centered'][start:end] = centered_one
    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5-data', help='h5 data file with /data/{X, y, species, seqids, etc...} '
                                                'AND with /evaluation/{{prefix}_coverage, {prefix}_spliced_coverage} '
                                                'to which {prefix}_scores will be added',
                        required=True)
    parser.add_argument('-s', '--species', help='species name matching that used in creation of geenuff/h5',
                        required=True)
    parser.add_argument('--dataset-prefix', help='prefix of h5 datasets to be used for scoring (default "rnaseq")',
                        default='rnaseq')
    args = parser.parse_args()
    COV_STR = f'{args.dataset_prefix}_coverage'
    SC_STR = f'{args.dataset_prefix}_spliced_coverage'
    META_STR = f'{args.dataset_prefix}_meta'
    SCORE_STR = f'{args.dataset_prefix}_scores'
    main(args.species, args.h5_data)

