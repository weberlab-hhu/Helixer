import sys
import argparse
import HTSeq
import h5py
import numpy as np
from helixerprep.evaluation import rnaseq
import copy


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
                          data=np.full(fill_value=-1, shape=(length, chunk_len)))


def add_empty_score_datasets(h5):
    length = h5['data/X'].shape[0]
    h5.create_group('scores')
    shapes = [(length, 4), (length, 4), (length, ), (length, )]
    max_shapes = [(None, 4), (None, 4), (None, ), (None, )]
    for i, key in enumerate(['four', 'four_centered', 'one', 'one_centered']):
        h5.create_dataset('scores/' + key,
                          shape=shapes[i],
                          maxshape=max_shapes[i],
                          dtype="float",
                          compression="lzf",
                          data=np.full(fill_value=-1., shape=shapes[i]))


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
    def __init__(self, column, coverage_helps, spliced_coverage_helps, median_cov, scale_to=4):
        # scale to default gives score of 0.98 @ median coverage w/o penalty
        # currently using the same for cov/sc, but idk if that's really a good idea
        self.column = column
        self.median_cov = median_cov
        self.scale_to = scale_to

        if coverage_helps:
            self.coverage_score_component = 1
        else:
            self.coverage_score_component = -1
        if spliced_coverage_helps:
            self.spliced_coverage_score_component = 1
        else:
            self.spliced_coverage_score_component = -1

        # if both help or both hurt, scale to still get numbers between 0 and 1
        if self.coverage_score_component + self.spliced_coverage_score_component == -2:
            self.final_scale = self.scale_neg_half
        elif self.coverage_score_component + self.spliced_coverage_score_component == 2:
            self.final_scale = self.scale_neg_half
        else:
            self.final_scale = self.do_nothing

    @staticmethod
    def scale_pos_half(score):
        # starts (0.5, 1), target (0, 1)
        return (score - 0.5) * 2

    @staticmethod
    def scale_neg_half(score):
        # starts (0, 0.5), target (0, 1)
        return score * 2

    @staticmethod
    def do_nothing(score):
        return score

    def score(self, datay, coverage, spliced_coverage):
        mask = datay[:, self.column] == 1
        cov = coverage[mask]
        sc = spliced_coverage[mask]
        if cov.shape[0]:
            x = cov * self.coverage_score_component + sc * self.spliced_coverage_score_component
            score = self.sigmoid(x * self.scale_to / self.median_cov)
            score = self.final_scale(score)
            score = np.mean(score)  # remove this if basewise scores are desired later
        else:
            score = 0

        return score

    @staticmethod
    def sigmoid(x):
        # avoid overflow
        x[x > 30] = 30
        x[x < -30] = -30
        # sigmoid
        return 1 / (1 + np.exp(-x))


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
        h5['scores/four']
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
    print('start, end', species_start, species_end)
    cov_counts = copy.deepcopy(rnaseq.COVERAGE_COUNTS)  # tracks number reads, bp coverage, bp spliced coverage
    if bam is not None:
        pad_to = h5['evaluation/coverage'].shape[1]

        # open bam (read alignment file)
        htseqbam = HTSeq.BAM_Reader(bam)
        for coord in coords:
            print(coord)
            coord_cov_counts = rnaseq.coverage_from_coord_to_h5(coord, h5, bam=htseqbam, d_utp=d_utp, pad_to=pad_to)
            for key in coord_cov_counts:
                cov_counts[key] += coord_cov_counts[key]

        # add bam related metadata
        h5['meta/bamfile'].attrs.create(name=species, data=bam.encode('utf-8'))
        for key in cov_counts:
            h5['meta/total_' + key].attrs.create(name=species, data=cov_counts[key])

        # add median coverage in regions annotated as UTR/CDS for slightly more robust scaling
        masked_cov = h5['evaluation/coverage'][:][np.logical_or(
            h5['data/y'][:, :, 1], h5['data/y'][:, :, 2])]
        median_coverage = np.quantile(masked_cov, 0.5)
        h5['meta/median_expected_coverage'].attrs.create(name=species, data=median_coverage)

    if not dont_score:
        # calculate coverage score  (0 - 2)
        # setup scorers
        mec = int(h5['meta/median_expected_coverage'].attrs[species])
        ig_scorer = Scorer(column=0, coverage_helps=False, spliced_coverage_helps=False, median_cov=mec)
        utr_scorer = Scorer(column=1, coverage_helps=True, spliced_coverage_helps=False, median_cov=mec)
        cds_scorer = Scorer(column=2, coverage_helps=True, spliced_coverage_helps=False, median_cov=mec)
        intron_scorer = Scorer(column=3, coverage_helps=False, spliced_coverage_helps=True, median_cov=mec)
        scorers = [ig_scorer, utr_scorer, cds_scorer, intron_scorer]

        # setup normalization factors (this would probably be better if it was more like the VST from DESeq...
        # or at least based on the median coverage, but mapped library size is easier for now
        cov_scale = 1 #10**9 / h5['meta/total_coverage'].attrs[species]
        print(cov_counts, ' so scaling cov/sc by ', cov_scale)
        # todo, this is really naive and probably terribly slow... fix
        counts = np.zeros(shape=(species_end - species_start, 4))
        print("scoring {}-{}".format(species_start, species_end))
        for i in range(species_start, species_end):
            i_rel = i - species_start
            datay = h5['data/y'][i]
            coverage = h5['evaluation/coverage'][i] * cov_scale
            spliced_coverage = h5['evaluation/spliced_coverage'][i] * cov_scale
            for scorer in scorers:
                score = scorer.score(datay=datay, coverage=coverage, spliced_coverage=spliced_coverage)
                h5['scores/four'][i, scorer.column] = score
            current_counts = np.sum(h5['data/y'][i], axis=0)
            counts[i_rel] = current_counts
            # weighted average
            h5['scores/one'][i] = np.sum(current_counts * h5['scores/four'][i]) / np.sum(current_counts)
            if not i % 200:
                print('reached i={}'.format(i))
        print('fin, i={}'.format(i))
        print(counts.shape, 'is counts.shape')
        print(np.sum(counts, axis=0), 'cat_counts to be')
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

