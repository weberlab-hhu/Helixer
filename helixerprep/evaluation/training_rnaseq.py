import sys
import argparse
import HTSeq
import h5py
import numpy as np
from helixerprep.evaluation import rnaseq


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
    def __init__(self, column, coverage_helps, spliced_coverage_helps):
        self.column = column
        if coverage_helps:
            self.coverage_score_component = self.coverage_helps
        else:
            self.coverage_score_component = self.coverage_hurts
        if spliced_coverage_helps:
            self.spliced_coverage_score_component = self.coverage_helps
        else:
            self.spliced_coverage_score_component = self.coverage_hurts

    def score(self, h5, i):
        mask = h5['data/y'][i][:, self.column] == 1
        cov = h5['evaluation/coverage'][i][mask]
        sc = h5['evaluation/spliced_coverage'][i][mask]
        if cov.shape[0]:
            cov_score = self.coverage_score_component(cov)
            sc_score = self.spliced_coverage_score_component(sc)
        else:
            cov_score, sc_score = 0, 0

        return cov_score + sc_score

    def coverage_hurts(self, array):
        # returns 1 when array has only 0s, approaches 0 when array is high
        return 1 / (np.mean(array) + 1)

    def coverage_helps(self, array):
        # returns 1 when array has only 0s, approaches 0 when array is high
        return 1 - self.coverage_hurts(array)


def main(species, bam, h5_data, d_utp):
    # setup scorers
    ig_scorer = Scorer(column=0, coverage_helps=False, spliced_coverage_helps=False)
    utr_scorer = Scorer(column=1, coverage_helps=True, spliced_coverage_helps=False)
    cds_scorer = Scorer(column=2, coverage_helps=True, spliced_coverage_helps=False)
    intron_scorer = Scorer(column=3, coverage_helps=False, spliced_coverage_helps=True)
    scorers = [ig_scorer, utr_scorer, cds_scorer, intron_scorer]

    # open h5 and bam
    h5 = h5py.File(h5_data, 'r+')
    htseqbam = HTSeq.BAM_Reader(bam)
    # create evaluation & score placeholders if they don't exist (evaluation/coverage, "/spliced_coverage, scores/*)
    try:
        h5['evaluation/coverage']
    except KeyError:
        add_empty_eval_datasets(h5)

    try:
        h5['scores/four']
    except KeyError:
        add_empty_score_datasets(h5)

    # identify regions in h5 corresponding to species
    species_start, species_end = species_range(h5, species)

    # insert coverage into said regions
    coords = rnaseq.gen_coords(h5, species_start, species_end)
    print('start, end', species_start, species_end)
    pad_to = h5['evaluation/coverage'].shape[1]

    for coord in coords:
        print(coord)
        rnaseq.coverage_from_coord_to_h5(coord, h5, bam=htseqbam, d_utp=d_utp, pad_to=pad_to)

    # calculate coverage score  (0 - 2)
    # todo, this is really naive and probably terribly slow... fix
    counts = np.zeros(shape=(species_end - species_start, 4))
    for i in range(species_start, species_end):
        i_rel = i - species_start
        for scorer in scorers:
            score = scorer.score(h5, i)
            print(score, i, scorer.column)
            h5['scores/four'][i][scorer.column] = score
        current_counts = np.sum(h5['data/y'][i], axis=0)
        counts[i_rel] = current_counts
        # weighted average
        h5['scores/one'][i] = np.sum(current_counts * h5['scores/four'][i]) / np.sum(current_counts)

    # normalize coverage score by species and category
    raw_four = h5['scores/four'][species_start:species_end]
    centers = np.sum(raw_four * counts, axis=0) / np.sum(counts, axis=0)
    centered_four = raw_four - centers
    centered_one = np.sum((centered_four * counts).T / np.sum(counts, axis=1))
    h5['scores/four_centered'][species_start:species_end] = centered_four
    h5['scores/one_centered'][species_start:species_end] = centered_one
    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', help="species name, matching geenuff db and h5 files", required=True)
    parser.add_argument('-b', '--bam', help='sorted (and indexed) bam file', required=True)
    parser.add_argument('-d', '--h5_data', help='h5 data file (with /data/{X, y, species, seqids, etc...}) '
                                                'to which evaluation coverage will be ADDED!',
                        required=True)
    parser.add_argument('-x', '--not_dUTP', help='bam does not contain stranded (from typical dUTP protocol) reads',
                        action='store_true')
    args = parser.parse_args()
    main(args.species,
         args.bam,
         args.h5_data,
         not args.not_dUTP)

