import sys
import argparse
from . import rnaseq
import HTSeq
import h5py
import numpy as np
from helixerprep.core.helpers import mk_keys


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
    i_of_true = [i for i in range(len(stretches)) if stretches[i][0]]
    assert len(i_of_true) == 1, "not contiguous or missing species ({}) in h5???".format(species)
    iot = i_of_true[0]
    if iot == 0:
        return 0, stretches[0][1]
    elif iot == 1:
        return stretches[0][1], stretches[1][1]
    else:
        raise ValueError("should never be reached, maybe h5 sorting something or failed bool comparisons (None or so?)")


def main(species, bam, h5_data, dUTP):
    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation placeholders if they don't exist (coverage, spliced_coverage, raw_score, scaled_score)
    try:
        h5['evaluation/coverage']
    except KeyError:
        add_empty_eval_datasets(h5)

    # identify regions in h5 corresponding to species
    species_start, species_end = species_range(h5, species)

    # insert coverage into said regions
    coords = rnaseq.gen_coords(h5, species_start, species_end)
    for seqid, start_i, end_i in coords:
        pass  # todo, modular rnaseq and import?
    # calculate coverage score  (0 - 2)
    ## intergenic (1 / (cov + 1) + 1 / (sc + 1)
    ## CDS and UTR ( 1 - (1 / (cov + 1)) + 1 / (sc + 1))
    ## intron  ( 1 - (1 / (sc + 1)) + 1 / (cov + 1))

    # normalize coverage score by species, category, both



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

