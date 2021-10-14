import sys
import argparse
import HTSeq
import h5py
import random
import os
import shutil
from helixer.evaluation import rnaseq
import copy
from helixer.evaluation.training_rnaseq import species_range


CAGE_COVERAGE_SETS = ['cage_coverage', 'cage_spliced_coverage']


def add_empty_cage_datasets(h5):
    length = h5['data/X'].shape[0]
    chunk_len = h5['data/X'].shape[1]
    if 'evaluation' not in h5.keys():
        h5.create_group('evaluation')
    for key in CAGE_COVERAGE_SETS:
        h5.create_dataset('evaluation/' + key,
                          shape=(length, chunk_len, 1),
                          maxshape=(None, chunk_len, None),
                          dtype="int",
                          compression="lzf",
                          fillvalue=-1)


def cage_coverage_from_coord_to_h5(coord, h5_out, bam, strandedness, chunk_size, memmap_dirs, final_dimension):
    """calculates coverage for a coordinate from bam, saves to h5, returns counts for aggregating"""
    b_seqid, start_i, end_i = coord
    seqid = b_seqid.decode('utf-8')
    print('{}: chunks from {}-{}'.format(seqid, start_i, end_i), file=sys.stderr)
    # prep contiguous bits (h5 will be written in chunks of this size)
    bits_plus, bits_minus = rnaseq.find_contiguous_segments(h5_out, start_i, end_i, chunk_size)
    bits = {"+": bits_plus, "-": bits_minus}

    # calculate coverage
    cov_array, spliced_array, length, counts = rnaseq.cov_by_chrom(seqid, bam, strandedness, memmap_dirs)
    all_coverage = {CAGE_COVERAGE_SETS[0]: cov_array, CAGE_COVERAGE_SETS[1]: spliced_array}

    # write to h5 contiguous bit by contiguous bit
    for direction in bits:
        for cov_type in all_coverage:
            htseq_array = all_coverage[cov_type]
            array = htseq_array[HTSeq.GenomicInterval(seqid, 0, length, direction)].array
            rnaseq.write_in_bits(array, bits[direction], h5_out['evaluation/{}'.format(cov_type)], chunk_size,
                                 final_dimension)

    return counts


def main(species, bam, h5_data, strandedness):

    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation, score, & metadata placeholders if they don't exist
    # (evaluation/coverage, "/spliced_coverage, scores/*, meta/*)

    try:
        old_shape = h5['evaluation/' + CAGE_COVERAGE_SETS[0]].shape
        # if the dataset is already there, enlarge and increment
        assert len(old_shape) == 3
        final_dimension = old_shape[2]  # +1 for incrementing -1 to count from 0, stays the same
        for cov_set in CAGE_COVERAGE_SETS:
            new_size = list(old_shape[:2])
            new_size.append(final_dimension + 1)
            h5['evaluation/' + cov_set].resize(tuple(new_size))

    except KeyError:
        add_empty_cage_datasets(h5)
        final_dimension = 0

    # identify regions in h5 corresponding to species
    species_start, species_end = species_range(h5, species)

    # insert coverage into said regions
    coords = rnaseq.gen_coords(h5, species_start, species_end)
    print('start, end', species_start, species_end, file=sys.stderr)
    cov_counts = copy.deepcopy(rnaseq.COVERAGE_COUNTS)  # tracks number reads, bp coverage, bp spliced coverage

    chunk_size = h5['evaluation/cage_coverage'].shape[1]
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
        cage_coverage_from_coord_to_h5(
            coord, h5, bam=htseqbam, strandedness=strandedness,
            chunk_size=chunk_size, memmap_dirs=memmap_dirs, final_dimension=final_dimension)

    for d in memmap_dirs:
        shutil.rmtree(d)  # rm -r

    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', help="species name, matching geenuff db and h5 files", required=True)
    parser.add_argument('-d', '--h5-data', help='h5 data file (with /data/{X, y, species, seqids, etc...}) '
                                                'to which evaluation coverage will be ADDED!',
                        required=True)
    parser.add_argument('-b', '--bam', help='sorted (and indexed) bam file. Omit to only score existing coverage.',
                        required=True)
    parser.add_argument('--first-read-is-sense-strand',
                        help='first strand is sense strand, e.g. reads are not from a typical dUTP protocol',
                        action='store_true')
    parser.add_argument('--second-read-is-sense-strand', action="store_true",
                        help='second strand is sense strand, e.g. reads ARE from a typical dUTP protocol')
    parser.add_argument('--unstranded', action='store_true',
                        help='reads are not stranded, final "strand" will simply arbitrarily match read strand')
    parser.add_argument('-r', '--skip-scoring', action="store_true",
                        help="set this to add coverage to the bam file, but not 'score' it (raw cov. only)")
    args = parser.parse_args()

    if args.first_read_is_sense_strand:
        strandedness = 1
    elif args.second_read_is_sense_strand:
        strandedness = 2
    elif args.unstranded:
        strandedness = None
    else:
        print("Exactly one of [--first-read-is-sense-strand, --second-read-is-sense-strand, "
              "--unstranded] must be set", file=sys.stderr)
        sys.exit(1)

    main(args.species,
         args.bam,
         args.h5_data,
         strandedness)
