#! /usr/bin/env python3

import sys
import argparse
import HTSeq
import h5py
import random
import os
import shutil
import copy

import h5py.utils
import numpy as np
from multiprocessing import Pool


class ContiguousBit:
    def __init__(self, seqid, start_ends, start_i_h5, end_i_h5):
        self.seqid = seqid
        self.start_ends = start_ends
        # indexes in h5 file
        self.start_i_h5 = start_i_h5
        self.end_i_h5 = end_i_h5
        assert len(self.start_ends) == end_i_h5 - start_i_h5, '{} {} {}'.format(len(self.start_ends), start_i_h5,
                                                                                end_i_h5)

    def __repr__(self):
        return "array from {}-{}; h5 from [{},{}) in h5 in {} pieces".format(self.start_ends[0][0],
                                                                             self.start_ends[-1][1],
                                                                             self.start_i_h5,
                                                                             self.end_i_h5,
                                                                             len(self.start_ends))


def get_length_from_header(htseqbam, chromosome):
    hdict = htseqbam.get_header_dict()
    sqs = [x for x in hdict['SQ'] if x['SN'] == chromosome]
    assert len(sqs) == 1, "found {}, searching for {}".format(sqs, chromosome)
    return sqs[0]['LN']


def skippable(read):
    out = False
    if read.iv is None:
        out = True
    elif read.not_primary_alignment:
        out = True
    elif read.failed_platform_qc:
        out = True
    return out


def is_coverage(cigar_entry):
    if cigar_entry.type in ["=", "X", "M"]:
        return True
    else:
        return False


def is_spliced_coverage(cigar_entry):
    # including D because interpreting deletions as splicing is small count error
    # but if any splicing is marked with D, missing this would be a large count error
    if cigar_entry.type in ["N", "D"]:
        return True
    else:
        return False


def get_sense_strand(read, sense_strand=2):
    """returns strand of original mRNA

    default is dUTP protocol, that is, 2nd read is sense strand"""

    assert sense_strand in [1, 2]

    # this part assumes sense strand is 2
    if not read.paired_end:
        flip = True
    elif read.pe_which == "first":
        flip = True
    elif read.pe_which == "second":
        flip = False
    else:
        raise Exception("failed read strand interpretation ({}, {}, {})".format(read.paired_end,
                                                                                read.pe_which,
                                                                                read))
    if sense_strand != 2:
        flip = not flip

    strand = read.iv.strand
    assert strand in ['-', "+"]

    if flip:
        if strand == "+":
            strand = "-"
        else:
            strand = "+"

    return strand


def get_sense_cov_intervals(read, chromosomes, strandedness, shift):
    """gets intervals for standard and spliced coverage"""
    chromosome = list(chromosomes.keys())[0]
    length = chromosomes[chromosome]

    if strandedness is not None:
        strand = get_sense_strand(read, strandedness)
    else:
        # take raw strand for un-stranded protocol, bc it's easier than dealing with the if/else of un-stranded
        strand = read.iv.strand

    # ignore clipping and other info for now, take only coverage regions
    standard_raw = [x for x in read.cigar if is_coverage(x)]
    spliced_raw = [x for x in read.cigar if is_spliced_coverage(x)]

    # handle special cases where we are not literally taking the read coverage
    if shift:
        assert strandedness is None, "shift not yet implemented for stranded data."
        start, end = get_shifted_interval(read)

        # handle edge of chromosome cases
        start = max(0, start)
        end = min(end, length)
        return [[HTSeq.GenomicInterval(chromosome, start, end, strand)], []]  # standard, spliced (which is absent)

    # otherwise continue with read coverage
    out = []
    for arr in [standard_raw, spliced_raw]:
        out.append([HTSeq.GenomicInterval(chromosome, x.ref_iv.start, x.ref_iv.end, strand) for x in arr])
    return out  # list [standard, spliced] of coverage intervals


def get_shifted_interval(read):
    """shift interval to tn5 cut site"""
    # tn5 bind about 9 bp of DNA, and adds adapters on the 5' end in each direction
    # so the middle of the binding site is just downstream of the 5' end
    # 5' adapter ->[____x____]....
    # .............[____x____] <- retpada '5
    # convention is to consider the center as +4 bp on "+" strand, and -5 bp on "-" strand reads
    if read.iv.strand == "+":
        # center is ~4.5 bp into the read (using standard +4)
        return read.iv.start + 4, read.iv.end
    elif read.iv.strand == "-":
        # center is ~4.5 bp into the read (using standard -5)
        # start < end for HTSeq, regardless of 5' vs 3'.
        # i.e. on the "-" strand, the 5' read begin is the 'end'
        return read.iv.start, read.iv.end - 5
    else:
        raise ValueError(f'unknown strand {read.iv.strand}')


def write_in_bits(array, contiguous_bits, h5_dataset, chunk_size, target_row=None):
    for bit in contiguous_bits:
        write_a_bit(array, bit, h5_dataset, chunk_size, target_row)


def write_a_bit(array, bit, h5_dataset, chunk_size, target_row=None):
    start_array = bit.start_ends[0][0]
    end_array = bit.start_ends[-1][1]

    is_plus_strand = start_array < end_array
    # extract sub region
    if is_plus_strand:
        array_slice = array[start_array:end_array]
    else:
        array_slice = np.flip(array[end_array:start_array], axis=0)

    # pad if need be
    raw_length = array_slice.shape[0]
    if raw_length % chunk_size:
        n_chunks = raw_length // chunk_size + 1
        # bc the edges will now be split off as "non contiguous" for easier handling
        # todo, clean up
        assert n_chunks == 1
        array_slice = pad_cov_right(array_slice, n_chunks * chunk_size)
    else:
        n_chunks = raw_length // chunk_size

    # shape into chunks
    array_slice = np.reshape(array_slice, [n_chunks, chunk_size])
    # and write to file
    if target_row is None:
        h5_dataset[bit.start_i_h5:bit.end_i_h5] = array_slice
    else:
        h5_dataset[bit.start_i_h5:bit.end_i_h5, :, target_row] = array_slice


def find_contiguous_segments(h5, start_i, end_i, chunk_size):
    bits_plus = []
    bits_minus = []

    seqids = h5['data/seqids'][start_i:end_i]
    start_ends = h5['data/start_ends'][start_i:end_i]

    # these reset every step
    prev_seqid = seqids[0]
    prev_start, prev_end = start_ends[0]
    prev_is_plus = prev_start < prev_end

    # these reset every contiguous bit
    current_start_ends = [(prev_start, prev_end)]
    curr_start_i_h5 = start_i

    for i_rel in range(1, start_ends.shape[0]):
        curr_seqid = seqids[i_rel]
        curr_start, curr_end = start_ends[i_rel]
        curr_is_plus = curr_start < curr_end
        # still necessary to handle edge-cases with padding, and detect +- strand flip
        if matches_and_no_end_case((curr_start, prev_start),
                                   (curr_end, prev_end),
                                   (curr_is_plus, prev_is_plus),
                                   chunk_size):
            current_start_ends.append((curr_start, curr_end))
        # else if there was a break in the continuity, save
        else:
            continguous_bit = ContiguousBit(seqid=prev_seqid,
                                            start_ends=current_start_ends,
                                            start_i_h5=curr_start_i_h5,
                                            end_i_h5=curr_start_i_h5 + len(current_start_ends))
            # save
            if prev_is_plus:
                bits_plus.append(continguous_bit)
            else:
                bits_minus.append(continguous_bit)

            # reset after save
            current_start_ends = [(curr_start, curr_end)]
            curr_start_i_h5 = start_i + i_rel
        # step
        prev_seqid, prev_start, prev_end, prev_is_plus = curr_seqid, curr_start, curr_end, curr_is_plus
    # save final step
    continguous_bit = ContiguousBit(seqid=prev_seqid,
                                    start_ends=current_start_ends,
                                    start_i_h5=curr_start_i_h5,
                                    end_i_h5=curr_start_i_h5 + len(current_start_ends))
    if prev_is_plus:
        bits_plus.append(continguous_bit)
    else:
        bits_minus.append(continguous_bit)

    return bits_plus, bits_minus


def add_empty_ngs_datasets(h5, n):
    length = h5['data/X'].shape[0]
    chunk_len = h5['data/X'].shape[1]
    if 'evaluation' not in h5.keys():
        h5.create_group('evaluation')
    for key in NGS_COVERAGE_SETS:
        h5.create_dataset('evaluation/' + key,
                          shape=(length, chunk_len, n),
                          maxshape=(None, chunk_len, None),
                          chunks=(1, chunk_len, 1),
                          dtype="int",
                          compression="lzf",
                          fillvalue=-1,
                          shuffle=True)


def add_empty_cov_meta(h5, n):
    meta_str = META_STR
    if meta_str not in h5['evaluation'].keys():
        h5.create_group(f'evaluation/{meta_str}')
    h5.create_dataset(f'evaluation/{meta_str}/{BAMFILES_DATASET}',
                      shape=(n,),
                      maxshape=(None, ),
                      dtype='S512',
                      fillvalue=''.encode('ASCII'))


def cov_by_chrom(chrm_bam_strandedness_shift):
    chromosome, bam_file, strandedness,  shift = chrm_bam_strandedness_shift
    htseqbam = H5_BAMS[bam_file]

    # setup dir for memmap array (AKA, don't try and store the whole chromosome in RAM
    memmap_dirs = ("memmap_dir_{}".format(random.getrandbits(128)),
                   "memmap_dir_{}".format(random.getrandbits(128)))
    for d in memmap_dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    length = get_length_from_header(htseqbam, chromosome)
    # returns htseq genomic array
    chromosomes = {chromosome: length}
    storage = "memmap"
    cov_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage=storage, memmap_dir=memmap_dirs[0])
    spliced_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i",
                                       storage=storage, memmap_dir=memmap_dirs[1])
    # 1 below because "pysam uses 0-based coordinates...The only exception is the region string in the fetch() and
    # pileup() methods. This string follows the convention of the samtools command line utilities." oh well.
    counts = copy.deepcopy(COVERAGE_COUNTS)
    for read in htseqbam.fetch(region="{}:1-{}".format(chromosome, length)):
        if not skippable(read):
            counts['reads'] += 1
            standard_ivs, spliced_ivs = get_sense_cov_intervals(read, chromosomes, strandedness,
                                                                shift=shift)
            for iv in standard_ivs:
                cov_array[iv] += 1
                counts['coverage'] += iv.end - iv.start
            for iv in spliced_ivs:
                spliced_array[iv] += 1
                counts['spliced_coverage'] += iv.end - iv.start

    return cov_array, spliced_array, length, counts, memmap_dirs


def gen_coords(h5_sorted, sp_start_i=0, sp_end_i=None):
    """gets unique seqids, range, and seq length from h5 file"""
    # uses tuple with (seqid, max_coord)
    previous = just_seqid(h5_sorted, sp_start_i)
    coord_start_i = sp_start_i
    if sp_end_i is None:
        sp_end_i = h5_sorted['data/seqids'].shape[0]
    for i in range(sp_start_i + 1, sp_end_i):
        current = just_seqid(h5_sorted, i)
        if current != previous:
            yield previous, coord_start_i, i
            coord_start_i = i
            previous = current
    yield previous, coord_start_i, sp_end_i


def just_seqid(h5, i):
    return h5['data/seqids'][i]


def matches_and_no_end_case(starts, ends, is_plusses, chunk_size):
    curr_start, prev_start = starts
    curr_end, prev_end = ends
    # we can assume contiguity and same sequence, only break at strand flip
    if is_plusses[0] != is_plusses[1]:
        out = False
    # detect edge cases (bc padding choices cause non-contiguity of minus strand edge case)
    # break both before and after edge case (to make sure it's on it's own)
    elif abs(curr_start - curr_end) != chunk_size:
        out = False
    elif abs(prev_start - prev_end) != chunk_size:
        out = False
    else:  # better be contiguous then
        out = True
    return out


def species_range(h5, species):
    mask = np.array(h5['/data/species'][:] == species.encode('utf-8'))
    stretches = list(get_bool_stretches(mask.tolist()))  # [(False, Count), (True, Count), (False, Count)]
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


def pad_cov_right(short_arr, length, fill_value=-1.):
    out = np.full(shape=(length,), fill_value=fill_value)
    out[:short_arr.shape[0]] = short_arr
    return out


def cage_coverage_from_coord_to_h5(coord, h5_out, strandedness, chunk_size, old_final_dimension, threads,
                                   shift):
    """calculates coverage for a coordinate from bam, saves to h5, returns counts for aggregating"""
    b_seqid, start_i, end_i = coord
    seqid = b_seqid.decode('utf-8')
    print('{}: chunks from {}-{}'.format(seqid, start_i, end_i), file=sys.stderr)
    # prep contiguous bits (h5 will be written in chunks of this size)
    bits_plus, bits_minus = find_contiguous_segments(h5_out, start_i, end_i, chunk_size)
    bits = {"+": bits_plus, "-": bits_minus}

    # calculate coverage
    nbams = len(H5_BAMS)
    # todo, guarnatee order?
    mapargs = zip([seqid] * nbams, H5_BAMS.keys(), [strandedness] * nbams, [shift] * nbams)
    if threads > 1:
        with Pool(threads) as p:
            coverage_out = p.map(cov_by_chrom, mapargs)
    else:
        coverage_out = []
        for marg in mapargs:
            coverage_out.append(cov_by_chrom(marg))

    for i, cov_sample in enumerate(coverage_out):
        cov_array, spliced_array, length, counts, memmap_dirs = cov_sample
        all_coverage = {NGS_COVERAGE_SETS[0]: cov_array, NGS_COVERAGE_SETS[1]: spliced_array}

        # write to h5 contiguous bit by contiguous bit
        for direction in bits:
            for cov_type in all_coverage:
                htseq_array = all_coverage[cov_type]
                array = htseq_array[HTSeq.GenomicInterval(seqid, 0, length, direction)].array
                write_in_bits(array, bits[direction], h5_out['evaluation/{}'.format(cov_type)], chunk_size,
                              old_final_dimension + i)

        for d in memmap_dirs:
            try:
                shutil.rmtree(d)  # rm -r
            except OSError:  # for e.g. mounted file systems that are slow...
                shutil.rmtree(d)

    return counts


def main(species, h5_data, strandedness, prefix, threads, shift):
    # open h5
    h5 = h5py.File(h5_data, 'r+')
    # create evaluation, score, & metadata placeholders if they don't exist
    # (evaluation/{prefix}_coverage, "evaluation/{prefix}_spliced_coverage, , meta/*)

    nbams = len(H5_BAMS)
    try:
        old_shape = h5['evaluation/' + NGS_COVERAGE_SETS[0]].shape
        # if the dataset is already there, enlarge and increment
        assert len(old_shape) == 3
        old_final_dimension = old_shape[2]  # +1 for incrementing -1 to count from 0, stays the same
        for cov_set in NGS_COVERAGE_SETS:
            new_size = list(old_shape[:2])
            new_size.append(old_final_dimension + nbams)
            h5['evaluation/' + cov_set].resize(tuple(new_size))
    except KeyError:
        add_empty_ngs_datasets(h5, nbams)
        old_final_dimension = 0

    try:
        h5[f'evaluation/{META_STR}/{BAMFILES_DATASET}'].resize((old_final_dimension + nbams,))
    except KeyError:
        add_empty_cov_meta(h5, nbams)

    bams_as_meta = np.array([str(x).encode('ASCII') for x in H5_BAMS.keys()])
    h5[f'evaluation/{META_STR}/{BAMFILES_DATASET}'][old_final_dimension:old_final_dimension + nbams] = bams_as_meta

    # identify regions in h5 corresponding to species
    species_start, species_end = species_range(h5, species)

    # insert coverage into said regions
    coords = gen_coords(h5, species_start, species_end)
    print('start, end', species_start, species_end, file=sys.stderr)

    chunk_size = h5['evaluation/' + NGS_COVERAGE_SETS[0]].shape[1]

    # open bam (read alignment file)

    for coord in coords:
        print(coord, file=sys.stderr)
        cage_coverage_from_coord_to_h5(
            coord, h5, strandedness=strandedness,
            chunk_size=chunk_size, old_final_dimension=old_final_dimension,
            threads=threads, shift=shift)

    h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', help="species name, matching geenuff db and h5 files", required=True)
    parser.add_argument('-d', '--h5-data', help='h5 data file (with /data/{X, y, species, seqids, etc...}) '
                                                'to which evaluation coverage will be added',
                        required=True)
    parser.add_argument('-b', '--bam', help='sorted (and indexed) bam file. Omit to only score existing coverage.',
                        nargs='+', required=True)
    parser.add_argument('--dataset-prefix', help="prefix for the datasets file to store the resulting coverage "
                                                 "current expected values are 'rnaseq', or 'cage' (default). "
                                                 "datasets will be /evaluation/{prefix}_(spliced_)coverage ",
                        default='cage')
    parser.add_argument('--first-read-is-sense-strand',
                        help='first strand is sense strand, e.g. reads are not from a typical dUTP protocol',
                        action='store_true')
    parser.add_argument('--second-read-is-sense-strand', action="store_true",
                        help='second strand is sense strand, e.g. reads ARE from a typical dUTP protocol')
    parser.add_argument('--unstranded', action='store_true',
                        help='reads are not stranded, final "strand" will simply arbitrarily match read strand')
    parser.add_argument('--threads', default=8, help="how many threads, set to a value <= 1 to not use multiprocessing",
                        type=int)
    parser.add_argument('--shift', action='store_true',
                        help='shift reads +4 (+ strand) or -5 (- strand) base pairs as is typically done for ATAC-seq '
                             'data')
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

    if not os.path.exists(args.h5_data):
        raise FileNotFoundError(f"file {args.h5_data} doesn't exist")

    try:
        h5py.File(args.h5_data, "r")
    except OSError as e:
        raise OSError(f"{args.h5_data} is not a h5 file, please provide a valid h5 file") from e

    H5_BAMS = {}
    for bam in args.bam:
        H5_BAMS[bam] = HTSeq.BAM_Reader(bam)

    pfx = args.dataset_prefix

    BAMFILES_DATASET = 'bam_files'
    NGS_COVERAGE_SETS = [f'{pfx}_coverage', f'{pfx}_spliced_coverage']
    COVERAGE_COUNTS = {'reads': 0, 'coverage': 0, 'spliced_coverage': 0}
    META_STR = f'{pfx}_meta'

    main(args.species,
         args.h5_data,
         strandedness,
         pfx,
         args.threads,
         args.shift)
