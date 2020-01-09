import sys
import argparse
import HTSeq
import h5py
import copy
import numpy as np
from helixerprep.core.helpers import mk_keys


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


def get_sense_strand(read):
    """returns strand of original mRNA

    assumes dUTP protocol, that is, 2nd read is sense strand"""
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
    strand = read.iv.strand
    assert strand in ['-', "+"]

    if flip:
        if strand == "+":
            strand = "-"
        else:
            strand = "+"

    return strand


def get_sense_cov_intervals(read, chromosome, d_utp):
    """gets intervals for standard and spliced coverage"""
    if d_utp:
        strand = get_sense_strand(read)
    else:
        # take raw strand for un-stranded protocol, bc it's easier than dealing with the if/else of un-stranded
        strand = read.iv.strand
    # todo, stranded but not d_utp

    # ignore clipping and other info for now, take only coverage regions
    standard_raw = [x for x in read.cigar if is_coverage(x)]
    spliced_raw = [x for x in read.cigar if is_spliced_coverage(x)]

    out = []
    for arr in [standard_raw, spliced_raw]:
        out.append([HTSeq.GenomicInterval(chromosome, x.ref_iv.start, x.ref_iv.end, strand) for x in arr])
    return out  # list [standard, spliced] of coverage intervals


def get_length_from_header(htseqbam, chromosome):
    hdict = htseqbam.get_header_dict()
    sqs = [x for x in hdict['SQ'] if x['SN'] == chromosome]
    assert len(sqs) == 1, "found {}, searching for {}".format(sqs, chromosome)
    return sqs[0]['LN']


COVERAGE_COUNTS = {'reads': 0, 'coverage': 0, 'spliced_coverage': 0}


def cov_by_chrom(chromosome, htseqbam, d_utp=False):
    length = get_length_from_header(htseqbam, chromosome)
    # returns htseq genomic array
    chromosomes = {chromosome: length}
    cov_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    spliced_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    # 1 below because "pysam uses 0-based coordinates...The only exception is the region string in the fetch() and
    # pileup() methods. This string follows the convention of the samtools command line utilities." oh well.
    counts = copy.deepcopy(COVERAGE_COUNTS)
    for read in htseqbam.fetch(region="{}:1-{}".format(chromosome, length)):
        if not skippable(read):
            counts['reads'] += 1
            standard_ivs, spliced_ivs = get_sense_cov_intervals(read, chromosome, d_utp)
            for iv in standard_ivs:
                cov_array[iv] += 1
                counts['coverage'] += iv.end - iv.start
            for iv in spliced_ivs:
                spliced_array[iv] += 1
                counts['spliced_coverage'] += iv.end - iv.start

    return cov_array, spliced_array, length, counts


def extract_np_arrays(cov_array, seqid, length):
    plus = cov_array[HTSeq.GenomicInterval(seqid, 0, length, "+")].array
    minus = cov_array[HTSeq.GenomicInterval(seqid, 0, length, "-")].array
    return plus, minus


def stranded_cov_by_chromosome(htseqbam, seqid, d_utp=False):
    cov_array, spliced_array, length, counts = cov_by_chrom(seqid, htseqbam, d_utp)
    cp, cm = extract_np_arrays(cov_array, seqid, length)
    sp, sm = extract_np_arrays(spliced_array, seqid, length)
    return cp, cm, sp, sm, counts


COVERAGE_SETS = ['coverage', 'spliced_coverage']


def setup_output4species(new_h5_path, h5_data, h5_preds, species):
    # open output file
    h5_file = h5py.File(new_h5_path, "w")

    # get output size
    b_species = species.encode("utf-8")
    length = len([x for x in h5_data['data/species'] if x == b_species])

    # setup empty datasets
    # data
    h5_file.create_group('data')
    for key in h5_data['data'].keys():
        dset = h5_data['data/' + key]
        shape = list(dset.shape)
        shape[0] = length
        h5_file.create_dataset('data/' + key,
                               shape=shape,
                               maxshape=[None] + shape[1:],
                               dtype=dset.dtype,
                               compression="lzf")
    # predictions
    h5_file.create_dataset('predictions',
                           shape=[length] + list(h5_preds['predictions'].shape[1:]),
                           maxshape=[None] + list(h5_preds['predictions'].shape[1:]),
                           dtype='f',
                           compression='lzf')
    # evaluation (coverage)
    h5_file.create_group('evaluation')
    chunk_len = h5_data['data/X'].shape[1]
    for key in COVERAGE_SETS:
        h5_file.create_dataset('evaluation/' + key,
                               shape=(length, chunk_len),
                               maxshape=(None, chunk_len),
                               dtype="int",
                               compression="lzf",
                               data=np.full(fill_value=-1, shape=(length, chunk_len)))

    # get output mask & sort
    mask, lexsort = mask_and_sort(h5_data, species)

    # and copy relevant data in
    for key in h5_data['data'].keys():
        full_key = 'data/' + key
        tosave = h5_data[full_key][:]
        tosave = tosave[mask]
        tosave = tosave[lexsort]
        h5_file[full_key][:] = tosave
    h5_file['predictions'][:] = h5_preds['predictions'][:][mask][lexsort]

    add_meta(h5_file)

    print("finished sub-setting and sorting existing data", file=sys.stderr)
    return h5_file


def add_meta(h5):
    """add organized hierarchy for by-species meta attributes to be attached to"""
    h5.create_group('meta')
    for key in ['bamfile', 'total_reads', 'total_coverage', 'total_spliced_coverage', 'start_end_i']:
        h5.create_group('meta/' + key)


def mask_and_sort(h5_data, species):
    mask = np.array(h5_data['/data/species'][:] == species.encode('utf-8'))
    pre_seq_keys = [for_sorting(x) for x in mk_keys(h5_data)]
    seq_keys = np.array(pre_seq_keys)[mask]
    lexsort = np.lexsort(np.flip(seq_keys.T, axis=0))
    return mask, lexsort


def for_sorting(four):
    zero = int(''.join([str(x) for x in four[0]]))
    one = int(''.join([str(x) for x in four[1]]))
    two = int(four[3] < four[2])  # is_not_plus_strand (so plus strand gets 0 and sorts first)
    return zero, one, two, four[2], four[3]


def write_next_2(h5_out, slices, i):
    for j, key in enumerate(COVERAGE_SETS):
        h5_out['evaluation/' + key][i] = slices[j]


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


def pad_cov_right(short_arr, length, fill_value=-1.):
    out = np.full(shape=(length,), fill_value=fill_value)
    out[:short_arr.shape[0]] = short_arr
    return out


def arrange_slice_for_h5(cov_arrays, i, h5_out, pad_to, b_seqid):
    assert h5_out['data/seqids'][i] == b_seqid
    start, end = h5_out['data/start_ends'][i]
    # subset and flip to match existing h5 chunks
    is_plus_strand = True
    if end < start:
        is_plus_strand = False
        start, end = end, start
    if is_plus_strand:
        slices = [cov_arrays[i][start:end] for i in [0, 2]]
    else:
        slices = [np.flip(cov_arrays[i][start:end], axis=0) for i in [1, 3]]
    if end - start != pad_to:
        slices = [pad_cov_right(x, pad_to) for x in slices]
        print('padding {}-{} + is {}'.format(start, end, is_plus_strand))
    return slices


def coverage_from_coord_to_h5(coord, h5_out, bam, d_utp, pad_to):
    """calculates coverage for a coordinate from bam, saves to h5, returns counts for aggregating"""
    b_seqid, start_i, end_i = coord
    seqid = b_seqid.decode('utf-8')
    print('{}: chunks from {}-{}'.format(seqid, start_i, end_i), file=sys.stderr)
    # coverage+, coverage-, spliced_coverage+, spliced_coverage-
    cov_arrays = stranded_cov_by_chromosome(bam, seqid, d_utp)
    counts = cov_arrays[4]
    cov_arrays = cov_arrays[:4]
    # split into pieces matching start/ends
    for i in range(start_i, end_i):
        slices = arrange_slice_for_h5(cov_arrays, i, h5_out, pad_to, b_seqid)
        # export
        # todo, write in larger chunks?? could read as well...
        write_next_2(h5_out, slices, i)
    return counts


def main(species, bamfile, h5_input, h5_predictions, h5_output, d_utp=False):
    # open data in files
    bam = HTSeq.BAM_Reader(bamfile)
    h5_data = h5py.File(h5_input, "r")
    h5_preds = h5py.File(h5_predictions, "r")

    # setup output file
    h5_out = setup_output4species(h5_output, h5_data, h5_preds, species)
    h5_data.close()
    h5_preds.close()

    # setup chromosome names & lengths
    coords = gen_coords(h5_out)
    pad_to = h5_out['evaluation/coverage'].shape[1]

    counts = copy.deepcopy(COVERAGE_COUNTS)
    # get coverage by chromosome
    for coord in coords:
        # writes coverage to h5, return totals for aggregating
        coord_counts = coverage_from_coord_to_h5(coord, h5_out, bam, d_utp, pad_to)
        for key in counts:
            counts[key] += coord_counts[key]

    # write meta info to h5
    h5_out['meta/bamfile'].attrs.create(name=species, data=bamfile)
    for key in counts:
        h5_out['meta/total_' + key].attrs.create(name=species, data=counts[key])

    # one species per file, but start end included for consistency
    h5_out['meta/start_end_i'].attrs.create(name=species, data=(0, h5_out['data/X'].shape[0]))
    h5_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', help="species name, matching geenuff db and h5 files", required=True)
    parser.add_argument('-b', '--bam', help='sorted (and indexed) bam file', required=True)
    parser.add_argument('-d', '--h5_data', help='h5 data file (with /data/{X, y, species, seqids, etc...})',
                        required=True)
    parser.add_argument('-p', '--h5_predictions', help='h5 predictions file with /predictions matching data file',
                        required=True)
    parser.add_argument('-o', '--out', help='output h5 file', required=True)
    parser.add_argument('-x', '--not_dUTP', help='bam does not contain stranded (from typical dUTP protocol) reads',
                        action='store_true')
    args = parser.parse_args()
    main(args.species,
         args.bam,
         args.h5_data,
         args.h5_predictions,
         args.out,
         not args.not_dUTP)
