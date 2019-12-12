import sys
import argparse
import HTSeq
import h5py
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
    assert len(sqs) == 1
    return sqs[0]['LN']


def cov_by_chrom(chromosome, length, htseqbam, d_utp=False):
    # todo, refactor so it isn't even passed, that from the h5 files can be wrong when err seqs were filtered
    del length
    length = get_length_from_header(htseqbam, chromosome)
    # returns htseq genomic array
    chromosomes = {chromosome: length}
    cov_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    spliced_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    # 1 below because "pysam uses 0-based coordinates...The only exception is the region string in the fetch() and
    # pileup() methods. This string follows the convention of the samtools command line utilities." oh well.

    for read in htseqbam.fetch(region="{}:1-{}".format(chromosome, length)):
        if not skippable(read):
            standard_ivs, spliced_ivs = get_sense_cov_intervals(read, chromosome, d_utp)
            for iv in standard_ivs:
                cov_array[iv] += 1
            for iv in spliced_ivs:
                spliced_array[iv] += 1

    return cov_array, spliced_array


def extract_np_arrays(cov_array, seqid, length):
    plus = cov_array[HTSeq.GenomicInterval(seqid, 0, length, "+")].array
    minus = cov_array[HTSeq.GenomicInterval(seqid, 0, length, "-")].array
    return plus, minus


def stranded_cov_by_chromosome(htseqbam, seqid, length, d_utp=False):
    cov_array, spliced_array = cov_by_chrom(seqid, length, htseqbam, d_utp)
    cp, cm = extract_np_arrays(cov_array, seqid, length)
    sp, sm = extract_np_arrays(spliced_array, seqid, length)
    return cp, cm, sp, sm


COVERAGE_SETS = ['coverage', 'spliced_coverage']


def setup_output4species(new_h5_path, h5_data, h5_preds, species):
    # open output file
    h5_file = h5py.File(new_h5_path, "w")

    # get output size
    b_species = species.encode("utf-8")  # todo, right encoding?
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
    print("finished sub-setting and sorting existing data", file=sys.stderr)
    return h5_file


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


def gen_coords(h5_sorted):
    """gets unique seqids, range, and seq length from h5 file"""
    # uses tuple with (seqid, max_coord)
    previous, highest = id_and_max(h5_sorted, 0)
    start_i, i = 0, 0
    for i in range(1, h5_sorted['data/seqids'].shape[0]):
        current, max_coord = id_and_max(h5_sorted, i)
        if current != previous:
            yield previous, highest, start_i, i
            start_i = i
            highest = max_coord
            previous = current
        else:
            highest = max(highest, max_coord)
    yield previous, highest, start_i, i


def id_and_max(h5, i):
    return h5['data/seqids'][i], max(h5['data/start_ends'][i])


def pad_cov_right(short_arr, length, fill_value=-1.):
    out = np.full(shape=(length,), fill_value=fill_value)
    out[:short_arr.shape[0]] = short_arr
    return out


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

    # get coverage by chromosome
    for seqid, length, start_i, end_i in coords:
        seqid = seqid.decode('utf-8')
        print(seqid, file=sys.stderr)
        # coverage+, coverage-, spliced_coverage+, spliced_coverage-
        cov_arrays = stranded_cov_by_chromosome(bam, seqid, length, d_utp)
        # split into pieces matching start/ends
        b_seqid = seqid.encode("utf-8")
        print('end at {}'.format(end_i))
        for i in range(start_i, end_i):
            print(i, h5_out['data/seqids'][i])
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
            # export
            write_next_2(h5_out, slices, i)
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
