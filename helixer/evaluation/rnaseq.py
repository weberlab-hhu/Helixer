import sys
import argparse
import HTSeq
import h5py
import copy
import random
import os
import shutil
import numpy as np
from helixer.core.helpers import mk_keys












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
                               fillvalue=-1)

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








def coverage_from_coord_to_h5(coord, h5_out, bam, strandedness, chunk_size, memmap_dirs):
    """calculates coverage for a coordinate from bam, saves to h5, returns counts for aggregating"""
    b_seqid, start_i, end_i = coord
    seqid = b_seqid.decode('utf-8')
    print('{}: chunks from {}-{}'.format(seqid, start_i, end_i), file=sys.stderr)
    # prep contiguous bits (h5 will be written in chunks of this size)
    bits_plus, bits_minus = find_contiguous_segments(h5_out, start_i, end_i, chunk_size)
    bits = {"+": bits_plus, "-": bits_minus}

    # calculate coverage
    cov_array, spliced_array, length, counts = cov_by_chrom(seqid, bam, strandedness, memmap_dirs)
    all_coverage = {"coverage": cov_array, "spliced_coverage": spliced_array}

    # write to h5 contiguous bit by contiguous bit
    for direction in bits:
        for cov_type in all_coverage:
            htseq_array = all_coverage[cov_type]
            array = htseq_array[HTSeq.GenomicInterval(seqid, 0, length, direction)].array
            write_in_bits(array, bits[direction], h5_out['evaluation/{}'.format(cov_type)], chunk_size)

    return counts


def main(species, bamfile, h5_input, h5_predictions, h5_output, d_utp=False):

    # change d_utp to strandedness (just 2 and None, to be consistent with old implementation, todo fix w/ argparse)
    if d_utp:
        strandedness = 2
    else:
        strandedness = None

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
    chunk_size = h5_out['evaluation/coverage'].shape[1]

    counts = copy.deepcopy(COVERAGE_COUNTS)
    # setup dir for memmap array (AKA, don't try and store the whole chromosome in RAM
    memmap_dirs = ["memmap_dir_{}".format(random.getrandbits(128)),
                   "memmap_dir_{}".format(random.getrandbits(128))]
    for d in memmap_dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    # get coverage by chromosome
    for coord in coords:
        # writes coverage to h5, return totals for aggregating
        coord_counts = coverage_from_coord_to_h5(coord, h5_out, bam, strandedness, chunk_size, memmap_dirs)
        for key in counts:
            counts[key] += coord_counts[key]

    for d in memmap_dirs:
        shutil.rmtree(d)

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
