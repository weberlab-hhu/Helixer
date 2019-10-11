import HTSeq
import h5py
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from geenuff.base.helpers import full_db_path
from geenuff.base import orm


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
        out.append([HTSeq.GenomicInterval(chromosome, x.start, x.end, strand) for x in arr])
    return out  # list [standard, spliced] of coverage intervals


def cov_by_chrom(chromosome, length, htseqbam, d_utp=False):
    # returns htseq genomic array
    chromosomes = {chromosome: length}

    cov_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    spliced_array = HTSeq.GenomicArray(chromosomes, stranded=True, typecode="i", storage="ndarray")
    for read in htseqbam.fetch(region="{}:0-{}".format(chromosome, length)):
        if not skippable(read):
            standard_ivs, spliced_ivs = get_sense_cov_intervals(read, chromosome, d_utp)
            for iv in standard_ivs:
                cov_array[iv] += 1
            for iv in spliced_ivs:
                spliced_array[iv] += 1

    return cov_array, spliced_array


def extract_np_arrays(cov_array, coord):
    plus = cov_array[HTSeq.GenomicInterval(coord.seqid, 0, coord.length, "+")].array
    minus = cov_array[HTSeq.GenomicInterval(coord.seqid, 0, coord.length, "-")].array
    return plus, minus


def stranded_cov_by_chromosome(htseqbam, coord, d_utp=False):
    cov_array, spliced_array = cov_by_chrom(coord.seqid, coord.length, htseqbam, d_utp)
    cp, cm = extract_np_arrays(cov_array, coord)
    sp, sm = extract_np_arrays(spliced_array, coord)
    return cp, cm, sp, sm


COVERAGE_SETS = ['coverage+', 'coverage-', 'spliced_coverage+', 'spliced_coverage-']


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
                               compression="lzf")
    return h5_file


def write_next_4(h5_out, slices, i):
    for j, key in enumerate(COVERAGE_SETS):
        h5_out['evaluation/' + key][i] = slices[j]


def main(species, geenuff, bamfile, h5_input, h5_predictions, h5_output, d_utp=False):
    # setup chromosome names & lengths
    engine = create_engine(full_db_path(geenuff), echo=False)
    session = sessionmaker(bind=engine)()
    genome = session.query(orm.Genome).filter(orm.Genome.species == species)

    # open data in files
    bam = HTSeq.BAM_Reader(bamfile)
    h5_data = h5py.File(h5_input, "r")
    h5_preds = h5py.File(h5_predictions, "r")

    # setup output file
    h5_out = setup_output4species(h5_output, h5_data, h5_preds, species)

    # get coverage by chromosome
    for coord in genome.coordinates:
        cov_arrays = stranded_cov_by_chromosome(bam, coord, d_utp)
        # split into pieces matching start/ends
        b_seqid = coord.seqid.encode("utf-8")
        for i in range(h5_out['data/species'].shape[0]):
            if h5_out['data/seqids'][i] == b_seqid:  # if output were sorted, this could be more efficient
                start, end = h5_out['data/start_ends'][i]
                # subset and flip to match existing h5 chunks
                is_plus_strand = True
                if end < start:
                    is_plus_strand = False
                    start, end = end, start
                slices = [x[start:end] for x in cov_arrays]
                if not is_plus_strand:
                    slices = [np.flip(x, axis=0) for x in slices]
                # export
                write_next_4(h5_out, slices, i)

