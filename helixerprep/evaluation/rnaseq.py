import HTSeq
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


def split_coverage(start, end, chromosome, coverage_array):
    # returns np array of coverage
    pass


def main(species, geenuff, bamfile, h5_input, h5_predictions, h5_output, d_utp=False):
    # setup chromosome names & lengths
    engine = create_engine(full_db_path(geenuff), echo=False)
    session = sessionmaker(bind=engine)()
    genome = session.query(orm.Genome).filter(orm.Genome.species == species)

    # open bam
    bam = HTSeq.BAM_Reader(bamfile)

    # get coverage by chromosome
    for coord in genome.coordinates:
        cov_array, spliced_array = cov_by_chrom(coord.seqid, coord.length, bam, d_utp)
        # split into pieces matching start/ends

        # write back to h5 (ori dat, predictions, coverage) subset for species

