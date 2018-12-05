import enum


def join_to_enum(name, *args):
    """joins enums from args into returned enum"""
    enum_bits = []
    for cls in args:
        enum_bits += list(cls)
    out = enum.Enum(name, [(x.name, x.value) for x in enum_bits])
    return out


class SuperLocus(enum.Enum):
    coding_gene = 'coding_gene'
    non_coding_gene = 'non_coding_gene'
    pseudogene = 'pseudogene'
    operon = 'operon'


class TranscriptLevelNice(enum.Enum):
    mRNA = 'mRNA'
    tRNA = 'tRNA'
    rRNA = 'rRNA'
    miRNA = 'miRNA'
    snoRNA = 'snoRNA'
    snRNA = 'snRNA'
    SRP_RNA = 'SRP_RNA'
    lnc_RNA = 'lnc_RNA'
    pre_miRNA = 'pre_miRNA'
    RNase_MRP_RNA = 'RNase_MRP_RNA'


class TranscriptLevelInput(enum.Enum):
    transcript = 'transcript'
    primary_transcript = 'primary_transcript'
    pseudogenic_transcript = 'pseudogenic_transcript'


TranscriptLevelAll = join_to_enum('TranscriptLevel', TranscriptLevelNice, TranscriptLevelInput)


# FEATURES
# transcription related features
class TranscribedGeneral(enum.Enum):
    transcription_start_site = 'TSS'
    transcription_termination_site = 'TTS'
    donor_splice_site = 'DSS'
    acceptor_splice_site = 'ASS'


class TranscribedInput(enum.Enum):
    exon = 'exon'
    intron = 'intron'
    five_prime_UTR = 'five_prime_UTR'
    three_prime_UTR = 'three_prime_UTR'

class TranscribedStatus(enum.Enum):
    in_raw_transcript = 'in_raw_transcript'
    in_intron = 'in_intron'


class TranscribedTransSplice(enum.Enum):
    donor_trans_splice_site = 'donor_trans_splice_site'
    acceptor_trans_splice_site = 'acceptor_trans_splice_site'


# joining transcription related
TranscribedAll = join_to_enum('TranscribedFeature', TranscribedGeneral, TranscribedStatus, TranscribedTransSplice,
                                  TranscribedInput)

TranscribedNice = join_to_enum('TranscribedNice', TranscribedGeneral, TranscribedStatus, TranscribedTransSplice)


# translation related features
class TranslatedInput(enum.Enum):
    cds = 'CDS'


class TranslatedGeneral(enum.Enum):
    start_codon = 'start_codon'
    stop_codon = 'stop_codon'


class TranslatedStatus(enum.Enum):
    in_translated_region = 'in_translated_region'  # so between start and stop codons, could still be in an intron


TranslatedAll = join_to_enum('TranslatedFeatureType', TranslatedInput, TranslatedGeneral, TranslatedStatus)

TranslatedNice = join_to_enum('TranslatedNice', TranslatedStatus, TranslatedGeneral)


# errror features
class ErrorFeature(enum.Enum):
    error = 'error'


# ignorable known
class IgnorableFeatures(enum.Enum):
    region = 'region'
    chromosome = 'chromosome'
    supercontig = 'supercontig'
    match = 'match'
    cDNA_match = 'cDNA_match'


# All known features (else error on import)
AllKnown = join_to_enum('AllKnown', SuperLocus, TranscriptLevelAll, TranslatedAll, TranscribedAll, ErrorFeature,
                        IgnorableFeatures)

AllKeepable = join_to_enum('AllKeepable', SuperLocus, TranscriptLevelNice, TranslatedNice, TranscribedNice,
                           ErrorFeature)
