import enum


# Enum makers
def join_to_enum(name, *args):
    """joins enums from args into returned enum"""
    enum_bits = []
    for cls in args:
        enum_bits += list(cls)
    out = enum.Enum(name, [(x.name, x.value) for x in enum_bits])
    return out


def make_enum(name, *args):
    """makes enum from list of strings"""
    return enum.Enum(name, [(x, x) for x in args])


# code words as variables
# SuperLocus
CODING_GENE = 'coding_gene'
NON_CODING_GENE = 'non_coding_gene'
PSEUDOGENE = 'pseudogene'
OPERON = 'operon'

# SuperLocusHistorical
GENE = 'gene'

# TranscriptLevelNice
MRNA = 'mRNA'
TRNA = 'tRNA'
RRNA = 'rRNA'
MIRNA = 'miRNA'
SNORNA = 'snoRNA'
SNRNA = 'snRNA'
SRP_RNA = 'SRP_RNA'
LNC_RNA = 'lnc_RNA'
PRE_MIRNA = 'pre_miRNA'
RNASE_MRP_RNA = 'RNase_MRP_RNA'

# TranscriptLevelInput
TRANSCRIPT = 'transcript'
PRIMARY_TRANSCRIPT = 'primary_transcript'
PSEUDOGENIC_TRANSCRIPT = 'pseudogenic_transcript'

# make all the enums
SuperLocus = make_enum('SuperLocus', CODING_GENE, NON_CODING_GENE, PSEUDOGENE, OPERON)
SuperLocusHistorical = make_enum('SuperLocusHistorical', GENE)
SuperLocusAll = join_to_enum('SuperLocusAll', SuperLocus, SuperLocusHistorical)


TranscriptLevelNice = make_enum('TranscriptLevelNice', MRNA, TRNA, RRNA,MIRNA, SNORNA, SNRNA, SRP_RNA,
                                LNC_RNA, PRE_MIRNA, RNASE_MRP_RNA)
TranscriptLevelInput = make_enum('TranscriptLevelInput', TRANSCRIPT, PRIMARY_TRANSCRIPT, PSEUDOGENIC_TRANSCRIPT)
TranscriptLevelAll = join_to_enum('TranscriptLevel', TranscriptLevelNice, TranscriptLevelInput)


# FEATURES
# TranscribedGeneral
TRANSCRIPTION_START_SITE = 'TSS'  # transcription_start_site
TRANSCRIPTION_TERMINATION_SITE = 'TTS'  # transcription_termination_site
DONOR_SPLICE_SITE = 'DSS'  # donor_splice_site
ACCEPTOR_SPLICE_SITE = 'ASS'  # acceptor_splice_site
# TranscribedInput
EXON = 'exon'
INTRON = 'intron'
FIVE_PRIME_UTR = 'five_prime_UTR'
THREE_PRIME_UTR = 'three_prime_UTR'
# TranscribedStatus
IN_RAW_TRANSCRIPT = 'in_raw_transcript'
IN_INTRON = 'in_intron'
IN_TRANS_INTRON = 'in_trans_intron'
# TranscribedTransSplice
DONOR_TRANS_SPLICE_SITE = 'donor_trans_splice_site'
ACCEPTOR_TRANS_SPLICE_SITE = 'acceptor_trans_splice_site'

# transcription related features
TranscribedGeneral = make_enum('TranscribedGeneral', TRANSCRIPTION_START_SITE, TRANSCRIPTION_TERMINATION_SITE,
                               DONOR_SPLICE_SITE, ACCEPTOR_SPLICE_SITE)
TranscribedInput = make_enum('TranscribedInput', EXON, INTRON, FIVE_PRIME_UTR, THREE_PRIME_UTR)
TranscribedStatus = make_enum('TranscribedStatus', IN_RAW_TRANSCRIPT, IN_INTRON, IN_TRANS_INTRON)
TranscribedTransSplice = make_enum('TranscribedTransSplice', DONOR_TRANS_SPLICE_SITE, ACCEPTOR_TRANS_SPLICE_SITE)
# joining transcription related
TranscribedAll = join_to_enum('TranscribedFeature', TranscribedGeneral, TranscribedStatus, TranscribedTransSplice,
                              TranscribedInput)
TranscribedNice = join_to_enum('TranscribedNice', TranscribedGeneral, TranscribedStatus, TranscribedTransSplice)


# translation related features
# TranslatedInput
CDS = 'CDS'
# TranslatedGeneral
START_CODON = 'start_codon'
STOP_CODON = 'stop_codon'
# TranslatedStatus
IN_TRANSLATED_REGION = 'in_translated_region'  # so between start and stop codons, could still be in an intron

# Enums
TranslatedInput = make_enum('TranslatedInput', CDS)
TranslatedGeneral = make_enum('TranslatedGeneral', START_CODON, STOP_CODON)
TranslatedStatus = make_enum('TranslatedStatus', IN_TRANSLATED_REGION)

TranslatedAll = join_to_enum('TranslatedFeatureType', TranslatedInput, TranslatedGeneral, TranslatedStatus)
TranslatedNice = join_to_enum('TranslatedNice', TranslatedStatus, TranslatedGeneral)

# Anything else
# errror features
IN_ERROR = 'in_error'
ERROR_OPEN = 'error_open'
ERROR_CLOSE = 'error_close'

REGION = 'region'
CHROMOSOME = 'chromosome'
SUPERCONTIG = 'supercontig'
MATCH = 'match'
CDNA_MATCH = 'cDNA_match'

ErrorFeature = make_enum('ErrorFeature', IN_ERROR, ERROR_OPEN, ERROR_CLOSE)

# ignorable known
IgnorableFeatures = make_enum('IgnorableFeatures', REGION, CHROMOSOME, SUPERCONTIG, MATCH, CDNA_MATCH)

# All known features (else error on import)
OnSequence = join_to_enum('OnSequence', TranscribedAll, TranslatedAll, ErrorFeature)
AllKnown = join_to_enum('AllKnown', SuperLocusAll, TranscriptLevelAll, TranslatedAll, TranscribedAll, ErrorFeature,
                        IgnorableFeatures)
AllKeepable = join_to_enum('AllKeepable', SuperLocusAll, TranscriptLevelNice, TranslatedNice, TranscribedNice,
                           ErrorFeature)
