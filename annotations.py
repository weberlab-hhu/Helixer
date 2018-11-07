from structure import GenericData
from dustdas import gffhelper


class FeatureDecoder(object):
    def __init__(self):
        # gene like, generally having a collection of transcripts
        self.gene = 'gene'
        self.super_gene = 'super_gene'
        self.ncRNA_gene = 'ncRNA_gene'
        self.pseudogene = 'pseudogene'
        self.gene_level = [self.gene, self.super_gene, self.ncRNA_gene, self.pseudogene]

        # transcript like, generally having a collection of exons, indicating how they are spliced
        # also ultimately, if not explicitly having a transcription start and termination site
        self.mRNA = 'mRNA'
        self.tRNA = 'tRNA'
        self.rRNA = 'rRNA'
        self.miRNA = 'miRNA'
        self.snoRNA = 'snoRNA'
        self.snRNA = 'snRNA'
        self.SRP_RNA = 'SRP_RNA'
        self.lnc_RNA = 'lnc_RNA'
        self.pre_miRNA = 'pre_miRNA'
        self.RNase_MRP_RNA = 'RNase_MRP_RNA'
        self.transcript = 'transcript'
        self.primary_transcript = 'primary_transcript'
        self.pseudogenic_transcript = 'pseudogenic_transcript'  # which may or may not be transcribed, hard to say
        self.transcribed = [self.mRNA, self.transcript, self.tRNA, self.primary_transcript, self.rRNA, self.miRNA,
                            self.snoRNA, self.snRNA, self.SRP_RNA, self.lnc_RNA, self.pre_miRNA, self.RNase_MRP_RNA,
                            self.pseudogenic_transcript]

        # regions of original (both) or processed (exon) transcripts
        self.exon = 'exon'
        self.intron = 'intron'
        self.sub_transcribed = [self.exon, self.intron]

        # sub-exon-level categorization (but should have transcribed as parent)
        self.cds = 'CDS'
        self.five_prime_UTR = 'five_prime_UTR'
        self.three_prime_UTR = 'three_prime_UTR'
        self.coding_info = [self.cds, self.five_prime_UTR, self.three_prime_UTR]

        # point annotations
        self.TSS = 'TSS'  # transcription start site
        self.TTS = 'TTS'  # transcription termination site
        self.start_codon = 'start_codon'
        self.stop_codon = 'stop_codon'
        self.point_annotations = [self.TSS, self.TTS, self.start_codon, self.stop_codon]

        # regions (often but not always included so one knows the size of the chromosomes / contigs / whatever
        self.region = 'region'
        self.chormosome = 'chromosome'
        self.supercontig = 'supercontig'
        self.regions = [self.region, self.chormosome, self.supercontig]

        # things that don't appear to really be annotations
        self.match = 'match'
        self.cDNA_match = 'cDNA_match'
        self.ignorable = [self.match, self.cDNA_match]

        # to be used when there are just obvious mistakes, like non-ATG start codon
        self.error = 'error'

        # and putting them together
        self.on_sequence = self.sub_transcribed + self.coding_info + self.point_annotations
        self.known = self.gene_level + self.transcribed + self.sub_transcribed + self.coding_info + \
                     self.point_annotations + self.regions + self.ignorable + [self.error]


class AnnotatedGenome(GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('super_loci', True, SuperLoci, list),
                      ('meta_info', True, MetaInfoAnnoGenome, None),
                      ('gffkey', False, FeatureDecoder, None)]

        self.super_loci = []
        self.meta_info = MetaInfoAnnoGenome()
        self.gffkey = FeatureDecoder()

    def add_gff(self, gff_file):

        for entry_group in self.group_gff_by_gene(gff_file):
            new_sl = SuperLoci()
            new_sl.add_gff_entry_group(entry_group, self.gffkey)
            self.super_loci.append(new_sl)
            if not new_sl.transcripts and not new_sl.features:
                print('{} from {} with {} transcripts and {} features'.format(new_sl.id,
                                                                              entry_group[0].source,
                                                                              len(new_sl.transcripts),
                                                                              len(new_sl.features)))

    def useful_gff_entries(self, gff_file):
        skipable = self.gffkey.regions + self.gffkey.ignorable
        reader = gffhelper.read_gff_file(gff_file)
        for entry in reader:
            if entry.type not in self.gffkey.known:
                raise ValueError("unrecognized feature type from gff: {}".format(entry.type))
            if entry.type not in skipable:
                yield entry

    def group_gff_by_gene(self, gff_file):
        reader = self.useful_gff_entries(gff_file)
        gene_group = [next(reader)]
        for entry in reader:
            if entry.type == 'gene':
                yield gene_group
                gene_group = [entry]
            else:
                gene_group.append(entry)
        yield gene_group


class MetaInfoAnnotation(GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('number_genes', True, int, None),
                      ('bp_intergenic', True, int, None),
                      ('bp_coding', True, int, None),
                      ('bp_intronic', True, int, None),
                      ('bp_3pUTR', True, int, None),
                      ('bp_5pUTR', True, int, None)]

        # todo, does this make sense, considering that any given bp could belong to multiple of the following
        self.number_genes = 0
        self.bp_intergenic = 0
        self.bp_coding = 0
        self.bp_intronic = 0
        self.bp_3pUTR = 0
        self.bp_5pUTR = 0


class MetaInfoAnnoGenome(MetaInfoAnnotation):
    def __init__(self):
        super().__init__()
        self.spec += [('species', True, str, None),
                      ('accession', True, str, None),
                      ('version', True, str, None),
                      ('acquired_from', True, str, None)]

        self.species = ""
        self.accession = ""
        self.version = ""
        self.acquired_from = ""


class MetaInfoAnnoSequence(MetaInfoAnnotation):
    def __init__(self):
        super().__init__()
        self.spec += [('seqid', True, str, None)]
        self.seqid = ""


class FeatureLike(GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('id', True, str, None),
                      ('type', True, str, None),
                      ('is_partial', True, bool, None),
                      ('is_reconstructed', True, bool, None),
                      ('is_type_in_question', True, bool, None)]
        self.id = ''
        self.type = ''
        self.is_partial = False
        self.is_reconstructed = False
        self.is_type_in_question = False


class SuperLoci(FeatureLike):
    # normally a loci, some times a short list of loci for "trans splicing"
    # this will define a group of exons that can possibly be made into transcripts
    # AKA this if you have to go searching through a graph for parents/children, at least said graph will have
    # a max size defined at SuperLoci
    def __init__(self):
        super().__init__()
        self.spec += [('transcripts', True, Transcribed, list),
                      ('features', True, StructuredFeature, list),
                      ('ids', True, list, None)]

        self.transcripts = []
        self.features = []
        self.ids = []

    def add_gff_entry(self, entry, gffkey):
        if entry.type == gffkey.gene:
            self.type = gffkey.gene
            gene_id = entry.get_ID()
            self.id = gene_id
            self.ids.append(gene_id)
        elif entry.type in gffkey.transcribed:
            parent = self.one_parent(entry)
            assert parent == self.id, "not True :( -- {} == {}".format(parent, self.id)
            transcript = Transcribed()
            transcript.add_data(self, entry)
            self.transcripts.append(transcript)
        elif entry.type in gffkey.on_sequence:
            try:
                transcript = self.get_matching_transcript(entry)
            except NoTranscriptError:
                transcript = None  # todo, should rather send entries to some other bin to be sorted later/mark
            feature = StructuredFeature()
            feature.add_data(self, transcript, entry)
            self.features.append(feature)
            # todo, checkfor and collapse identical exons / features

    def add_gff_entry_group(self, entries, gffkey):
        for entry in entries:
            self.add_gff_entry(entry, gffkey)
        self.check_and_fix_structure()

    @staticmethod
    def one_parent(entry):
        parents = entry.get_Parent()
        assert len(parents) == 1
        return parents[0]

    def get_matching_transcript(self, entry):
        parent = self.one_parent(entry)
        try:
            transcript = self.transcripts[-1]
            assert parent == transcript.id
        except IndexError:
            raise NoTranscriptError("0 transcripts found")
        except AssertionError:
            transcripts = [x for x in self.transcripts if x.id == parent]
            if len(transcripts) == 1:
                transcript = transcripts[0]
            else:
                raise NoTranscriptError("can't find {} in {}".format(parent, [x.id for x in self.transcripts]))
        return transcript

    def check_and_fix_structure(self):
        # collapse identical final features

        # check that all features have a 'transcribed' parent

        # check that all non-exons are in regions covered by an exon

        # if not check that they have the 'gene' as parent

        # recreate transcribed / exon as necessary, but with reconstructed flag (also check for and mark pseudogenes)
        pass  # todo

    def implicit_to_explicit(self):
        # make introns, tss, tts, and maybe start/stop codons
        # add UTR if they are not there
        # check start stop codons and splice sites against sequence and flag errors
        pass

    def add_to_interval_tree(self, itree):
        pass  # todo, make sure at least all features are loaded to interval tree

    def load_jsonable(self, jsonable):
        super().load_jsonable(jsonable)
        # todo restore super_loci objects, transcript objects, feature_objects to self and children


class NoTranscriptError(Exception):
    pass


class Transcribed(FeatureLike):
    def __init__(self):
        super().__init__()
        self.spec += [('super_loci', False, SuperLoci, None),
                      ('features', True, list, None),
                      ('feature_objects', False, StructuredFeature, list)]

        self.super_loci = None
        self.features = []
        self.feature_objects = []

    def add_data(self, super_loci, gff_entry):
        self.super_loci = super_loci
        self.id = gff_entry.get_ID()
        self.type = gff_entry.type

    def make_feature_objects(self):
        pass  # todo link to features and link back


class StructuredFeature(FeatureLike):
    # todo, this will probably hold the graph (parent/child relations)
    # todo, also basics like complete/incomplete/
    def __init__(self):
        super().__init__()
        self.spec += [('start', True, int, None),
                      ('end', True, int, None),
                      ('seqid', True, str, None),
                      ('strand', True, str, None),
                      ('score', True, float, None),
                      ('source', True, str, None),
                      ('transcripts', False, Transcribed, list),
                      ('super_loci', False, SuperLoci, None)]

        self.start = -1
        self.end = -1
        self.seqid = ''
        self.strand = '.'
        self.score = -1.
        self.source = ''
        self.transcripts = []
        self.super_loci = None

    def add_data(self, super_loci, transcript, gff_entry):
        self.id = gff_entry.get_ID()
        self.type = gff_entry.type
        self.start = gff_entry.start
        self.end = gff_entry.end
        self.strand = gff_entry.strand
        self.seqid = gff_entry.strand
        try:
            self.score = float(gff_entry.score)
        except ValueError:
            pass
        self.super_loci = super_loci
        self.transcripts.append(transcript)

    def link_back(self, transcript):
        self.transcripts.append(transcript)
