from structure import GenericData
from dustdas import gffhelper


class FeatureDecoder(object):
    def __init__(self):
        self.gene = 'gene'
        self.super_gene = 'super_gene'
        self.mRNA = 'mRNA'
        self.transcript = 'transcript'
        self.transcribed = [self.mRNA, self.transcript]
        self.cds = 'CDS'
        self.exon = 'exon'
        self.intron = 'intron'
        self.coding_related = [self.gene, self.mRNA, self.exon, self.cds]


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

        new_sl = SuperLoci()  # todo better fix than blank at start
        for entry in gffhelper.read_gff_file(gff_file):
            if entry.type == 'gene':
                self.super_loci.append(new_sl)
                print('saving {} with {} transcripts and {} features'.format(new_sl.id, len(new_sl.transcripts),
                                                                             len(new_sl.features)))
                new_sl = SuperLoci()
                new_sl.add_gff_entry(entry, self.gffkey)
                print(new_sl.id)
            else:
                print(new_sl.id)
                new_sl.add_gff_entry(entry, self.gffkey)
        self.super_loci.append(new_sl)


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
                      ('is_partial', True, bool, None)]
        self.id = ''
        self.type = ''
        self.is_partial = None


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
        if entry.type not in gffkey.coding_related:
            pass
        elif entry.type == gffkey.gene:
            self.type = gffkey.gene
            gene_id = entry.get_ID()
            print('set id to {}'.format(gene_id))
            self.id = gene_id
            self.ids.append(gene_id)
        elif entry.type in gffkey.transcribed:
            print('transcript')
            parent = self.one_parent(entry)
            assert parent == self.id, "not True :( -- {} == {}".format(parent, self.id)
            transcript = Transcribed()
            transcript.add_data(self, entry)
            self.transcripts.append(transcript)
            print(self.transcripts)
        elif entry.type == gffkey.exon:
            print('exon')
            try:
                transcript = self.get_matching_transcript(entry)
            except NoTranscriptError:
                transcript = None  # todo, should rather send entries to some other bin to be sorted later/mark
            feature = StructuredFeature()
            feature.add_data(self, transcript, entry)
            self.features.append(feature)
            # todo, checkfor and collapse identical exons / features

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