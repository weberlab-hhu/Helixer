from structure import GenericData
from dustdas import gffhelper
import logging
import copy


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


class IDMaker(object):
    def __init__(self, prefix='', width=6):
        self._counter = 0
        self.prefix = prefix
        self._seen = set()
        self._width = width

    @property
    def seen(self):
        return self._seen

    def next_unique_id(self, suggestion=None):
        if suggestion is not None:
            suggestion = str(suggestion)
            if suggestion not in self._seen:
                self._seen.add(suggestion)
                return suggestion
        # you should only get here if a) there was no suggestion or b) it was not unique
        return self._new_id()

    def _new_id(self):
        new_id = self._fmt_id()
        self._seen.add(new_id)
        self._counter += 1
        return new_id

    def _fmt_id(self):
        to_format = '{}{:0' + str(self._width) + '}'
        return to_format.format(self.prefix, self._counter)


class AnnotatedGenome(GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('super_loci', True, SuperLoci, list),
                      ('meta_info', True, MetaInfoAnnoGenome, None),
                      ('gffkey', False, FeatureDecoder, None),
                      ('transcript_ider', False, IDMaker, None),
                      ('feature_ider', False, IDMaker, None)]

        self.super_loci = []
        self.meta_info = MetaInfoAnnoGenome()
        self.gffkey = FeatureDecoder()
        self.transcript_ider = IDMaker(prefix='trx')
        self.feature_ider = IDMaker(prefix='ftr')

    def add_gff(self, gff_file):
        for entry_group in self.group_gff_by_gene(gff_file):
            new_sl = SuperLoci(self)
            new_sl.add_gff_entry_group(entry_group)
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
    def __init__(self, genome):
        super().__init__()
        self.spec += [('transcripts', True, Transcribed, dict),
                      ('features', True, StructuredFeature, dict),
                      ('ids', True, list, None),
                      ('genome', False, AnnotatedGenome, None),
                      ('_dummy_transcript', False, Transcribed, None)]
        self.transcripts = {}
        self.features = {}
        self.ids = []
        self._dummy_transcript = None
        self.genome = genome

    def dummy_transcript(self):
        if self._dummy_transcript is not None:
            return self._dummy_transcript
        else:
            # setup new blank transcript
            transcript = Transcribed()
            transcript.id = self.genome.transcript_ider.next_unique_id()  # add an id
            self._dummy_transcript = transcript  # save to be returned by next call of dummy_transcript
            self.transcripts[transcript.id] = transcript  # save into main dict of transcripts
            return transcript

    def add_gff_entry(self, entry):
        gffkey = self.genome.gffkey
        if entry.type == gffkey.gene:
            self.type = gffkey.gene
            gene_id = entry.get_ID()
            self.id = gene_id
            self.ids.append(gene_id)
        elif entry.type in gffkey.transcribed:
            parent = self.one_parent(entry)
            assert parent == self.id, "not True :( [{} == {}]".format(parent, self.id)
            transcript = Transcribed()
            transcript.add_data(self, entry)
            self.transcripts[transcript.id] = transcript
        elif entry.type in gffkey.on_sequence:
            feature = StructuredFeature()
            feature.add_data(self, entry)
            self.features[feature.id] = feature

    def add_gff_entry_group(self, entries):
        entries = list(entries)
        for entry in entries:
            self.add_gff_entry(entry)
        self.check_and_fix_structure(entries)

    @staticmethod
    def one_parent(entry):
        parents = entry.get_Parent()
        assert len(parents) == 1
        return parents[0]

    def get_matching_transcript(self, entry):
        # deprecating
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

    def _mark_erroneous(self, entry):
        assert entry.type in self.genome.gffkey.gene_level
        feature = StructuredFeature()
        feature.start = entry.start
        feature.end = entry.end
        feature.type = self.genome.gffkey.error
        feature.id = self.genome.feature_ider.next_unique_id()
        logging.warning(
            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, No valid features found - marking erroneous'.format(
                src=entry.source, species=self.genome.meta_info.species, seqid=entry.seqid, start=entry.start,
                end=entry.end, gene_id=self.id
            ))
        self.features[feature.id] = feature

    def check_and_fix_structure(self, entries):
        # if it's empty (no bottom level features at all) mark as erroneous
        if not self.features:
            self._mark_erroneous(entries[0])

        # collapse identical final features
        self.collapse_identical_features()
        # check that all non-exons are in regions covered by an exon

        # recreate transcribed / exon as necessary, but with reconstructed flag (also check for and mark pseudogenes)
        pass  # todo

    def collapse_identical_features(self):
        i = 0
        features = self.features
        while i < len(features) - 1:
            # sort and copy keys so that removal of the merged from the dict causes neither sorting nor looping trouble
            feature_keys = sorted(features.keys())
            feature = features[feature_keys[i]]
            for j in range(i + 1, len(feature_keys)):
                o_key = feature_keys[j]
                if feature.fully_overlaps(features[o_key]):
                    feature.merge(features[o_key])  # todo logging debug
                    features.pop(o_key)
                    logging.warning('removing {} from {} as it overlaps {}'.format(o_key, self.id, feature.id))
            i += 1

    def implicit_to_explicit(self):
        # make introns, tss, tts, and maybe start/stop codons
        # add UTR if they are not there
        # check start stop codons and splice sites against sequence and flag errors
        pass

    def check_sequence_assumptions(self):
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
                      ('features', True, list, None)]

        self.super_loci = None
        self.features = []
        self.feature_objects = []

    def add_data(self, super_loci, gff_entry):
        self.super_loci = super_loci
        self.id = gff_entry.get_ID()
        self.type = gff_entry.type

    def link_to_feature(self, feature_id):
        self.features.append(feature_id)

    def remove_feature(self, feature_id):
        self.features.pop(self.features.index(feature_id))


class StructuredFeature(FeatureLike):
    def __init__(self):
        super().__init__()
        self.spec += [('start', True, int, None),
                      ('end', True, int, None),
                      ('seqid', True, str, None),
                      ('strand', True, str, None),
                      ('score', True, float, None),
                      ('source', True, str, None),
                      ('frame', True, str, None),
                      ('transcripts', False, list, None),
                      ('super_loci', False, SuperLoci, None)]

        self.start = -1
        self.end = -1
        self.seqid = ''
        self.strand = '.'
        self.frame = '.'
        self.score = -1.
        self.source = ''
        self.transcripts = []
        self.super_loci = None

    def add_data(self, super_loci, gff_entry):
        gffkey = super_loci.genome.gffkey
        fid = gff_entry.get_ID()
        self.id = super_loci.genome.feature_ider.next_unique_id(fid)
        self.type = gff_entry.type
        self.start = gff_entry.start
        self.end = gff_entry.end
        self.strand = gff_entry.strand
        self.seqid = gff_entry.seqid
        try:
            self.score = float(gff_entry.score)
        except ValueError:
            pass
        self.super_loci = super_loci
        new_transcripts = gff_entry.get_Parent()
        if not new_transcripts:
            self.type = gffkey.error
            logging.warning('{species}:{seqid}:{fid}:{new_id} - No Parents listed'.format(
                species=super_loci.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
            ))
        for transcript_id in new_transcripts:
            new_t_id = transcript_id
            try:
                transcript = super_loci.transcripts[transcript_id]
                transcript.link_to_feature(self.id)
            except KeyError:
                if transcript_id == super_loci.id:
                    # if we just skipped the transcript, and linked to gene, use dummy transcript in between
                    transcript = super_loci.dummy_transcript()
                    logging.info(
                        '{species}:{seqid}:{fid}:{new_id} - Parent gene instead of transcript, recreating'.format(
                            species=super_loci.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
                        ))
                    transcript.link_to_feature(self.id)
                    new_t_id = transcript.id
                else:
                    self.type = gffkey.error
                    new_t_id = None
                    logging.warning(
                        '{species}:{seqid}:{fid}:{new_id} - Parent: "{parent}" not found at loci'.format(
                            species=super_loci.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id,
                            parent=transcript_id
                        ))
            self.link_to_transcript_and_back(new_t_id)

    def link_to_transcript_and_back(self, transcript_id):
        transcript = self.super_loci.transcripts[transcript_id]  # get transcript
        transcript.link_to_feature(self.id)  # link to and from self
        self.transcripts.append(transcript_id)

    def fully_overlaps(self, other):
        should_match = ['type', 'start', 'end', 'seqid', 'strand', 'frame']
        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
        same_gene = self.super_loci is other.super_loci
        out = False
        if all(does_it_match + [same_gene]):
            out = True
        return out

    def merge(self, other):
        assert self is not other
        # move transcript reference from other to self
        for transcript_id in copy.deepcopy(other.transcripts):
            self.link_to_transcript_and_back(transcript_id)
            other.de_link_from_transcript(transcript_id)

    def de_link_from_transcript(self, transcript_id):
        transcript = self.super_loci.transcripts[transcript_id]  # get transcript
        transcript.remove_feature(self.id)  # drop other
        self.transcripts.pop(self.transcripts.index(transcript_id))
