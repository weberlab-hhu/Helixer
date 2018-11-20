from structure import GenericData
from dustdas import gffhelper
import logging
import copy
import intervaltree
import helpers


class FeatureDecoder(object):
    def __init__(self):
        self.error_buffer = 2000
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
        self.donor_splice_site = 'donor_splice_site'
        self.acceptor_splice_site = 'acceptor_splice_site'
        # use the following when the far side of a splice site is on a different strand and/or sequence
        # does not imply an intron!
        # e.g. for trans-splicing
        self.trans_donor_splice_site = 'trans_donor_splice_site'
        self.trans_acceptor_splice_site = 'trans_acceptor_splice_site'
        self.point_annotations = [self.TSS, self.TTS, self.start_codon, self.stop_codon, self.donor_splice_site,
                                  self.acceptor_splice_site, self.trans_donor_splice_site,
                                  self.trans_acceptor_splice_site]

        # regions (often but not always included so one knows the size of the chromosomes / contigs / whatever
        self.region = 'region'
        self.chromosome = 'chromosome'
        self.supercontig = 'supercontig'
        self.regions = [self.region, self.chromosome, self.supercontig]

        # things that don't appear to really be annotations
        self.match = 'match'
        self.cDNA_match = 'cDNA_match'
        self.ignorable = [self.match, self.cDNA_match]

        # for mistakes or near-mistakes / marking partials
        self.error = 'error'
        self.status_coding = 'status_coding'
        self.status_intron = 'status_intron'
        self.status_five_prime_UTR = 'status_five_prime_UTR'
        self.status_three_prime_UTR = 'status_three_prime_UTR'
        self.status_intergenic = 'status_intergenic'
        self.statuses = [self.status_coding, self.status_intron, self.status_five_prime_UTR,
                         self.status_three_prime_UTR, self.status_intergenic]
        # and putting them together
        self.on_sequence = self.sub_transcribed + self.coding_info + self.point_annotations
        self.known = self.gene_level + self.transcribed + self.sub_transcribed + self.coding_info + \
                     self.point_annotations + self.regions + self.ignorable + [self.error] + self.statuses


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
                      ('meta_info_sequences', True, MetaInfoAnnoSequence, dict),
                      ('gffkey', False, FeatureDecoder, None),
                      ('transcript_ider', False, IDMaker, None),
                      ('feature_ider', False, IDMaker, None),
                      ('mapper', False, helpers.Mapper, None)]

        self.super_loci = []
        self.meta_info = MetaInfoAnnoGenome()
        self.meta_info_sequences = {}
        self.gffkey = FeatureDecoder()
        self.transcript_ider = IDMaker(prefix='trx')
        self.feature_ider = IDMaker(prefix='ftr')
        self.mapper = helpers.Mapper()

    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
        err_handle = open(err_file, 'w')
        for seq in genome.sequences:
            mi = MetaInfoAnnoSequence()
            mi.seqid = seq.meta_info.seqid
            mi.total_bp = seq.meta_info.total_bp
            self.meta_info_sequences[mi.seqid] = mi

        gff_seq_ids = helpers.get_seqids_from_gff(gff_file)
        mapper, is_forward = helpers.two_way_key_match(self.meta_info_sequences.keys(), gff_seq_ids)
        self.mapper = mapper

        if not is_forward:
            raise NotImplementedError("Still need to implement backward match if fasta IDs are subset of gff IDs")

        for entry_group in self.group_gff_by_gene(gff_file):
            new_sl = SuperLoci(self)
            new_sl.add_gff_entry_group(entry_group, err_handle)

            self.super_loci.append(new_sl)
            if not new_sl.transcripts and not new_sl.features:
                print('{} from {} with {} transcripts and {} features'.format(new_sl.id,
                                                                              entry_group[0].source,
                                                                              len(new_sl.transcripts),
                                                                              len(new_sl.features)))
        err_handle.close()

    def useful_gff_entries(self, gff_file):
        skipable = self.gffkey.regions + self.gffkey.ignorable
        reader = gffhelper.read_gff_file(gff_file)
        for entry in reader:
            if entry.type not in self.gffkey.known:
                raise ValueError("unrecognized feature type fr:qom gff: {}".format(entry.type))
            if entry.type not in skipable:
                yield entry

    def group_gff_by_gene(self, gff_file):
        reader = self.useful_gff_entries(gff_file)
        gene_group = [next(reader)]
        for entry in reader:
            if entry.type in self.gffkey.gene_level:
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
        self.spec += [('seqid', True, str, None),
                      ('total_bp', True, int, None)]
        self.seqid = ""
        self.total_bp = 0


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
        # self.is_partial = False
        # self.is_reconstructed = False
        # self.is_type_in_question = False


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
            transcript.super_loci = self
            self._dummy_transcript = transcript  # save to be returned by next call of dummy_transcript
            self.transcripts[transcript.id] = transcript  # save into main dict of transcripts
            return transcript

    def add_gff_entry(self, entry):
        exceptions = entry.attrib_filter(tag="exception")
        for exception in [x.value for x in exceptions]:
            if 'trans-splicing' in exception:
                raise TransSplicingError('trans-splice in attribute {} {}'.format(entry.get_ID(), entry.attribute))
        gffkey = self.genome.gffkey
        if entry.type in gffkey.gene_level:
            self.type = entry.type
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

    def _add_gff_entry_group(self, entries):
        entries = list(entries)
        for entry in entries:
            self.add_gff_entry(entry)
        self.check_and_fix_structure(entries)

    def add_gff_entry_group(self, entries, ts_err_handle):
        try:
            self._add_gff_entry_group(entries)
        except TransSplicingError as e:
            self._mark_erroneous(entries[0])
            logging.warning('skipping but noting trans-splicing: {}'.format(str(e)))
            ts_err_handle.writelines([x.to_json() for x in entries])
            # todo, log to file

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
        logging.warning(
            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, No valid features found - marking erroneous'.format(
                src=entry.source, species=self.genome.meta_info.species, seqid=entry.seqid, start=entry.start,
                end=entry.end, gene_id=self.id
            ))
        sf = StructuredFeature()
        feature = sf.add_erroneous_data(self, entry)
        self.features[feature.id] = feature

    def check_and_fix_structure(self, entries):
        # if it's empty (no bottom level features at all) mark as erroneous
        if not self.features:
            self._mark_erroneous(entries[0])

        # collapse identical final features
        #self.collapse_identical_features()  # todo, can I deprecate?
        # check that all non-exons are in regions covered by an exon
        #self.maybe_reconstruct_exons()  # todo, can I deprecate?
        # recreate transcribed / exon as necessary
        # todo, but with reconstructed flag (also check for and mark pseudogenes)
        to_remove = []
        for transcript in self.transcripts.values():
            old_features = copy.deepcopy(transcript.features)
            t_interpreter = TranscriptInterpreter(transcript)
            t_interpreter.decode_raw_features()

            self.add_features(t_interpreter.clean_features, transcript=None)  # no transcript, as they're already linked
            transcript.delink_features(old_features)
            to_remove += old_features
        self.remove_features(to_remove)

    def add_features(self, features, transcript=None):
        for feature in features:
            self.features[feature.id] = feature
            if transcript is not None:
                feature.link_to_transcript_and_back(transcript.id)

    def maybe_reconstruct_exons(self):
        """creates any exons necessary, so that all CDS/UTR is contained within an exon"""
        # because introns will be determined from exons, every CDS etc, has to have an exon
        new_exons = []
        exons = self.exons()
        coding_info = self.coding_info_features()
        for f in coding_info:
            if not any([f.is_contained_in(exon) for exon in exons]):
                new_exons.append(f.reconstruct_exon())  # todo, logging info/debug?
        for e in new_exons:
            self.features[e.id] = e

    def remove_features(self, to_remove):
        for f_key in to_remove:
            self.features.pop(f_key)

    def exons(self):
        return [self.features[x] for x in self.features if self.features[x].type == self.genome.gffkey.exon]

    def coding_info_features(self):
        return [self.features[x] for x in self.features if self.features[x].type in self.genome.gffkey.coding_info]

    def implicit_to_explicit(self):
        # make introns, tss, tts, and maybe start/stop codons, utr if necessary
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


class TransSplicingError(Exception):
    pass


class Transcribed(FeatureLike):
    def __init__(self):
        super().__init__()
        self.spec += [('super_loci', False, SuperLoci, None),
                      ('features', True, list, None)]

        self.super_loci = None
        self.features = []

    def add_data(self, super_loci, gff_entry):
        self.super_loci = super_loci
        self.id = gff_entry.get_ID()
        self.type = gff_entry.type

    def link_to_feature(self, feature_id):
        assert feature_id not in self.features, "{} already in features {} for {} {} in loci {}".format(
            feature_id, self.features, self.type, self.id, self.super_loci.id)
        self.features.append(feature_id)

    def remove_feature(self, feature_id):
        self.features.pop(self.features.index(feature_id))

    def short_str(self):
        return '{}. --> {}'.format(self.id, self.features)

    def delink_features(self, features):
        for feature in features:
            try:
                self.super_loci.features[feature].de_link_from_transcript(self.id)
            except ValueError:
                raise ValueError('{} not in {}'.format(self.id,
                                                       self.super_loci.features[feature].transcripts))


class StructuredFeature(FeatureLike):
    def __init__(self):
        super().__init__()
        self.spec += [('start', True, int, None),
                      ('end', True, int, None),
                      ('seqid', True, str, None),
                      ('strand', True, str, None),
                      ('score', True, float, None),
                      ('source', True, str, None),
                      ('phase', True, int, None),
                      ('transcripts', True, list, None),
                      ('super_loci', False, SuperLoci, None)]

        self.start = -1
        self.end = -1
        self.seqid = ''
        self.strand = '.'
        self.phase = None
        self.score = None
        self.source = ''
        self.transcripts = []
        self.super_loci = None

    @property
    def py_start(self):
        return self.start - 1

    @property
    def py_end(self):
        return self.end
# to do, make short / long print methods as part of object

    def short_str(self):
        return '{} is {}: {}-{} on {}. --> {}'.format(self.id, self.type, self.start, self.end, self.seqid,
                                                      self.transcripts)

    def add_data(self, super_loci, gff_entry):
        gffkey = super_loci.genome.gffkey
        try:
            fid = gff_entry.get_ID()
        except TypeError:
            fid = None
            logging.debug('no ID in attr {} in {}, making new unique ID'.format(gff_entry.attribute, super_loci.id))
        self.super_loci = super_loci
        self.id = super_loci.genome.feature_ider.next_unique_id(fid)
        self.type = gff_entry.type
        self.start = int(gff_entry.start)
        self.end = int(gff_entry.end)
        self.strand = gff_entry.strand
        self.seqid = self.super_loci.genome.mapper(gff_entry.seqid)
        if gff_entry.phase == '.':
            self.phase = None
        else:
            self.phase = int(gff_entry.phase)
        try:
            self.score = float(gff_entry.score)
        except ValueError:
            pass
        new_transcripts = gff_entry.get_Parent()
        if not new_transcripts:
            self.type = gffkey.error
            logging.warning('{species}:{seqid}:{fid}:{new_id} - No Parents listed'.format(
                species=super_loci.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
            ))
        for transcript_id in new_transcripts:
            new_t_id = transcript_id
            if new_t_id not in super_loci.transcripts:
                if transcript_id == super_loci.id:
                    # if we just skipped the transcript, and linked to gene, use dummy transcript in between
                    transcript = super_loci.dummy_transcript()
                    logging.info(
                        '{species}:{seqid}:{fid}:{new_id} - Parent gene instead of transcript, recreating'.format(
                            species=super_loci.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
                        ))
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

    def add_erroneous_data(self, super_loci, gff_entry):
        self.super_loci = super_loci
        feature_e = self.clone()
        feature_e.start = int(gff_entry.start)
        feature_e.end = int(gff_entry.end)
        feature_e.strand = gff_entry.strand
        feature_e.seqid = gff_entry.seqid
        feature_e.change_to_error()
        return feature_e

    def change_to_error(self):
        self.type = self.super_loci.genome.gffkey.error

    def link_to_transcript_and_back(self, transcript_id):
        assert transcript_id not in self.transcripts, "{} already in transcripts {}".format(transcript_id,
                                                                                            self.transcripts)
        transcript = self.super_loci.transcripts[transcript_id]  # get transcript
        transcript.link_to_feature(self.id)  # link to and from self
        self.transcripts.append(transcript_id)

    def fully_overlaps(self, other):
        should_match = ['type', 'start', 'end', 'seqid', 'strand', 'phase']
        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
        same_gene = self.super_loci is other.super_loci
        out = False
        if all(does_it_match + [same_gene]):
            out = True
        return out

    def is_contained_in(self, other):
        should_match = ['seqid', 'strand', 'phase']
        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
        same_gene = self.super_loci is other.super_loci
        coordinates_within = self.start >= other.start and self.end <= other.end
        return all(does_it_match + [coordinates_within, same_gene])

    def reconstruct_exon(self):
        """creates an exon exactly containing this feature"""
        exon = self.clone()
        exon.type = self.super_loci.genome.gffkey.exon
        return exon

    def clone(self, copy_transcripts=True):
        """makes valid, independent clone/copy of this feature"""
        new = StructuredFeature()
        copy_over = copy.deepcopy(list(new.__dict__.keys()))

        for to_skip in ['super_loci', 'id', 'transcripts']:
            copy_over.pop(copy_over.index(to_skip))

        # handle can't just be copied things
        new.super_loci = self.super_loci
        new.id = self.super_loci.genome.feature_ider.next_unique_id()
        if copy_transcripts:
            for transcript in self.transcripts:
                new.link_to_transcript_and_back(transcript)

        # copy the rest
        for item in copy_over:
            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
        return new

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

    def is_plus_strand(self):
        if self.strand == '+':
            return True
        elif self.strand == '-':
            return False
        else:
            raise ValueError('strand should be +- {}'.format(self.strand))

    def upstream(self):
        if self.is_plus_strand():
            return self.start
        else:
            return self.end

    def downstream(self):
        if self.is_plus_strand():
            return self.end
        else:
            return self.start

    # inclusive and from 1 coordinates
    def upstream_from_interval(self, interval):
        if self.is_plus_strand():
            return interval.begin + 1
        else:
            return interval.end

    def downstream_from_interval(self, interval):
        if self.is_plus_strand():
            return interval.end
        else:
            return interval.begin + 1


#### section TranscriptInterpreter, might end up in a separate file later
class TranscriptStatus(object):
    """can hold and manipulate all the info on current status of a transcript"""
    def __init__(self):
        # initializes to intergenic
        self.genic = False  # todo, have some thoughts about how trans-splicing will fit in
        self.in_intron = False
        self.seen_start = False
        self.seen_stop = False
        self.phase = None  # todo, proper tracking / handling

    def saw_tss(self):
        self.genic = True

    def saw_start(self, phase):
        self.genic = True
        self.seen_start = True
        self.phase = phase

    def saw_stop(self):
        self.seen_stop = True
        self.phase = None

    def saw_tts(self):
        self.genic = False

    def splice_open(self):
        self.in_intron = True

    def splice_close(self):
        self.in_intron = False

    def is_5p_utr(self):
        return self.genic and not any([self.in_intron, self.seen_start, self.seen_stop])

    def is_3p_utr(self):
        return all([self.genic, self.seen_stop, self.seen_start]) and not self.in_intron

    def is_coding(self):
        return self.genic and self.seen_start and not any([self.in_intron, self.seen_stop])

    def is_intronic(self):
        return self.in_intron and self.genic

    def is_intergenic(self):
        return not self.genic


class TranscriptInterpreter(object):
    """takes raw/from-gff transcript, and makes totally explicit"""
    def __init__(self, transcript):
        self.status = TranscriptStatus()
        self.super_loci = transcript.super_loci
        self.gffkey = transcript.super_loci.genome.gffkey
        self.transcript = transcript
        self.clean_features = []  # will hold all the 'fixed' features

    @staticmethod
    def new_feature(template, **kwargs):
        new = template.clone()
        for key in kwargs:
            new.__setattr__(key, kwargs[key])
        return new

    @staticmethod
    def pick_one_interval(interval_set, target_type=None):
        if target_type is None:
            return interval_set[0]
        else:
            return [x for x in interval_set if x.data.type == target_type][0]

    def is_plus_strand(self):
        features = [self.super_loci.features[f] for f in self.transcript.features]
        seqids = [x.seqid for x in features]
        if not all([x == seqids[0] for x in seqids]):
            raise TransSplicingError("non matching seqids {}, for {}".format(seqids, self.super_loci.id))
        if all([x.strand == '+' for x in features]):
            return True
        elif all([x.strand == '-' for x in features]):
            return False
        else:
            raise TransSplicingError("Mixed strands at {} with {}".format(self.super_loci.id,
                                                                          [(x.seqid, x.strand) for x in features]))

    def interpret_transition(self, ivals_before, ivals_after, plus_strand=True):
        sign = 1
        if not plus_strand:
            sign = -1
        before_types = self.possible_types(ivals_before)
        after_types = self.possible_types(ivals_after)
        # 5' UTR can hit either start codon or splice site
        if self.status.is_5p_utr():
            # start codon
            self.handle_from_5p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_coding():
            self.handle_from_coding(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_3p_utr():
            self.handle_from_3p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_intronic():
            self.handle_from_intron()
        elif self.status.is_intergenic():
            self.handle_from_intergenic()
        else:
            raise ValueError('unknown status {}'.format(self.status.__dict__))

    def handle_from_coding(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.cds in before_types
        # stop codon
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=False)
        # splice site
        elif self.gffkey.cds in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)

    def handle_from_intron(self):
        raise NotImplementedError  # todo later

    def handle_from_3p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.three_prime_UTR in before_types
        # the only thing we should encounter is a splice site
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after three prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_5p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.five_prime_UTR in before_types
        # start codon
        if self.gffkey.cds in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=True)
        # intron
        elif self.gffkey.five_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after five prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_intergenic(self):
        raise NotImplementedError  # todo later

    def is_gap(self, ivals_before, ivals_after, sign):
        """checks for a gap between intervals, and validates it's a positive one on strand of interest"""
        after0 = self.pick_one_interval(ivals_after)
        before0 = self.pick_one_interval(ivals_before)
        before_downstream = before0.data.downstream_from_interval(before0)
        after_upstream = after0.data.upstream_from_interval(after0)
        is_gap = before_downstream + 1 * sign != after_upstream
        if is_gap:
            # if there's a gap, confirm it's in the right direction
            gap_len = (after_upstream - (before_downstream + 1 * sign)) * sign
            assert gap_len > 0, "inverse gap between {} and {} at putative control codon seq {}, gene {}, " \
                                "features {} {}".format(
                before_downstream, after_upstream, after0.data.seqid, self.super_loci.id, before0.data.id,
                after0.data.id
            )
        return is_gap

    def handle_control_codon(self, ivals_before, ivals_after, sign, is_start=True):
        target_type = None
        if is_start:
            target_type = self.gffkey.cds

        after0 = self.pick_one_interval(ivals_after, target_type)
        before0 = self.pick_one_interval(ivals_before, None)
        # make sure there is no gap
        is_gap = self.is_gap(ivals_before, ivals_after, sign)

        if is_start:
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

            template = after0.data
            # it better be std phase if it's a start codon
            at = template.upstream_from_interval(after0)
            if template.phase == 0: # "non-0 phase @ {} in {}".format(template.id, template.super_loci.id)
                start, end = min_max(at, at + 2 * sign)
                start_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.start_codon)
                self.status.saw_start(phase=0)
                self.clean_features.append(start_codon)
            else:
                upstream_buffered = before0.data.upstream_from_interval(before0) - sign * self.gffkey.error_buffer
                err_start, err_end = min_max(at - 1 * sign, upstream_buffered)
                feature_e = self.new_feature(template=template, type=self.gffkey.error,
                                             start=err_start, end=err_end, phase=None)
                coding_status = self.new_feature(template=template, type=self.gffkey.status_coding, start=at, end=at)
                self.status.saw_start(template.phase)
                self.clean_features += [feature_e, coding_status]
        else:
            # todo, confirm phase for stop codon
            template = before0.data
            at = template.downstream_from_interval(before0)
            start, end = min_max(at, at - 2 * sign)
            stop_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.stop_codon)
            self.status.saw_stop()
            self.clean_features.append(stop_codon)
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

    def handle_splice(self, ivals_before, ivals_after, sign):
        target_type = None
        if self.status.is_coding():
            target_type = self.gffkey.cds

        before0 = self.pick_one_interval(ivals_before, target_type)
        after0 = self.pick_one_interval(ivals_after, target_type)
        donor_tmplt = before0.data
        acceptor_tmplt = after0.data
        donor_at = donor_tmplt.downstream_from_interval(before0) + (1 * sign)
        acceptor_at = acceptor_tmplt.upstream_from_interval(after0) - (1 * sign)
        # add splice sites if there's a gap
        between_splice_sites = (acceptor_at - donor_at) * sign
        min_intron_len = 3  # todo, maybe get something small but not entirely impossible?
        if between_splice_sites > min_intron_len - 1:  # -1 because the splice sites are _within_ the intron
            donor = self.new_feature(template=donor_tmplt, start=donor_at, end=donor_at, phase=None,
                                     type=self.gffkey.donor_splice_site)
            # todo, check position of DSS/ASS to be consistent with Augustus, hopefully
            acceptor = self.new_feature(template=acceptor_tmplt, start=acceptor_at, end=acceptor_at,
                                        type=self.gffkey.acceptor_splice_site)
            self.clean_features += [donor, acceptor]
        # do nothing if there is just no gap between exons for a techinical / reporting error
        elif between_splice_sites == -1:
            pass
        # everything else is invalid
        else:
            feature_e = before0.data.clone()
            all_coords = [before0.data.start, before0.data.end, after0.data.start, after0.data.end]
            feature_e.start = sorted(all_coords)[0]
            feature_e.end = sorted(all_coords)[-1]
            feature_e.type = self.gffkey.error
            self.clean_features.append(feature_e)

    def interpret_first_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.upstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.five_prime_UTR in possible_types:
            # this should indicate we're good to go and have a transcription start site
            tss = self.new_feature(template=i0.data, type=self.gffkey.TSS, start=at, end=at, phase=None)
            self.clean_features.append(tss)
            self.status.saw_tss()
        elif self.gffkey.cds in possible_types:
            # this could be first exon detected or start codon, ultimately, indeterminate
            cds_feature = self.pick_one_interval(intervals, target_type=self.gffkey.cds).data
            coding = self.new_feature(template=cds_feature, type=self.gffkey.status_coding, start=at, end=at)
            self.clean_features.append(coding)
            self.status.saw_start(phase=coding.phase)
            # mask a dummy region up-stream as it's very unclear whether it should be intergenic/intronic/utr
            if plus_strand:
                # unless we're at the start of the sequence
                if at != 1:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error,
                                                 start=max(1, at - self.gffkey.error_buffer - 1), end=at - 1, phase=None)
                    self.clean_features.insert(0, feature_e)
            else:
                end_of_sequence = self.get_seq_length(cds_feature.seqid)
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error, start=at + 1,
                                                 end=min(end_of_sequence, at + self.gffkey.error_buffer + 1), phase=None)
                    feature_e.type = self.gffkey.error
                    self.clean_features.insert(0, feature_e)
        else:
            raise ValueError("why's this gene not start with 5' utr nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types))

    def interpret_last_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.downstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.three_prime_UTR in possible_types:
            # this should be transcription termination site
            tts = self.new_feature(template=i0.data, type=self.gffkey.TTS, start=at, end=at, phase=None)
            self.clean_features.append(tts)
            self.status.saw_tts()
        elif self.gffkey.cds in possible_types:
            # may or may not be stop codon, but will just mark as error (unless at edge of sequence)
            end_of_sequence = self.get_seq_length(i0.data.seqid)
            if plus_strand:
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, start=at + 1, phase=None,
                                                 end=min(at + 1 + self.gffkey.error_buffer, end_of_sequence))
                    self.clean_features.append(feature_e)
            else:
                if at != 1:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, end=at - 1, phase=None,
                                                 start=max(1, at - self.gffkey.error_buffer - 1))
                    self.clean_features.append(feature_e)
        else:
            raise ValueError("why's this gene not end with 3' utr/exon nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types)
            )

    def intervals_5to3(self, plus_strand=False):
        interval_sets = list(self.organize_and_split_features())
        if not plus_strand:
            interval_sets.reverse()
        return interval_sets

    def decode_raw_features(self):
        plus_strand = self.is_plus_strand()
        interval_sets = self.intervals_5to3(plus_strand)

        self.interpret_first_pos(interval_sets[0], plus_strand)
        for i in range(len(interval_sets) - 1):
            ivals_before = interval_sets[i]
            ivals_after = interval_sets[i + 1]
            self.interpret_transition(ivals_before, ivals_after, plus_strand)

        self.interpret_last_pos(intervals=interval_sets[-1])

    def possible_types(self, intervals):
        # shortcuts
        cds = self.gffkey.cds
        five_prime = self.gffkey.five_prime_UTR
        exon = self.gffkey.exon
        three_prime = self.gffkey.three_prime_UTR

        # what we see
        observed_types = [x.data.type for x in intervals]
        set_o_types = set(observed_types)
        # check length
        if len(intervals) not in [1, 2]:
            raise ValueError('check interpretation by hand for transcript start with {}, {}'.format(
                '\n'.join([ival.data.short_str() for ival in intervals]), observed_types
            ))
        # interpret type combination
        if set_o_types == {exon, five_prime} or set_o_types == {five_prime}:
            out = [five_prime]
        elif set_o_types == {exon, three_prime} or set_o_types == {three_prime}:
            out = [three_prime]
        elif set_o_types == {exon}:
            out = [five_prime, three_prime]
        elif set_o_types == {cds, exon} or set_o_types == {cds}:
            out = [cds]
        else:
            raise ValueError('check interpretation of combination for transcript start with {}, {}'.format(
                intervals, observed_types
            ))
        return out

    def organize_and_split_features(self):
        # todo, handle non-single seqid loci
        tree = intervaltree.IntervalTree()
        features = [self.super_loci.features[f] for f in self.transcript.features]
        for f in features:
            tree[f.py_start:f.py_end] = f
        tree.split_overlaps()
        # todo, minus strand
        intervals = iter(sorted(tree))
        out = [next(intervals)]
        for interval in intervals:
            if out[-1].begin == interval.begin:
                out.append(interval)
            else:
                yield out
                out = [interval]
        yield out

    def get_seq_length(self, seqid):
        return self.super_loci.genome.meta_info_sequences[seqid].total_bp


def min_max(x, y):
    return min(x, y), max(x, y)


def upstream(x, y, sign):
    if (y - x) * sign >= 0:
        return x
    else:
        return y


def downstream(x, y, sign):
    if (x - y) * sign >= 0:
        return x
    else:
        return y
