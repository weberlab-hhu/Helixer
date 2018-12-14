from dustdas import gffhelper
import intervaltree
import annotations
import annotations_orm
import type_enums
import sequences
import copy
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


##### main flow control #####
class ImportControl(object):

    def __init__(self, database_path, err_path=None):
        self.database_path = database_path
        self.session = None
        self.err_path = err_path
        self.engine = None
        self.annotated_genome = None
        self.sequence_info = None
        self.super_loci = []
        self.mk_session()

    def mk_session(self):
        self.engine = create_engine(self.database_path, echo=False)  # todo, dynamic / real path
        annotations_orm.Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def gff_gen(self, gff_file):
        known = [x.value for x in type_enums.AllKnown]
        reader = gffhelper.read_gff_file(gff_file)
        for entry in reader:
            if entry.type not in known:
                raise ValueError("unrecognized feature type from gff: {}".format(entry.type))
            else:
                self.clean_entry(entry)
                yield entry

    @staticmethod
    def clean_entry(entry):
        # always present and integers
        entry.start = int(entry.start)
        entry.end = int(entry.end)
        # clean up score
        if entry.score == '.':
            entry.score = None
        else:
            entry.score = float(entry.score)

        # clean up phase
        if entry.phase == '.':
            entry.phase = None
        else:
            entry.phase = int(entry.phase)
        assert entry.phase in [None, 0, 1, 2]

        # clean up strand
        if entry.strand == '.':
            entry.strand = None
        else:
            assert entry.strand in ['+', '-']

    def useful_gff_entries(self, gff_file):
        skipable = [x.value for x in type_enums.IgnorableFeatures]
        reader = self.gff_gen(gff_file)
        for entry in reader:
            if entry.type not in skipable:
                yield entry

    def group_gff_by_gene(self, gff_file):
        gene_level = [x.value for x in type_enums.SuperLocusAll]
        reader = self.useful_gff_entries(gff_file)
        gene_group = [next(reader)]
        for entry in reader:
            if entry.type in gene_level:
                yield gene_group
                gene_group = [entry]
            else:
                gene_group.append(entry)
        yield gene_group

    def make_anno_genome(self, **kwargs):
        # todo, parse in meta data from kwargs?
        self.annotated_genome = annotations.AnnotatedGenomeHandler()
        ag = annotations_orm.AnnotatedGenome()
        self.annotated_genome.add_data(ag)
        self.session.add(ag)
        self.session.commit()

    def add_sequences(self, json_path):
        if self.annotated_genome is None:
            self.make_anno_genome()
        sg = sequences.StructuredGenome()
        sg.from_json(json_path)
        seq_info = annotations_orm.SequenceInfo(annotated_genome=self.annotated_genome.data)
        self.sequence_info = annotations.SequenceInfoHandler()
        self.sequence_info.add_data(seq_info)
        self.sequence_info.add_sequences(sg)

    def add_gff(self, gff_file):
        super_loci = []
        err_handle = open(self.err_path, 'w')
        for entry_group in self.group_gff_by_gene(gff_file):
            super_locus = SuperLocusHandler()
            super_locus.add_gff_entry_group(entry_group, err_handle)
            try:
                super_locus.data.sequence_info = self.sequence_info.data
            except AttributeError as e:
                raise AttributeError(str(e) + ' You need to import sequence data (self.add_sequences(...)')
            self.session.add(super_locus.data)
            self.session.commit()
            super_loci.append(super_locus)  # just to keep some direct python link to this
        self.super_loci = super_loci
        err_handle.close()

    def clean_super_loci(self):
        for sl in self.super_loci:
            sl.check_and_fix_structure()
#    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
#        err_handle = open(err_file, 'w')
#        self._add_sequences(genome)
#
#        gff_seq_ids = helpers.get_seqids_from_gff(gff_file)
#        mapper, is_forward = helpers.two_way_key_match(self.seq_info.keys(), gff_seq_ids)
#        self.mapper = mapper
#
#        if not is_forward:
#            raise NotImplementedError("Still need to implement backward match if fasta IDs are subset of gff IDs")
#
#        for entry_group in self.group_gff_by_gene(gff_file):
#            new_sl = SuperLocus()
#            new_sl.slice = self
#            new_sl.add_gff_entry_group(entry_group, err_handle)
#
#            self.super_loci.append(new_sl)
#            if not new_sl.transcripts and not new_sl.features:
#                print('{} from {} with {} transcripts and {} features'.format(new_sl.id,
#                                                                              entry_group[0].source,
#                                                                              len(new_sl.transcripts),
#                                                                              len(new_sl.features)))
#        err_handle.close()


def in_values(x, enum):
    return x in [item.value for item in enum]



##### gff parsing subclasses #####
class GFFDerived(object):
    def __init__(self):
        self.gffentry = None

    def process_gffentry(self, gffentry, gen_data=True, **kwargs):
        self.gffentry = gffentry
        data = None
        if gen_data:
            data = self.gen_data_from_gffentry(gffentry, **kwargs)
            #self.add_data(data)
        return data

    def gen_data_from_gffentry(self, gffentry, **kwargs):
        # should create 'data' object (annotations_orm.Base subclass) and then call self.add_data(data)
        raise NotImplementedError


class SuperLocusHandler(annotations.SuperLocusHandler, GFFDerived):
    def __init__(self):
        annotations.SuperLocusHandler.__init__(self)
        GFFDerived.__init__(self)
        self.transcribed_handlers = []
        self.translated_handlers = []
        self.feature_handlers = []

    def gen_data_from_gffentry(self, gffentry, sequence_info=None, **kwargs):
        data = self.data_type(type=gffentry.type,
                              given_id=gffentry.get_ID(),
                              sequence_info=sequence_info)
        self.add_data(data)
        # todo, grab more aliases from gff attribute

#    def dummy_transcript(self):
#        if self._dummy_transcript is not None:
#            return self._dummy_transcript
#        else:
#            # setup new blank transcript
#            transcript = FeatureHolder()
#            transcript.id = self.genome.transcript_ider.next_unique_id()  # add an id
#            transcript.super_locus = self
#            self._dummy_transcript = transcript  # save to be returned by next call of dummy_transcript
#            self.generic_holders[transcript.id] = transcript  # save into main dict of transcripts
#            return transcript
#

    def add_gff_entry(self, entry):
        exceptions = entry.attrib_filter(tag="exception")
        for exception in [x.value for x in exceptions]:
            if 'trans-splicing' in exception:
                raise TransSplicingError('trans-splice in attribute {} {}'.format(entry.get_ID(), entry.attribute))
        if in_values(entry.type, type_enums.SuperLocusAll):
            self.process_gffentry(gffentry=entry)

        elif in_values(entry.type, type_enums.TranscriptLevelAll):
            transcribed = TranscribedHandler()
            transcribed.process_gffentry(entry, super_locus=self.data)
            self.transcribed_handlers.append(transcribed)

        elif in_values(entry.type, type_enums.OnSequence):
            feature = FeatureHandler()
            assert len(self.transcribed_handlers) > 0, "no transcribeds found before feature"
            feature.process_gffentry(entry, super_locus=self.data,
                                     transcribeds=[self.transcribed_handlers[-1].data])
        else:
            raise ValueError("problem handling entry of type {}".format(entry.type))

    def _add_gff_entry_group(self, entries):
        entries = list(entries)
        for entry in entries:
            self.add_gff_entry(entry)

    def add_gff_entry_group(self, entries, ts_err_handle):
        try:
            self._add_gff_entry_group(entries)
            #self.check_and_fix_structure(entries)
        except TransSplicingError as e:
            self._mark_erroneous(entries[0], 'trans-splicing')
            logging.warning('skipping but noting trans-splicing: {}'.format(str(e)))
            ts_err_handle.writelines([x.to_json() for x in entries])
        except AssertionError as e:
            self._mark_erroneous(entries[0], str(e))
            # todo, log to file

#    @staticmethod
#    def one_parent(entry):
#        parents = entry.get_Parent()
#        assert len(parents) == 1
#        return parents[0]
#
    def _mark_erroneous(self, entry, msg=''):
        assert entry.type in [x.value for x in type_enums.SuperLocusAll]
        logging.warning(
            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, {msg} - marking erroneous'.format(
                src=entry.source, species="todo", seqid=entry.seqid, start=entry.start,
                end=entry.end, gene_id=self.data.given_id, msg=msg
            ))
        sf = FeatureHandler()
        sf.gen_data_from_gffentry(entry, super_locus=self.data)
        sf.set_data_attribute('type', type_enums.ErrorFeature.error.name)

    def check_and_fix_structure(self, sess):
        # if it's empty (no bottom level features at all) mark as erroneous
        if not self.data.features:
            self._mark_erroneous(self.gffentry)

        for transcript in self.data.transcribeds:
            # mark old features
            for feature in transcript.features:
                feature.handler.mark_for_deletion()
            # make new features
            t_interpreter = TranscriptInterpreter(transcript.handler)
            t_interpreter.decode_raw_features()
            # make sure the new features link to protein if appropriate
            t_interpreter.mv_coding_features_to_proteins()
        # remove old features
        self.delete_marked_underlings(sess)


class FeatureHandler(annotations.FeatureHandler, GFFDerived):

    def __init__(self):
        annotations.FeatureHandler.__init__(self)
        GFFDerived.__init__(self)

    def gen_data_from_gffentry(self, gffentry, super_locus=None, transcribeds=None, translateds=None, **kwargs):
        if transcribeds is None:
            transcribeds = []
        if translateds is None:
            translateds = []

        parents = gffentry.get_Parent()
        for transcribed in transcribeds:
            assert transcribed.given_id in parents

        given_id = gffentry.get_ID()  # todo, None on missing
        is_plus_strand = gffentry.strand == '+'

        data = self.data_type(
            given_id=given_id,
            type=gffentry.type,
            seqid=gffentry.seqid,
            start=gffentry.start,
            end=gffentry.end,
            is_plus_strand=is_plus_strand,
            score=gffentry.score,
            source=gffentry.source,
            phase=gffentry.phase,
            super_locus=super_locus,
            transcribeds=transcribeds,
            translateds=translateds
        )
        self.add_data(data)

    # inclusive and from 1 coordinates
    def upstream_from_interval(self, interval):
        if self.data.is_plus_strand:
            return interval.begin + 1
        else:
            return interval.end

    def downstream_from_interval(self, interval):
        if self.data.is_plus_strand:
            return interval.end
        else:
            return interval.begin + 1


class TranscribedHandler(annotations.TranscribedHandler, GFFDerived):
    def __init__(self):
        annotations.TranscribedHandler.__init__(self)
        GFFDerived.__init__(self)

    def gen_data_from_gffentry(self, gffentry, super_locus=None, **kwargs):
        parents = gffentry.get_Parent()
        # the simple case
        if len(parents) == 1:
            assert super_locus.given_id == parents[0]
            data = self.data_type(type=gffentry.type,
                                  given_id=gffentry.get_ID(),
                                  super_locus=super_locus)
            self.add_data(data)
        else:
            raise NotImplementedError  # todo handle multi inheritance, etc...


class TranslatedHandler(annotations.TranslatedHandler):
    pass


class NoTranscriptError(Exception):
    pass


class TransSplicingError(Exception):
    pass


class NoGFFEntryError(Exception):
    pass


class OverlapStatus(object):
    contained = 'contained'
    contains = 'contains'
    no_overlap = 'no_overlap'
    overlaps_upstream = 'overlaps_upstream'
    overlaps_downstream = 'overlaps_downstream'
    accepted_stati = (contained, no_overlap, overlaps_upstream, overlaps_downstream)

    def __init__(self):
        self._status = None

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        assert status in OverlapStatus.accepted_stati
        self._status = status

    def set_status(self, feature, seqid, start, end):
        err_str = 'Non handled overlap feature({}, {}, {}) vs slice({}, {}, {})'.format(
                feature.seqid, feature.start, feature.end,
                seqid, start, end
            )
        overlaps_at_start = False
        overlaps_at_end = False
        if feature.seqid != seqid:
            out = OverlapStatus.no_overlap
        elif feature.start >= start and feature.end <= end:
            out = OverlapStatus.contained
        elif feature.start < start and feature.end > end:
            out = OverlapStatus.contains
        elif feature.end < start or feature.start > end:
            out = OverlapStatus.no_overlap
        elif feature.start < start and feature.end >= start:
            overlaps_at_start = True
        elif feature.end > end and feature.start <= end:
            overlaps_at_end = True
        else:
            raise ValueError(err_str)

        plus_strand = feature.is_plus_strand()
        if overlaps_at_start and overlaps_at_end:
            raise ValueError(err_str + ' Overlaps both ends???')  # todo, test this properly and remove run time check

        if (overlaps_at_start and plus_strand) or (overlaps_at_end and not plus_strand):
            out = OverlapStatus.overlaps_upstream
        if (overlaps_at_end and plus_strand) or (overlaps_at_start and not plus_strand):
            out = OverlapStatus.overlaps_downstream
        self.status = out


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


class TranscriptInterpBase(object):
    def __init__(self, transcript):
        assert isinstance(transcript, TranscribedHandler)
        self.status = TranscriptStatus()
        self.transcript = transcript

    @property
    def super_locus(self):
        return self.transcript.data.super_locus.handler


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript):
        super().__init__(transcript)

    def crop_to_slice(self, seqid, start, end):
        """crops transcript in place"""
        pass

    def transition_5p_to_3p(self):
        pass


class TranscriptInterpreter(TranscriptInterpBase):
    """takes raw/from-gff transcript, and makes totally explicit"""
    def __init__(self, transcript):
        super().__init__(transcript)
        self.clean_features = []  # will hold all the 'fixed' feature handlers (convenience? or can remove?)
        self.transcript = transcript
        try:
            self.proteins = self._setup_proteins()
        except NoGFFEntryError:
            self.proteins = None  # this way we only run into an error if we actually wanted to use proteins

    def new_feature(self, template, **kwargs):
        handler = FeatureHandler()
        data = annotations_orm.Feature()
        handler.add_data(data)
        template.fax_all_attrs_to_another(another=handler)
        handler.gffentry = copy.deepcopy(template.gffentry)

        for key in kwargs:
            handler.set_data_attribute(key, kwargs[key])
        return handler

    @staticmethod
    def pick_one_interval(interval_set, target_type=None):
        if target_type is None:
            return interval_set[0]
        else:
            print([x.data.data.type for x in interval_set])
            return [x for x in interval_set if x.data.data.type.value == target_type][0]

    @staticmethod
    def _get_protein_id_from_cds(cds_feature):
        try:
            assert cds_feature.gffentry.type == type_enums.CDS, "{} != {}".format(cds_feature.gff_entry.type,
                                                                                  type_enums.CDS)
        except AttributeError:
            raise NoGFFEntryError('No gffentry for {}'.format(cds_feature.data.given_id))
        # check if anything is labeled as protein_id
        protein_id = cds_feature.gffentry.attrib_filter(tag='protein_id')
        # failing that, try and get parent ID (presumably transcript, maybe gene)
        if not protein_id:
            protein_id = cds_feature.gffentry.get_Parent()
        # hopefully take single hit
        if len(protein_id) == 1:
            protein_id = protein_id[0]
            if isinstance(protein_id, gffhelper.GFFAttribute):
                protein_id = protein_id.value
                assert len(protein_id) == 1
                protein_id = protein_id[0]
        # or handle other cases
        elif len(protein_id) == 0:
            protein_id = None
        else:
            raise ValueError('indeterminate single protein id {}'.format(protein_id))
        print(protein_id, type(protein_id), 'pid, type')
        return protein_id

    def _get_raw_protein_ids(self):
        # only meant for use before feature interpretation
        protein_ids = set()
        for feature in self.transcript.data.features:
            if feature.type.value == type_enums.CDS:
                protein_id = self._get_protein_id_from_cds(feature.handler)
                protein_ids.add(protein_id)
        return protein_ids

    def _setup_proteins(self):
        # only meant for use before feature interpretation
        pids = self._get_raw_protein_ids()
        proteins = {}
        for key in pids:
            print('making protein {}'.format(key))
            # setup blank protein
            protein = TranslatedHandler()
            pdata = annotations_orm.Translated()
            protein.add_data(pdata)
            # copy most all attributes from self to protein (except:
            # translateds bc invalid, and
            # features bc at this point they are all input features that'll be removed anyways
            self.transcript.fax_all_attrs_to_another(another=protein, skip_linking=['translateds', 'features'])
            pdata.transcribeds = [self.transcript.data]  # link back to transcript
            pdata.given_id = key  # set given_id to what was found
            proteins[key] = protein

        return proteins

    def mv_coding_features_to_proteins(self):
        print('proteins to move to', self.proteins)
        # only meant for use after feature interpretation
        print('clean features {}'.format([x.data.type for x in self.clean_features]))
        for feature in self.clean_features:
            if feature.data.type in [x.value for x in type_enums.TranslatedAll]:  # todo, fix brittle to pre/post commit
                print('swapping {} {}'.format(feature, feature.data.type))
                pid = self._get_protein_id_from_cds(feature)
                self.transcript.replace_selflink_with_replacementlink(replacement=self.proteins[pid],
                                                                      data=feature.data)
            else:
                print('not swapping {} {}'.format(feature, feature.data.type))

    def is_plus_strand(self):
        features = self.transcript.data.features
        seqids = [x.seqid for x in features]
        if not all([x == seqids[0] for x in seqids]):
            raise TransSplicingError("non matching seqids {}, for {}".format(seqids, self.super_locus.id))
        if all([x.is_plus_strand for x in features]):
            return True
        elif all([not x.is_plus_strand for x in features]):
            return False
        else:
            raise TransSplicingError("Mixed strands at {} with {}".format(self.super_locus.id,
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
        assert type_enums.CDS in before_types
        # stop codon
        if type_enums.THREE_PRIME_UTR in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=False)
        # splice site
        elif type_enums.CDS in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError("don't know how to transition from coding to {}".format(after_types))

    def handle_from_intron(self):
        raise NotImplementedError  # todo later

    def handle_from_3p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert type_enums.THREE_PRIME_UTR in before_types
        # the only thing we should encounter is a splice site
        if type_enums.THREE_PRIME_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after three prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_5p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert type_enums.FIVE_PRIME_UTR in before_types
        # start codon
        if type_enums.CDS in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=True)
        # intron
        elif type_enums.FIVE_PRIME_UTR in after_types:
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
                before_downstream, after_upstream, after0.data.seqid, self.super_locus.id, before0.data.id,
                after0.data.id
            )
        return is_gap

    def handle_control_codon(self, ivals_before, ivals_after, sign, is_start=True, error_buffer=2000):
        target_after_type = None
        target_before_type = None
        if is_start:
            target_after_type = type_enums.CDS
        else:
            target_before_type = type_enums.CDS

        after0 = self.pick_one_interval(ivals_after, target_after_type)
        before0 = self.pick_one_interval(ivals_before, target_before_type)
        # make sure there is no gap
        is_gap = self.is_gap(ivals_before, ivals_after, sign)

        if is_start:
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

            template = after0.data
            # it better be std phase if it's a start codon
            at = template.upstream_from_interval(after0)
            if template.data.phase == 0:  # "non-0 phase @ {} in {}".format(template.id, template.super_locus.id)
                start, end = min_max(at, at + 2 * sign)
                start_codon = self.new_feature(template=template, start=start, end=end, type=type_enums.START_CODON)
                self.status.saw_start(phase=0)
                self.clean_features.append(start_codon)
            else:
                upstream_buffered = before0.data.upstream_from_interval(before0) - sign * error_buffer
                err_start, err_end = min_max(at - 1 * sign, upstream_buffered)
                feature_e = self.new_feature(template=template, type=type_enums.ERROR,
                                             start=err_start, end=err_end, phase=None)
                coding_status = self.new_feature(template=template, type=type_enums.IN_TRANSLATED_REGION, start=at,
                                                 end=at)
                transcribed_status = self.new_feature(template=template, type=type_enums.IN_RAW_TRANSCRIPT, start=at,
                                                      end=at, phase=None)
                self.status.saw_start(template.phase)
                self.clean_features += [feature_e, coding_status, transcribed_status]
        else:
            # todo, confirm phase for stop codon
            template = before0.data
            at = template.downstream_from_interval(before0)
            start, end = min_max(at, at - 2 * sign)
            stop_codon = self.new_feature(template=template, start=start, end=end, type=type_enums.STOP_CODON)
            self.status.saw_stop()
            self.clean_features.append(stop_codon)
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

    def handle_splice(self, ivals_before, ivals_after, sign):
        target_type = None
        if self.status.is_coding():
            target_type = type_enums.CDS

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
                                     type=type_enums.DONOR_SPLICE_SITE)
            # todo, check position of DSS/ASS to be consistent with Augustus, hopefully
            acceptor = self.new_feature(template=acceptor_tmplt, start=acceptor_at, end=acceptor_at,
                                        type=type_enums.ACCEPTOR_SPLICE_SITE)
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
            feature_e.type = type_enums.ERROR
            self.clean_features.append(feature_e)

    def interpret_first_pos(self, intervals, plus_strand=True, error_buffer=2000):
        # shortcuts
        cds = type_enums.CDS
        error = type_enums.ERROR

        i0 = self.pick_one_interval(intervals)
        at = i0.data.upstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if type_enums.FIVE_PRIME_UTR in possible_types:
            # this should indicate we're good to go and have a transcription start site
            tss = self.new_feature(template=i0.data, type=type_enums.TRANSCRIPTION_START_SITE, start=at, end=at,
                                   phase=None)
            self.clean_features.append(tss)
            self.status.saw_tss()
        elif cds in possible_types:
            # this could be first exon detected or start codon, ultimately, indeterminate
            cds_feature = self.pick_one_interval(intervals, target_type=cds).data
            coding = self.new_feature(template=cds_feature, type=type_enums.IN_TRANSLATED_REGION, start=at, end=at)
            transcribed = self.new_feature(template=cds_feature, type=type_enums.IN_RAW_TRANSCRIPT, start=at, end=at)
            self.clean_features += [coding, transcribed]
            self.status.saw_start(phase=coding.data.phase)
            # mask a dummy region up-stream as it's very unclear whether it should be intergenic/intronic/utr
            if plus_strand:
                # unless we're at the start of the sequence
                start_of_sequence = self.get_seq_start(cds_feature.data.seqid)
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=error,
                                                 start=max(start_of_sequence, at - error_buffer - 1),
                                                 end=at - 1, phase=None)
                    self.clean_features.insert(0, feature_e)
            else:
                end_of_sequence = self.get_seq_end(cds_feature.data.seqid)
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=error, start=at + 1,
                                                 end=min(end_of_sequence, at + error_buffer + 1),
                                                 phase=None)
                    self.clean_features.insert(0, feature_e)
        else:
            raise ValueError("why's this gene not start with 5' utr nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types))

    def interpret_last_pos(self, intervals, plus_strand=True, error_buffer=2000):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.downstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if type_enums.THREE_PRIME_UTR in possible_types:
            # this should be transcription termination site
            tts = self.new_feature(template=i0.data, type=type_enums.TRANSCRIPTION_TERMINATION_SITE, start=at, end=at, phase=None)
            self.clean_features.append(tts)
            self.status.saw_tts()
        elif type_enums.CDS in possible_types:
            # may or may not be stop codon, but will just mark as error (unless at edge of sequence)
            start_of_sequence = self.get_seq_start(i0.data.data.seqid)
            end_of_sequence = self.get_seq_end(i0.data.data.seqid)
            if plus_strand:
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=type_enums.ERROR, start=at + 1, phase=None,
                                                 end=min(at + 1 + error_buffer, end_of_sequence))
                    self.clean_features.append(feature_e)
            else:
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=type_enums.ERROR, end=at - 1, phase=None,
                                                 start=max(start_of_sequence, at - error_buffer - 1))
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
        cds = type_enums.CDS
        five_prime = type_enums.FIVE_PRIME_UTR
        exon = type_enums.EXON
        three_prime = type_enums.THREE_PRIME_UTR

        # what we see
        observed_types = [x.data.data.type.name for x in intervals]
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
        features = [f.handler for f in self.transcript.data.features]
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

    def get_seq_end(self, seqid):
        return self.transcript.data.super_locus.sequence_info.handler.seq_info[seqid].end

    def get_seq_start(self, seqid):
        return self.transcript.data.super_locus.sequence_info.handler.seq_info[seqid].start


def min_max(x, y):
    return min(x, y), max(x, y)


def none_to_list(x):
    if x is None:
        return []
    else:
        assert isinstance(x, list)
        return x


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