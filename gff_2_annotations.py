from dustdas import gffhelper
import intervaltree
import annotations
import annotations_orm
import type_enums
import sequences
import helpers

import copy
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


##### main flow control #####
class ImportControl(object):

    def __init__(self, database_path, err_path=None):
        if not database_path.startswith('sqlite:///'):
            database_path = 'sqlite:///{}'.format(database_path)
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
        self.sequence_info = SequenceInfoHandler()
        self.sequence_info.add_data(seq_info)
        self.sequence_info.add_sequences(sg)

    def add_gff(self, gff_file):
        # final prepping of seqid match up
        self.sequence_info.mk_mapper(gff_file)

        super_loci = []
        err_handle = open(self.err_path, 'w')
        for entry_group in self.group_gff_by_gene(gff_file):
            super_locus = SuperLocusHandler()
            if self.sequence_info is None:
                raise AttributeError(
                    ' sequence_info cannot be None when .add_gff is called, use (self.add_sequences(...)')
            super_locus.add_gff_entry_group(entry_group, err_handle, sequence_info=self.sequence_info.data)
            self.session.add(super_locus.data)
            self.session.commit()
            super_loci.append(super_locus)  # just to keep some direct python link to this
        self.super_loci = super_loci
        err_handle.close()

    def clean_super_loci(self):
        for sl in self.super_loci:
            coordinates =  self.sequence_info.handler.gffid_to_coords[sl.gff_entry.seqid]
            sl.check_and_fix_structure(self.session, coordinates)
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


class SequenceInfoHandler(annotations.SequenceInfoHandler):
    def __init__(self):
        super().__init__()
        self.mapper = None
        self._seq_info = None
        self._gffid_to_coords = None
        self._gff_seq_ids = None

    def mk_mapper(self, gff_file=None):
        fa_ids = [x.seqid for x in self.data.coordinates]
        if gff_file is not None:  # allow setup without ado when we know IDs match exactly
            self._gff_seq_ids = helpers.get_seqids_from_gff(gff_file)
        else:
            self._gff_seq_ids = fa_ids
        mapper, is_forward = helpers.two_way_key_match(fa_ids, self._gff_seq_ids)
        self.mapper = mapper

        if not is_forward:
            raise NotImplementedError("Still need to implement backward match if fasta IDs are subset of gff IDs")

    @property
    def gffid_to_coords(self):
        if self._gffid_to_coords is not None:
            pass
        else:
            gffid2coords = {}
            for gffid in self._gff_seq_ids:
                fa_id = self.mapper(gffid)
                x = self.seq_info[fa_id]
                gffid2coords[gffid] = x
            self._gffid_to_coords = gffid2coords
        return self._gffid_to_coords

    @property
    def seq_info(self):
        if self._seq_info is not None:
            pass
        else:
            seq_info = {}
            for x in self.data.coordinates:
                seq_info[x.seqid] = x
            self._seq_info = seq_info
        return self._seq_info


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
        self.transcribed_piece_handlers = []
        self.feature_handlers = []

    def gen_data_from_gffentry(self, gffentry, **kwargs):
        data = self.data_type(type=gffentry.type,
                              given_id=gffentry.get_ID())
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

    def add_gff_entry(self, entry, sequence_info):
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

            piece = TranscribedPieceHandler()
            piece.process_gffentry(entry, super_locus=self.data, transcribed=transcribed.data)
            self.transcribed_piece_handlers.append(piece)

        elif in_values(entry.type, type_enums.OnSequence):
            feature = FeatureHandler()
            coordinates = sequence_info.handler.gffid_to_coords[entry.seqid]
            assert len(self.transcribed_handlers) > 0, "no transcribeds found before feature"
            feature.process_gffentry(entry, super_locus=self.data,
                                     transcribed_pieces=[self.transcribed_handlers[-1].one_piece().data],
                                     coordinates=coordinates)
            self.feature_handlers.append(feature)
        else:
            raise ValueError("problem handling entry of type {}".format(entry.type))

    def _add_gff_entry_group(self, entries, sequence_info):
        entries = list(entries)
        for entry in entries:
            self.add_gff_entry(entry, sequence_info)

    def add_gff_entry_group(self, entries, ts_err_handle, sequence_info):
        try:
            self._add_gff_entry_group(entries, sequence_info)
            #self.check_and_fix_structure(entries)
        except TransSplicingError as e:
            coordinates = sequence_info.handler.gffid_to_coords[entries[0].seqid]
            self._mark_erroneous(entries[0], coordinates, 'trans-splicing')
            logging.warning('skipping but noting trans-splicing: {}'.format(str(e)))
            ts_err_handle.writelines([x.to_json() for x in entries])
        except AssertionError as e:
            coordinates = sequence_info.handler.gffid_to_coords[entries[0].seqid]
            self._mark_erroneous(entries[0], coordinates, str(e))
            # todo, log to file

#    @staticmethod
#    def one_parent(entry):
#        parents = entry.get_Parent()
#        assert len(parents) == 1
#        return parents[0]
#
    def _mark_erroneous(self, entry, coordinates, msg=''):
        assert entry.type in [x.value for x in type_enums.SuperLocusAll]
        logging.warning(
            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, {msg} - marking erroneous'.format(
                src=entry.source, species="todo", seqid=entry.seqid, start=entry.start,
                end=entry.end, gene_id=self.data.given_id, msg=msg
            ))
        # reset start and stop so
        if entry.strand == '+':
            err_start = entry.start
            err_end = entry.end
        else:
            assert entry.strand == '-'
            err_start = entry.end
            err_end = entry.start
        # dummy transcript
        transcribed_e_handler = TranscribedHandler()
        transcribed_e = annotations_orm.Transcribed(super_locus=self.data)
        transcribed_e_handler.add_data(transcribed_e)
        piece_handler = TranscribedPieceHandler()
        piece = annotations_orm.TranscribedPiece(super_locus=self.data, transcribed=transcribed_e)
        piece_handler.add_data(piece)
        # open error
        feature_err_open = FeatureHandler()
        feature_err_open.process_gffentry(entry, super_locus=self.data, coordinates=coordinates)
        for key, val in [('type', type_enums.ERROR_OPEN), ('start', err_start), ('end', err_start),
                         ('transcribed_pieces', [piece])]:
            feature_err_open.set_data_attribute(key, val)
        # close error
        feature_err_close = FeatureHandler()
        feature_err_close.process_gffentry(entry, super_locus=self.data, coordinates=coordinates)
        for key, val in [('type', type_enums.ERROR_CLOSE), ('start', err_end), ('end', err_end),
                         ('transcribed_pieces', [piece])]:
            feature_err_close.set_data_attribute(key, val)
        # sf.gen_data_from_gffentry(entry, super_locus=self.data)
        self.feature_handlers += [feature_err_open, feature_err_close]
        self.transcribed_handlers.append(transcribed_e_handler)
        self.transcribed_piece_handlers.append(piece_handler)

    def check_and_fix_structure(self, sess, coordinates):
        # if it's empty (no bottom level features at all) mark as erroneous
        if not self.data.features:
            self._mark_erroneous(self.gffentry, coordinates=coordinates)

        for transcript in self.data.transcribeds:
            piece = transcript.handler.one_piece().data
            # mark old features
            for feature in piece.features:
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

    def gen_data_from_gffentry(self, gffentry, super_locus=None, transcribed_pieces=None, translateds=None,
                               coordinates=None, **kwargs):
        if transcribed_pieces is None:
            transcribed_pieces = []
        if translateds is None:
            translateds = []

        parents = gffentry.get_Parent()
        for piece in transcribed_pieces:
            assert piece.given_id in parents

        given_id = gffentry.get_ID()  # todo, None on missing
        is_plus_strand = gffentry.strand == '+'

        data = self.data_type(
            given_id=given_id,
            type=gffentry.type,
            coordinates=coordinates, #super_locus.sequence_info.handler.gffid_to_coords[gffentry.seqid],
            #seqid=gffentry.seqid,
            start=gffentry.start,
            end=gffentry.end,
            is_plus_strand=is_plus_strand,
            score=gffentry.score,
            source=gffentry.source,
            phase=gffentry.phase,
            super_locus=super_locus,
            transcribed_pieces=transcribed_pieces,
            translateds=translateds
        )
        self.add_data(data)

    # inclusive and from 1 coordinates
    def upstream_from_interval(self, interval):
        if self.data.is_plus_strand:
            return helpers.as_bio_start(interval.begin)
        else:
            return helpers.as_bio_end(interval.end)

    def downstream_from_interval(self, interval):
        if self.data.is_plus_strand:
            return helpers.as_bio_end(interval.end)
        else:
            return helpers.as_bio_start(interval.begin)

    def upstream(self):
        if self.data.is_plus_strand:
            return self.data.start
        else:
            return self.data.end

    def downstream(self):
        if self.data.is_plus_strand:
            return self.data.end
        else:
            return self.data.start


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

    def one_piece(self):
        pieces = self.data.transcribed_pieces
        assert len(pieces) == 1
        return pieces[0].handler


class TranscribedPieceHandler(annotations.TranscribedPieceHandler, GFFDerived):
    def __init__(self):
        annotations.TranscribedPieceHandler.__init__(self)
        GFFDerived.__init__(self)

    def gen_data_from_gffentry(self, gffentry, super_locus=None, transcribed=None, **kwargs):
        parents = gffentry.get_Parent()
        # the simple case
        if len(parents) == 1:
            assert super_locus.given_id == parents[0]
            data = self.data_type(given_id=gffentry.get_ID(),
                                  super_locus=super_locus,
                                  transcribed=transcribed)
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


#### section TranscriptInterpreter, might end up in a separate file later
class TranscriptStatus(object):
    """can hold and manipulate all the info on current status of a transcript"""

    def __init__(self):
        # initializes to intergenic
        self.genic = False
        self.in_intron = False
        self.in_trans_intron = False
        self.in_translated_region = False
        self.seen_start = False  # todo, move EUK specific stuff to subclass?
        self.seen_stop = False
        self.erroneous = False
        self.phase = None  # todo, proper tracking / handling

    def __repr__(self):
        return "genic: {}, intronic: {}, translated_region: {}, trans_intronic: {}, phase: {}".format(
            self.genic, self.in_intron, self.in_translated_region, self.in_trans_intron, self.phase
        )

    def saw_tss(self):
        self.genic = True

    def saw_start(self, phase):
        #self.genic = True  # TODO, disentangle this from at least status markers, better all coding & call saw_tss
        self.seen_start = True
        self.in_translated_region = True
        self.phase = phase

    def saw_stop(self):
        self.seen_stop = True
        self.in_translated_region = False
        self.phase = None

    def saw_tts(self):
        self.genic = False

    def splice_open(self):
        self.in_intron = True

    def splice_close(self):
        self.in_intron = False

    def trans_splice_open(self):
        self.in_trans_intron = True

    def trans_splice_close(self):
        self.in_trans_intron = False

    def error_open(self):
        self.erroneous = True

    def error_close(self):
        self.erroneous = False

    def is_5p_utr(self):
        return self.is_utr() and not any([self.seen_start, self.seen_stop])

    def is_3p_utr(self):
        return self.is_utr() and self.seen_stop and self.seen_start

    def is_utr(self):
        return self.genic and not any([self.in_intron, self.in_translated_region, self.in_trans_intron])

    def is_coding(self):
        return self.genic and self.in_translated_region and not any([self.in_intron, self.in_trans_intron])

    def is_intronic(self):
        return self.in_intron and self.genic

    def is_trans_intronic(self):
        return self.in_trans_intron and self.genic

    def is_intergenic(self):
        return not self.genic


class TranscriptInterpBase(object):
    # todo, move this to generic location and/or skip entirely
    def __init__(self, transcript):
        assert isinstance(transcript, annotations.TranscribedHandler)
        self.status = TranscriptStatus()
        self.transcript = transcript

    @property
    def super_locus(self):
        return self.transcript.data.super_locus.handler

    @staticmethod
    def update_status(status, aligned_features):
        for feature in aligned_features:
            ftype = feature.type.value
            # standard features
            if ftype == type_enums.TRANSCRIPTION_START_SITE:
                status.saw_tss()
            elif ftype == type_enums.START_CODON:
                status.saw_start(phase=0)
            elif ftype == type_enums.STOP_CODON:
                status.saw_stop()
            elif ftype == type_enums.TRANSCRIPTION_TERMINATION_SITE:
                status.saw_tts()
            elif ftype == type_enums.DONOR_SPLICE_SITE:
                status.splice_open()
            elif ftype == type_enums.ACCEPTOR_SPLICE_SITE:
                status.splice_close()
            elif ftype == type_enums.DONOR_TRANS_SPLICE_SITE:
                status.trans_splice_open()
            elif ftype == type_enums.ACCEPTOR_TRANS_SPLICE_SITE:
                status.trans_splice_close()
            # status features
            elif ftype == type_enums.IN_RAW_TRANSCRIPT:
                status.saw_tss()
            elif ftype == type_enums.IN_TRANSLATED_REGION:
                status.saw_start(phase=feature.phase)
            elif ftype == type_enums.IN_INTRON:
                status.splice_open()
            elif ftype == type_enums.IN_TRANS_INTRON:
                status.trans_splice_open()
            # error (and error status)
            elif ftype == type_enums.ERROR_OPEN:
                status.error_open()
            elif ftype == type_enums.ERROR_CLOSE:
                status.error_close()
            elif ftype == type_enums.IN_ERROR:
                status.error_open()
            else:
                raise ValueError('no implementation for updating status with feature of type {}'.format(ftype))


class TranscriptInterpreter(TranscriptInterpBase):
    """takes raw/from-gff transcript, and makes totally explicit"""
    HANDLED = 'handled'

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
        return protein_id

    def _get_raw_protein_ids(self):
        # only meant for use before feature interpretation
        protein_ids = set()
        for piece in self.transcript.data.transcribed_pieces:
            for feature in piece.features:
                if feature.type.value == type_enums.CDS:
                    protein_id = self._get_protein_id_from_cds(feature.handler)
                    protein_ids.add(protein_id)
        return protein_ids

    def _setup_proteins(self):
        # only meant for use before feature interpretation
        pids = self._get_raw_protein_ids()
        proteins = {}
        for key in pids:
            # setup blank protein
            protein = TranslatedHandler()
            pdata = annotations_orm.Translated()
            protein.add_data(pdata)
            # copy most all attributes from self to protein (except:
            # translateds bc invalid, and
            # features bc at this point they are all input features that'll be removed anyways
            self.transcript.fax_all_attrs_to_another(another=protein,
                                                     skip_linking=['translateds', 'transcribed_pieces'])
            pdata.transcribeds = [self.transcript.data]  # link back to transcript
            pdata.given_id = key  # set given_id to what was found
            proteins[key] = protein

        return proteins

    def mv_coding_features_to_proteins(self):
        # only meant for use after feature interpretation
        for feature in self.clean_features:
            if feature.data.type in [x.value for x in type_enums.TranslatedAll]:  # todo, fix brittle to pre/post commit
                pid = self._get_protein_id_from_cds(feature)
                piece = self.transcript.one_piece()
                #piece.replace_selflink_with_replacementlink(replacement=self.proteins[pid],
                #                                            data=feature.data)
                piece.copy_selflink_to_another(another=self.proteins[pid], data=feature.data)

    def is_plus_strand(self):
        features = set()
        for piece in self.transcript.data.transcribed_pieces:
            for feature in piece.features:
                features.add(feature)
        features = list(features)
        seqids = [x.coordinates.seqid for x in features]
        if not all([x == seqids[0] for x in seqids]):
            raise TransSplicingError("non matching seqids {}, for {}".format(seqids, self.super_locus.id))
        if all([x.is_plus_strand for x in features]):
            return True
        elif all([not x.is_plus_strand for x in features]):
            return False
        else:
            raise TransSplicingError(
                "Mixed strands at {} with {}".format(self.super_locus.id,
                                                     [(x.coordinates.seqid, x.strand) for x in features]))

    def drop_invervals_with_duplicated_data(self, ivals_before, ivals_after):
        all_data = [x.data.data for x in ivals_before + ivals_after]
        if len(all_data) > len(set(all_data)):

            marked_before = []
            for ival in ivals_before:
                new = {'interval': ival, 'matches_edge': self.matches_edge_before(ival),
                       'repeated': self.is_in_list(ival.data, [x.data for x in ivals_after])}
                marked_before.append(new)
            marked_after = []
            for ival in ivals_after:
                new = {'interval': ival, 'matches_edge': self.matches_edge_after(ival),
                       'repeated': self.is_in_list(ival.data, [x.data for x in ivals_before])}
                marked_after.append(new)

            # warn if slice-causer (matches_edge) is on both sides
            # should be fine in cases with [exon, CDS] -> [exon, UTR] or similar
            slice_frm_before = any([x['matches_edge'] for x in marked_before])
            slice_frm_after = any([x['matches_edge'] for x in marked_after])
            if slice_frm_before and slice_frm_after:
                logging.warning('slice causer on both sides with repeates\nbefore: {}, after: {}'.format(
                    [x.data.type for x in ivals_before],
                    [x.data.type for x in ivals_after]
                ))
            # finally, keep non repeats or where this side didn't cause slice
            ivals_before = [x['interval'] for x in marked_before if not x['repeated'] or not slice_frm_before]
            ivals_after = [x['interval'] for x in marked_after if not x['repeated'] or not slice_frm_after]
        return ivals_before, ivals_after

    @staticmethod
    def is_in_list(target, the_list):
        matches = [x for x in the_list if x is target]
        return len(matches) > 0

    @staticmethod
    def matches_edge_before(ival):
        data = ival.data
        if data.data.is_plus_strand:
            out = data.py_end == ival.end
        else:
            out = data.py_start == ival.begin
        return out

    @staticmethod
    def matches_edge_after(ival):
        data = ival.data
        if data.data.is_plus_strand:
            out = data.py_start == ival.begin
        else:
            out = data.py_end == ival.end
        return out

    def interpret_transition(self, ivals_before, ivals_after, plus_strand=True):
        sign = 1
        if not plus_strand:
            sign = -1
        ivals_before, ivals_after = self.drop_invervals_with_duplicated_data(ivals_before, ivals_after)
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
                before_downstream, after_upstream, after0.data.coordinates.seqid, self.super_locus.id, before0.data.id,
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
                start = end = at
                start_codon = self.new_feature(template=template, start=start, end=end, type=type_enums.START_CODON)
                self.status.saw_start(phase=0)
                self.clean_features.append(start_codon)
            else:
                err_start = before0.data.upstream_from_interval(before0) - sign * error_buffer  # mask prev feat. too
                err_end = at - 1 * sign  # 1bp upstream of erroneous "start"

                feature_err_open = self.new_feature(template=template, type=type_enums.ERROR_OPEN, start=err_start,
                                                    end=err_start, phase=None)
                feature_err_close = self.new_feature(template=template, type=type_enums.ERROR_CLOSE, start=err_end,
                                                     end=err_end, phase=None)
                coding_status = self.new_feature(template=template, type=type_enums.IN_TRANSLATED_REGION, start=at,
                                                 end=at)
                transcribed_status = self.new_feature(template=template, type=type_enums.IN_RAW_TRANSCRIPT, start=at,
                                                      end=at, phase=None)
                self.status.saw_start(template.phase)
                self.clean_features += [feature_err_open, feature_err_close, coding_status, transcribed_status]
        else:
            # todo, confirm phase for stop codon
            template = before0.data
            at = template.downstream_from_interval(before0)
            start = end = at
            stop_codon = self.new_feature(template=template, start=start, end=end, type=type_enums.STOP_CODON)
            self.status.saw_stop()
            self.clean_features.append(stop_codon)
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

    def handle_splice(self, ivals_before, ivals_after, sign):
        print('handling splice')
        target_type = None
        if self.status.is_coding():
            target_type = type_enums.CDS

        before0 = self.pick_one_interval(ivals_before, target_type)
        after0 = self.pick_one_interval(ivals_after, target_type)
        donor_tmplt = before0.data
        acceptor_tmplt = after0.data
        donor_at = donor_tmplt.downstream() + (1 * sign)  # mod by one because sites within intron
        acceptor_at = acceptor_tmplt.upstream() - (1 * sign)
        # add splice sites if there's a gap
        between_splice_sites = (acceptor_at - donor_at) * sign
        min_intron_len = 3  # todo, maybe get something small but not entirely impossible?
        if between_splice_sites > min_intron_len - 1:  # -1 because the splice sites are _within_ the intron
            print('positive intron')
            donor = self.new_feature(template=donor_tmplt, start=donor_at, end=donor_at, phase=None,
                                     type=type_enums.DONOR_SPLICE_SITE)
            # todo, check position of DSS/ASS to be consistent with Augustus, hopefully
            acceptor = self.new_feature(template=acceptor_tmplt, start=acceptor_at, end=acceptor_at,
                                        type=type_enums.ACCEPTOR_SPLICE_SITE)
            self.clean_features += [donor, acceptor]
        # do nothing if there is just no gap between exons for a techinical / reporting error
        elif between_splice_sites == -1:
            print('null intron')
            pass
        # everything else is invalid
        else:
            print('error')
            err_start = before0.data.upstream()
            err_end = after0.data.downstream()
            feature_err_open = self.new_feature(template=before0.data, start=err_start, end=err_start,
                                                type=type_enums.ERROR_OPEN)
            feature_err_close = self.new_feature(template=before0.data, start=err_end, end=err_end,
                                                 type=type_enums.ERROR_CLOSE)

            self.clean_features += [feature_err_open, feature_err_close]

    def interpret_first_pos(self, intervals, plus_strand=True, error_buffer=2000):
        # shortcuts
        cds = type_enums.CDS

        i0 = self.pick_one_interval(intervals)
        at = i0.data.upstream_from_interval(i0)
        # todo, allow for handles errors to already exist at this point (type= error_open/error_close) WAS HERE
        possible_types = self.possible_types(intervals)
        if possible_types == [TranscriptInterpreter.HANDLED]:
            self.update_status(self.status, aligned_features=[x.data.data for x in intervals])
        elif type_enums.FIVE_PRIME_UTR in possible_types:
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
            self.status.saw_tss()  # coding implies the transcript
            # mask a dummy region up-stream as it's very unclear whether it should be intergenic/intronic/utr
            if plus_strand:
                # unless we're at the start of the sequence
                start_of_sequence = cds_feature.data.coordinates.start
                if at != start_of_sequence:
                    err_start = max(start_of_sequence, at - error_buffer - 1)
                    err_end = at - 1
                    feature_err_open = self.new_feature(template=cds_feature, type=type_enums.ERROR_OPEN,
                                                        start=err_start, end=err_start, phase=None)
                    feature_err_close = self.new_feature(template=cds_feature, type=type_enums.ERROR_CLOSE,
                                                         start=err_end, end=err_end, phase=None)
                    self.clean_features.insert(0, feature_err_close)
                    self.clean_features.insert(0, feature_err_open)
            else:
                end_of_sequence = cds_feature.data.coordinates.end
                if at != end_of_sequence:
                    err_start = min(end_of_sequence, at + error_buffer + 1)
                    err_end = at + 1
                    feature_err_open = self.new_feature(template=cds_feature, type=type_enums.ERROR_OPEN,
                                                        start=err_start, end=err_start, phase=None)
                    feature_err_close = self.new_feature(template=cds_feature, type=type_enums.ERROR_CLOSE,
                                                         start=err_end, end=err_end, phase=None)
                    self.clean_features.insert(0, feature_err_close)
                    self.clean_features.insert(0, feature_err_open)
        else:
            raise ValueError("why's this gene not start with 5' utr nor cds? types: {}, interpretations: {}".format(
                [x.data.data.type for x in intervals], possible_types))

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
            start_of_sequence = i0.data.data.coordinates.start
            end_of_sequence = i0.data.data.coordinates.end
            if plus_strand:
                if at != end_of_sequence:
                    err_start = at + 1
                    err_end = min(at + 1 + error_buffer, end_of_sequence)
                    feature_err_open = self.new_feature(template=i0.data, type=type_enums.ERROR_OPEN, start=err_start,
                                                        phase=None, end=err_start)
                    feature_err_close = self.new_feature(template=i0.data, type=type_enums.ERROR_CLOSE, start=err_end,
                                                         phase=None, end=err_end)
                    self.clean_features += [feature_err_open, feature_err_close]
            else:
                if at != start_of_sequence:
                    err_start = at - 1
                    err_end = max(start_of_sequence, at - error_buffer - 1)
                    feature_err_open = self.new_feature(template=i0.data, type=type_enums.ERROR_OPEN, end=err_start,
                                                        phase=None, start=err_start)
                    feature_err_close = self.new_feature(template=i0.data, type=type_enums.ERROR_CLOSE, end=err_end,
                                                         phase=None, start=err_end)
                    self.clean_features += [feature_err_open, feature_err_close]
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
        # todo, detect completely handled (prolly error-error) transcript here and pass WAS HERE, TUESDAY
        plus_strand = self.is_plus_strand()
        interval_sets = self.intervals_5to3(plus_strand)
        print([len(ivs) for ivs in interval_sets], 'ivs lengths')
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
        #uniq_datas = set([x.data.data for x in intervals])  # todo, revert and skip unique once handled above
        observed_types = [x.data.data.type.name for x in intervals]
        set_o_types = set(observed_types)
        # check length
        if len(intervals) not in [1, 2]:
            raise IntervalCountError('check interpretation by hand for transcript start with {}, {}'.format(
                '\n'.join([str(ival.data.data) for ival in intervals]), observed_types
            ))
        if set_o_types.issubset(set([x.value for x in type_enums.KeepOnSequence])):
            out = [TranscriptInterpreter.HANDLED]
        # interpret type combination
        elif set_o_types == {exon, five_prime} or set_o_types == {five_prime}:
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
        features = set()
        for piece in self.transcript.data.transcribed_pieces:
            for feature in piece.features:
                features.add(feature)
        features = [f.handler for f in features]
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


class IntervalCountError(Exception):
    pass


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