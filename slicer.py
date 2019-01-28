"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
from shutil import copyfile
import intervaltree
import copy

import annotations
import annotations_orm
import slice_dbmods
import type_enums

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import os
import sequences
from helpers import as_py_start, as_py_end
from gff_2_annotations import TranscriptStatus, TranscriptInterpBase  # todo, move to helpers?


class SliceController(object):

    def __init__(self, db_path_in=None, db_path_sliced=None, sequences_path=None):
        self.db_path_in = db_path_in
        self.db_path_sliced = db_path_sliced
        self.sequences_path = sequences_path
        self.structured_genome = None
        self.engine = None
        self.session = None
        self.super_loci = []
        self.interval_trees = {}

    def get_one_annotated_genome(self):
        ags = self.session.query(annotations_orm.AnnotatedGenome).all()
        assert len(ags) == 1
        return ags[0]

    def copy_db(self):
        copyfile(self.db_path_in, self.db_path_sliced)

    def full_db_path(self):
        return 'sqlite:///{}'.format(self.db_path_sliced)

    def mk_session(self):
        if not os.path.exists(self.db_path_sliced):
            self.copy_db()
        self.engine = create_engine(self.full_db_path(), echo=False)
        annotations_orm.Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def load_annotations(self):
        sl_data = self.session.query(annotations_orm.SuperLocus).all()
        for sl in sl_data:
            super_locus = SuperLocusHandler()
            super_locus.add_data(sl)
            self.super_loci.append(super_locus)

    def load_sliced_seqs(self):
        if self.sequences_path is None:
            raise ValueError('Cannot load sequences from undefined (None) path')
        sg = sequences.StructuredGenome()
        sg.from_json(self.sequences_path)
        self.structured_genome = sg

    def fill_intervaltrees(self):
        for sl in self.super_loci:
            sl.load_to_intervaltree(self.interval_trees)

    def gen_slices(self):
        for seq in self.structured_genome.sequences:
            for slice in seq.slices:
                yield seq.meta_info.seqid, slice.start, slice.end, slice.slice_id

    def slice_annotations(self, annotated_genome):
        slices = list(self.gen_slices())  # todo, double check whether I can assume sorted
        self._slice_annotations_1way(slices, annotated_genome, is_plus_strand=True)
        slices.reverse()
        self._slice_annotations_1way(slices, annotated_genome, is_plus_strand=False)

    def _slice_annotations_1way(self, slices, annotated_genome, is_plus_strand):
        for seqid, start, end, slice_id in slices:
            seq_info = annotations_orm.SequenceInfo(annotated_genome=annotated_genome)
            coordinates = annotations_orm.Coordinates(seqid=seqid, start=start, end=end, sequence_info=seq_info)
            overlapping_super_loci = self.get_super_loci_frm_slice(seqid, start, end)
            for super_locus in overlapping_super_loci:
                super_locus.modify4slice(new_coords=coordinates, is_plus_strand=is_plus_strand,
                                         session=self.session)
            # todo, setup slice as coordinates w/ seq info in database
            # todo, get features & there by superloci in slice
            # todo, crop/reconcile superloci/transcripts/transcribeds/features with slice

    def get_super_loci_frm_slice(self, seqid, start, end):
        features = self.get_features_from_slice(seqid, start, end)
        super_loci = self.get_super_loci_frm_features(features)
        return super_loci

    def get_features_from_slice(self, seqid, start, end):
        if self.interval_trees == {}:
            raise ValueError('No, interval trees defined. The method .fill_intervaltrees must be called first')
        tree = self.interval_trees[seqid]
        intervals = tree[as_py_start(start):as_py_end(end)]
        features = [x.data for x in intervals]
        return features

    def get_super_loci_frm_features(self, features):
        super_loci = set()
        for feature in features:
            super_loci.add(feature.data.super_locus.handler)
        return super_loci

    def clean_slice(self):
        pass


class HandleMaker(object):
    def __init__(self, super_locus_handler):
        self.super_locus_handler = super_locus_handler
        self.handles = []

    @staticmethod
    def _get_paired_item(search4, search_col, return_col, nested_list):
        matches = [x[return_col] for x in nested_list if x[search_col] == search4]
        assert len(matches) == 1
        return matches[0]

    def make_all_handlers(self):
        self.handles = []
        sl = self.super_locus_handler.data
        datas = sl.transcribed_pieces + sl.translateds + sl.features
        for transcribed in sl.transcribeds:
            datas.append(transcribed)
            datas += transcribed.pairs

        for item in datas:
            self.handles.append(self._get_or_make_one_handler(item))

    def mk_n_append_handler(self, data):
        handler = self._get_or_make_one_handler(data)
        self.handles.append(handler)
        return handler

    def _get_or_make_one_handler(self, data):
        try:
            handler = data.hanlder
        except AttributeError:
            handler_type = self._get_handler_type(data)
            handler = handler_type()
            handler.add_data(data)
        return handler

    def _get_handler_type(self, old_data):
        key = [(SuperLocusHandler, annotations_orm.SuperLocus),
               (TranscribedHandler, annotations_orm.Transcribed),
               (TranslatedHandler, annotations_orm.Translated),
               (TranscribedPieceHandler, annotations_orm.TranscribedPiece),
               (FeatureHandler, annotations_orm.Feature),
               (UpstreamFeatureHandler, annotations_orm.UpstreamFeature),
               (DownstreamFeatureHandler, annotations_orm.DownstreamFeature),
               (UpDownPairHandler, annotations_orm.UpDownPair)]

        return self._get_paired_item(type(old_data), search_col=1, return_col=0, nested_list=key)


class SequenceInfoHandler(annotations.SequenceInfoHandler):
    def processing_set(self, sess):
        return sess.query(
            slice_dbmods.SequenceInfoSets
        ).filter(
            slice_dbmods.SequenceInfoSets.id == self.data.id
        ).first()

    def processing_set_val(self, sess):
        si_set_obj = self.processing_set(sess)
        if si_set_obj is None:
            return None
        else:
            return si_set_obj.processing_set.value

    def set_processing_set(self, sess, processing_set):
        current = self.processing_set(sess)
        if current is None:
            current = slice_dbmods.SequenceInfoSets(sequence_info=self.data, processing_set=processing_set)
        else:
            current.processing_set = processing_set
        sess.add(current)
        sess.commit()


class SuperLocusHandler(annotations.SuperLocusHandler):
    def __init__(self):
        super().__init__()
        self.handler_holder = HandleMaker(self)

    def make_all_handlers(self):
        self.handler_holder.make_all_handlers()

    def load_to_intervaltree(self, trees):
        features = self.data.features
        for f in features:
            feature = FeatureHandler()  # recreate feature handler post load (todo, mv elsewhere so it's always done?)
            feature.add_data(f)
            feature.load_to_intervaltree(trees)

    def modify4slice(self, new_coords, is_plus_strand, session):
        print('modifying sl {} for new slice {}:{}-{},  is plus: {}'.format(
            self.data.id, new_coords.seqid, new_coords.start, new_coords.end, is_plus_strand))
        for transcribed in self.data.transcribeds:
            trimmer = TranscriptTrimmer(transcript=transcribed.handler, sess=session)
            trimmer.modify4new_slice(new_coords=new_coords, is_plus_strand=is_plus_strand)


class TranscribedHandler(annotations.TranscribedHandler):
    def reconcile_with_slice(self, seqid, start, end):
        pass


class TranslatedHandler(annotations.TranslatedHandler):
    def reconcile_translated_with_slice(self, seqid, start, end):
        pass


class TranscribedPieceHandler(annotations.TranscribedPieceHandler):
    pass


class FeatureHandler(annotations.FeatureHandler):
    def load_to_intervaltree(self, trees):
        seqid = self.data.coordinates.seqid
        if seqid not in trees:
            trees[seqid] = intervaltree.IntervalTree()
        tree = trees[seqid]
        tree[self.py_start:self.py_end] = self


class UpstreamFeatureHandler(annotations.UpstreamFeatureHandler):
    pass


class DownstreamFeatureHandler(annotations.DownstreamFeatureHandler):
    pass


class UpDownPairHandler(annotations.UpDownPairHandler):
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


class PositionInterpreter(object):

    def __init__(self, feature, prev_feature, slice_coordinates, is_plus_strand):
        self.feature = feature
        self.prev_feature = prev_feature
        self.slice_coordinates = slice_coordinates
        self.is_plus_strand = is_plus_strand
        # precalculate shared data
        if is_plus_strand:
            self.sign = 1
            self.f_upstream = feature.start
            self.f_downstream = feature.end
            self.c_upstream = slice_coordinates.start
            self.c_downstream = slice_coordinates.end
        else:
            self.sign = -1
            self.f_upstream = feature.end
            self.f_downstream = feature.start
            self.c_upstream = slice_coordinates.end
            self.c_downstream = slice_coordinates.start

    def is_detached(self):
        out = False
        if self.slice_coordinates.seqid != self.feature.coordinates.seqid:
            out = True
        elif self.is_plus_strand != self.feature.is_plus_strand:
            out = True
        return out

    def is_upstream(self):
        return self.sign * (self.c_upstream - self.f_downstream) > 0

    def is_downstream(self):
        return self.sign * (self.f_upstream - self.c_downstream) > 0

    def overlaps_downstream(self):
        f_upstream_contained = self.slice_coordinates.start <= self.f_upstream <= self.slice_coordinates.end
        f_downstream_is_downstream = self.sign * (self.f_downstream - self.c_downstream) > 0
        return f_upstream_contained and f_downstream_is_downstream

    def overlaps_upstream(self):
        f_downstream_contained = self.slice_coordinates.start <= self.f_downstream <= self.slice_coordinates.end
        f_upstream_is_upstream = self.sign * (self.c_upstream - self.f_upstream) > 0
        return f_downstream_contained and f_upstream_is_upstream

    def is_contained(self):
        start_contained = self.slice_coordinates.start <= self.feature.start <= self.slice_coordinates.end
        end_contained = self.slice_coordinates.start <= self.feature.end <= self.slice_coordinates.end
        return start_contained and end_contained

    def just_passed_downstream(self):
        if self.prev_feature is None:
            out = False
        else:
            if self.is_plus_strand:
                if self.prev_feature.end <= self.c_downstream <= self.feature.start:
                    out = True
                else:
                    out = False
            else:
                if self.feature.end <= self.c_downstream <= self.prev_feature.start:
                    out = True
                else:
                    out = False
        return out


class IndecipherableLinkageError(Exception):
    pass


class NoFeaturesInSliceError(Exception):
    pass


class StepHolder(object):
    def __init__(self, features=None, status=None, old_piece=None, replacement_piece=None):
        self.features = features
        self.status = status
        self.old_piece = old_piece
        self.replacement_piece = replacement_piece

    @property
    def example_feature(self):
        if self.features is None:
            return None
        else:
            return self.features[0]

    def set_as_previous_of(self, new_step_holder):
        # because we only want to use previous features from the _same_ piece
        if self.old_piece is not new_step_holder.old_piece:
            self.features = None


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript, sess):
        super().__init__(transcript)
        self.session = sess
        self.handlers = []

    def new_handled_data(self, template=None, new_type=annotations_orm.Feature, **kwargs):
        data = new_type()
        # todo, simplify below...
        handler = self.transcript.data.super_locus.handler.handler_holder.mk_n_append_handler(data)
        if template is not None:
            template.handler.fax_all_attrs_to_another(another=handler)

        for key in kwargs:
            handler.set_data_attribute(key, kwargs[key])
        return handler

    def transition_5p_to_3p(self):
        status = TranscriptStatus()
        for piece in self.sort_pieces():
            piece_features = self.sorted_features(piece)
            for aligned_features in self.stack_matches(piece_features):
                self.update_status(status, aligned_features)
                yield aligned_features, copy.deepcopy(status), piece

    def _transition_5p_to_3p_with_new_pieces(self):
        transition_gen = self.transition_5p_to_3p()
        aligned_features, status, prev_piece = next(transition_gen)
        new_piece = self.mk_new_piece()
        yield aligned_features, status, prev_piece, new_piece

        for aligned_features, status, piece in transition_gen:
            if piece is not prev_piece:
                new_piece = self.mk_new_piece()
            yield aligned_features, status, piece, new_piece
            prev_piece = piece

    def transition_5p_to_3p_with_new_pieces(self):
        for features, status, old_piece, replacement_piece in self._transition_5p_to_3p_with_new_pieces():
            yield StepHolder(features=features, status=status, old_piece=old_piece, replacement_piece=replacement_piece)

    def mk_new_piece(self):
        new_piece = annotations_orm.TranscribedPiece(super_locus=self.transcript.data.super_locus,
                                                     transcribed=self.transcript.data)
        new_handler = TranscribedPieceHandler()
        new_handler.add_data(new_piece)
        self.session.add(new_piece)
        self.session.commit()
        self.handlers.append(new_handler)
        return new_piece

    def update_status(self, status, aligned_features):
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

    def modify4new_slice(self, new_coords, is_plus_strand=True):
        print('mod4slice, transcribed: {}, {}'.format(self.transcript.data.id, self.transcript.data.given_id))
        seen_one_overlap = False
        transition_gen = self.transition_5p_to_3p_with_new_pieces()
        previous_step = StepHolder()
        for current_step in transition_gen:
            # if we've switched pieces in transcript, reset the previous features seen on piece to None
            previous_step.set_as_previous_of(current_step)

            f0 = current_step.example_feature  # take first as all "aligned" features have the same coordinates
            position_interp = PositionInterpreter(f0, previous_step.example_feature, new_coords, is_plus_strand)
            # before or detached coordinates (already handled or good as-is, at least for now)
            if position_interp.is_detached():
                pass
            elif position_interp.is_upstream():
                pass
            # it should never overlap start (because this should have been handled and split already)
            elif position_interp.overlaps_upstream():
                seen_one_overlap = True
                raise ValueError('unhandled straddling of upstream boarder')
            # within new_coords -> swap coordinates
            elif position_interp.is_contained():
                seen_one_overlap = True
                for f in current_step.features:
                    f.coordinates = new_coords
                    self.swap_piece(feature_handler=f.handler, new_piece=current_step.replacement_piece,
                                    old_piece=current_step.old_piece)
            # handle feature [  |  ] straddling end of coordinates
            elif position_interp.overlaps_downstream():
                seen_one_overlap = True
                # make new UpDownLink and status features to handle split
                upstream_half = self.split_feature_downstream_border(
                    new_coords=new_coords, new_piece=current_step.replacement_piece,
                    old_piece=current_step.old_piece, is_plus_strand=is_plus_strand, feature=f0)
                self.set_status_downstream_border(new_coords=new_coords, new_piece=current_step.replacement_piece,
                                                  old_coords=f0.coordinates, template_feature=upstream_half.data,
                                                  old_piece=current_step.old_piece, is_plus_strand=is_plus_strand,
                                                  status=current_step.status)


            # handle pass end of coordinates between previous and current feature, [p] | [f]
            elif position_interp.just_passed_downstream():
                if not seen_one_overlap:
                    raise NoFeaturesInSliceError("seen no features overlapping or contained in new piece '{}', can't "
                                                 "set downstream pass.\n  Last feature: '{}'\n  "
                                                 "Current feature: '{}'".format(
                                                     current_step.replacement_piece, previous_step.example_feature, f0,
                                                 ))

                self.set_status_downstream_border(new_coords=new_coords, old_coords=f0.coordinates,
                                                  is_plus_strand=is_plus_strand,
                                                  template_feature=previous_step.example_feature,
                                                  status=previous_step.status, old_piece=current_step.old_piece,
                                                  new_piece=current_step.replacement_piece)

            elif position_interp.is_downstream():
                pass  # will get to this at next slice

            else:
                print('ooops', f0)
                raise AssertionError('this code should be unreachable...? Check what is up!')

            # and step
            previous_step = copy.copy(current_step)

        # clean up any unused or abandoned pieces
        for piece in self.transcript.data.transcribed_pieces:
            if piece.features == []:
                self.session.delete(piece)
        self.session.commit()

        if not seen_one_overlap:
            raise NoFeaturesInSliceError("Saw no features what-so-ever in new_coords {} for transcript {}".format(
                new_coords, self.transcript.data
            ))

    def get_rel_feature_position(self, feature, prev_feature, new_coords, is_plus_strand):
        pass

    @staticmethod
    def swap_piece(feature_handler, new_piece, old_piece):
        try:
            feature_handler.de_link(old_piece.handler)
        except ValueError as e:
            coords_old = [x.coordinates for x in old_piece.features]
            coords_new = [x.coordinates for x in new_piece.features]
            print('trying to swap feature: {}\n  from: {} (coords: {})\n  to: {} (coords: {})'.format(
                feature_handler.data, old_piece, set(coords_old), new_piece, set(coords_new)))
            raise e
        feature_handler.link_to(new_piece.handler)

    def set_status_downstream_border(self, new_coords, old_coords, is_plus_strand, template_feature, status, new_piece,
                                     old_piece):
        if is_plus_strand:
            up_at = new_coords.end
            down_at = new_coords.end + 1
        else:
            up_at = new_coords.start
            down_at = new_coords.start - 1

        at_least_one_link = False
        if status.genic:
            self._set_one_status_at_border(old_coords, template_feature, type_enums.IN_RAW_TRANSCRIPT, up_at, down_at,
                                           new_piece, old_piece)
            at_least_one_link = True
        if status.in_intron:
            self._set_one_status_at_border(old_coords, template_feature, type_enums.IN_INTRON, up_at, down_at,
                                           new_piece, old_piece)
            at_least_one_link = True
        if status.seen_start and not status.seen_stop:
            self._set_one_status_at_border(old_coords, template_feature, type_enums.IN_TRANSLATED_REGION, up_at,
                                           down_at, new_piece, old_piece)
            at_least_one_link = True
        if status.in_trans_intron:
            self._set_one_status_at_border(old_coords, template_feature, type_enums.IN_TRANS_INTRON, up_at, down_at,
                                           new_piece, old_piece)
            at_least_one_link = True

        if status.erroneous:
            self._set_one_status_at_border(old_coords, template_feature, type_enums.IN_ERROR, up_at, down_at, new_piece,
                                           old_piece)
            at_least_one_link = True
        if not at_least_one_link:
            raise ValueError('Expected some sort of know status to set the status at border')

    def _set_one_status_at_border(self, old_coords, template_feature, status_type, up_at, down_at, new_piece,
                                  old_piece):
        print('template feature {}, pieces {}'.format(template_feature, template_feature.transcribed_pieces))
        assert new_piece in template_feature.transcribed_pieces
        upstream = self.new_handled_data(template_feature, annotations_orm.UpstreamFeature, start=up_at, end=up_at,
                                         given_id=None, type=status_type)
        downstream = self.new_handled_data(template_feature, annotations_orm.DownstreamFeature, start=down_at,
                                           end=down_at, given_id=None, type=status_type, coordinates=old_coords)
        # swap piece back to previous, as downstream is outside of new_coordinates (slice), and will be ran through
        # modify4new_slice once more and get it's final coordinates/piece then
        self.swap_piece(feature_handler=downstream, new_piece=old_piece, old_piece=new_piece)
        self.new_handled_data(new_type=annotations_orm.UpDownPair, upstream=upstream.data,
                              downstream=downstream.data, transcribed=self.transcript.data)
        self.session.add_all([upstream.data, downstream.data])
        self.session.commit()  # todo, figure out what the real rules are for committing, bc slower, but less buggy?

    def split_feature_downstream_border(self, new_coords, new_piece, old_piece, is_plus_strand, feature):
        if is_plus_strand:
            before_border = self.new_handled_data(template=feature, new_type=annotations_orm.Feature,
                                                  end=new_coords.end, coordinates=new_coords)
            self.swap_piece(feature_handler=before_border, new_piece=new_piece, old_piece=old_piece)
            feature.start = new_coords.end + 1
        else:
            before_border = self.new_handled_data(template=feature, new_type=annotations_orm.Feature,
                                                  start=new_coords.start, coordinates=new_coords)
            self.swap_piece(feature_handler=before_border, new_piece=new_piece, old_piece=old_piece)
            feature.end = new_coords.start - 1
        self.session.add(before_border.data)
        self.session.commit()
        return before_border

    @staticmethod
    def sorted_features(piece):
        features = piece.features
        # confirm strand & seqid
        assert all([f.coordinates == features[0].coordinates for f in features])
        assert all([f.is_plus_strand == features[0].is_plus_strand for f in features])
        features = sorted(features, key=lambda x: x.pos_cmp_key())
        if not features[0].is_plus_strand:
            features.reverse()
        return features

    def sort_pieces(self):
        pieces = self.transcript.data.transcribed_pieces
        # start with one piece, extend until both ends are reached
        ordered_pieces = pieces[0:1]
        print(ordered_pieces, 'of', pieces,'start')
        self._extend_to_end(ordered_pieces, downstream=True)
        self._extend_to_end(ordered_pieces, downstream=False)
        assert set(ordered_pieces) == set(pieces), "{} != {}".format(set(ordered_pieces), set(pieces))
        return ordered_pieces

    def _extend_to_end(self, ordered_pieces, downstream=True, filter_fn=None):
        if downstream:
            next_fn = self.get_downstream_link
            latest_i = -1
            attr = 'downstream'
        else:
            next_fn = self.get_upstream_link
            latest_i = 0
            attr = 'upstream'

        while True:
            nextlink = next_fn(current_piece=ordered_pieces[latest_i])
            if nextlink is None:
                break
            nextstream = nextlink.__getattribute__(attr)
            print(nextstream, 'nextstream')
            print(nextstream.transcribed_pieces)

            nextpiece = self._get_one_piece_from_stream(nextstream)
            if nextpiece in ordered_pieces:
                raise IndecipherableLinkageError('Circular linkage inserting {} into {}'.format(nextpiece,
                                                                                                ordered_pieces))
            else:
                self._extend_by_one(ordered_pieces, nextpiece, downstream)

    @staticmethod
    def _extend_by_one(ordered_pieces, new, downstream=True):
        if downstream:
            ordered_pieces.append(new)
        else:
            ordered_pieces.insert(0, new)

    def _get_one_piece_from_stream(self, stream):
        pieces = self.transcript.data.transcribed_pieces
        matches = [x for x in stream.transcribed_pieces if x in pieces]
        assert len(matches) == 1, 'len(matches) != 1, matches: {}'.format(matches)  # todo; can we guarantee this?
        return matches[0]

    def get_upstream_link(self, current_piece):
        downstreams = self.session.query(annotations_orm.DownstreamFeature).all()
        # DownstreamFeature s of this pice
        downstreams_current = [x for x in downstreams if current_piece in x.transcribed_pieces]
        links = self._find_matching_links(updown_candidates=downstreams_current, get_upstreams=True)
        return self._links_list2link(links, direction='upstream', current_piece=current_piece)

    def get_downstream_link(self, current_piece):
        upstreams = self.session.query(annotations_orm.UpstreamFeature).all()
        upstreams_current = [x for x in upstreams if current_piece in x.transcribed_pieces]
        links = self._find_matching_links(updown_candidates=upstreams_current, get_upstreams=False)
        return self._links_list2link(links, direction='downstream', current_piece=current_piece)

    def _find_matching_links(self, updown_candidates, get_upstreams=True):
        links = []
        pairs = self.transcript.data.pairs
        for cand in updown_candidates:
            if get_upstreams:
                links += [x for x in pairs if x.downstream == cand]
            else:
                links += [x for x in pairs if x.upstream == cand]
        return links

    def _links_list2link(self, links, direction, current_piece):
        stacked = self.stack_matches(links)
        collapsed = [x[0] for x in stacked]

        if len(collapsed) == 0:
            return None
        elif len(collapsed) == 1:
            return collapsed[0]
        else:
            raise IndecipherableLinkageError("Multiple possible within-transcript {} links found from {}, ({})".format(
                direction, current_piece, collapsed
            ))

    def sort_all(self):
        out = []
        for piece in self.sort_pieces():
            out.append(self.sorted_features(piece))
        return out

    @staticmethod
    def stack_matches(features):
        ifeatures = iter(features)
        try:
            prev = next(ifeatures)
        except StopIteration:
            return
        current = [prev]
        for feature in ifeatures:
            if feature.pos_cmp_key() == prev.pos_cmp_key():
                current.append(feature)
            else:
                yield current
                current = [feature]
            prev = feature
        yield current
        return
