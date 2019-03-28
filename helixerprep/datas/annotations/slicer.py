"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
from shutil import copyfile
import intervaltree
import copy
import logging

import geenuff
from helixerprep.datas.annotations import slice_dbmods

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import bindparam

import os
from helixerprep.datas import sequences
#from annotations import TranscriptInterpBase
TranscriptInterpBase = geenuff.api.TranscriptInterpBase


class CoreQueue(object):
    def __init__(self, session, engine):
        self.engine = engine
        self.session = session
        # updates
        self.piece_swaps = []  # {'piece_id_old':, 'piece_id_new':, 'feat_id':}
        self.coord_swaps = []  # {'feat_id':, 'coordinate_id_new':}
        # insertions
        self.transcribed_pieces = []
        self.upstream_features = []
        self.downstream_features = []
        self.up_down_pairs = []
        self.association_transcribeds_to_features = []
        self.association_translateds_to_features = []

    @property
    def piece_swaps_update(self):
        out = geenuff.orm.association_transcribeds_to_features.update().\
            where(geenuff.orm.association_transcribeds_to_features.c.transcribed_piece_id == bindparam('piece_id_old')).\
            where(geenuff.orm.association_transcribeds_to_features.c.feature_id == bindparam('feat_id')).\
            values(transcribed_piece_id=bindparam('piece_id_new'))
        print(out, 'swap call')
        return out

    @property
    def coord_swaps_update(self):
        return geenuff.orm.Feature.__table__.update().\
            where(geenuff.orm.Feature.id == bindparam('feat_id')).\
            values(coordinate_id=bindparam('coordinate_id_new'))

    @property
    def actions_lists(self):
        return [(self.piece_swaps_update, self.piece_swaps),
                (self.coord_swaps_update, self.coord_swaps)]

    def execute_so_far(self):
        print(self.piece_swaps)
        conn = self.engine.connect()
        for action, a_list in self.actions_lists:
            if a_list:
                conn.execute(action, a_list)
                del a_list[:]
        self.session.commit()

        # and remove any abandoned pieces
        #todel = []
        #for piece in self.session.query(geenuff.orm.TranscribedPiece).all():
        #    if not piece.features:
        #        todel.append({'piece_id': piece.id})
        #del_cmd = geenuff.orm.TranscribedPiece.__table__.delete().\
        #    where(geenuff.orm.TranscribedPiece.id == bindparam('piece_id'))
        #if todel:
        #    conn.execute(del_cmd, todel)


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
        self.core_queue = None

    def get_one_annotated_genome(self):
        ags = self.session.query(geenuff.orm.AnnotatedGenome).all()
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
        geenuff.orm.Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.core_queue = CoreQueue(self.session, self.engine)

    def load_annotations(self):
        sl_data = self.session.query(geenuff.orm.SuperLocus).all()
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
        """artificially slices annotated genome to match sequence slices and adjusts transcripts as appropriate"""
        slices = list(self.gen_slices())  # todo, double check whether I can assume sorted
        self._slice_annotations_1way(slices, annotated_genome, is_plus_strand=True)
        slices.reverse()
        self._slice_annotations_1way(slices, annotated_genome, is_plus_strand=False)

    def _slice_annotations_1way(self, slices, annotated_genome, is_plus_strand):
        for seqid, start, end, slice_id in slices:
            seq_info = geenuff.orm.SequenceInfo(annotated_genome=annotated_genome)
            coordinates = geenuff.orm.Coordinates(seqid=seqid, start=start, end=end, sequence_info=seq_info)
            self.session.add(coordinates)
            self.session.commit()
            overlapping_super_loci = self.get_super_loci_frm_slice(seqid, start, end, is_plus_strand=is_plus_strand)
            for super_locus in overlapping_super_loci:
                super_locus.make_all_handlers()
                super_locus.modify4slice(new_coords=coordinates, is_plus_strand=is_plus_strand,
                                         session=self.session, trees=self.interval_trees, core_queue=self.core_queue)
            self.core_queue.execute_so_far()
            # todo, setup slice as coordinates w/ seq info in database
            # todo, get features & there by superloci in slice
            # todo, crop/reconcile superloci/transcripts/transcribeds/features with slice

    def get_super_loci_frm_slice(self, seqid, start, end, is_plus_strand):
        features = self.get_features_from_slice(seqid, start, end, is_plus_strand)
        super_loci = self.get_super_loci_frm_features(features)
        return super_loci

    def get_features_from_slice(self, seqid, start, end, is_plus_strand):
        if self.interval_trees == {}:
            raise ValueError('No, interval trees defined. The method .fill_intervaltrees must be called first')
        tree = self.interval_trees[seqid]
        intervals = tree[start:end]
        print('start, end', start, end)
        features = [x.data for x in intervals if x.data.data.is_plus_strand == is_plus_strand]
        return features

    def get_super_loci_frm_features(self, features):
        super_loci = set()
        for feature in features:
            super_loci.add(feature.data.super_locus.handler)
        return super_loci

    def clean_slice(self):
        pass


class HandleMaker(geenuff.api.HandleMaker):
    # redefine to get handlers from slicer, here
    def _get_handler_type(self, old_data):
        key = [(SuperLocusHandler, geenuff.orm.SuperLocus),
               (TranscribedHandler, geenuff.orm.Transcribed),
               (TranslatedHandler, geenuff.orm.Translated),
               (TranscribedPieceHandler, geenuff.orm.TranscribedPiece),
               (FeatureHandler, geenuff.orm.Feature),
               (UpstreamFeatureHandler, geenuff.orm.UpstreamFeature),
               (DownstreamFeatureHandler, geenuff.orm.DownstreamFeature),
               (UpDownPairHandler, geenuff.orm.UpDownPair)]

        return self._get_paired_item(type(old_data), search_col=1, return_col=0, nested_list=key)


class SequenceInfoHandler(geenuff.api.SequenceInfoHandler):
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


class SuperLocusHandler(geenuff.api.SuperLocusHandler):
    def __init__(self):
        super().__init__()
        self.handler_holder = HandleMaker(self)

    def make_all_handlers(self):
        self.handler_holder.make_all_handlers()

    def load_to_intervaltree(self, trees):
        self.make_all_handlers()
        features = self.data.features
        for f in features:
            feature = f.handler
            feature.load_to_intervaltree(trees)

    def modify4slice(self, new_coords, is_plus_strand, session, core_queue, trees=None):  # todo, can trees then be None?
        logging.debug('modifying sl {} for new slice {}:{}-{},  is plus: {}'.format(
            self.data.id, new_coords.seqid, new_coords.start, new_coords.end, is_plus_strand))
        for transcribed in self.data.transcribeds:
            trimmer = TranscriptTrimmer(transcript=transcribed.handler, super_locus=self, sess=session,
                                        core_queue=core_queue)
            try:
                trimmer.modify4new_slice(new_coords=new_coords, is_plus_strand=is_plus_strand, trees=trees)
            except NoFeaturesInSliceError:
                # temporary patch to not die on case where gene, but not _this_ transcript overlaps, todo, fix!
                # but try and double check no-overlap first
                for piece in transcribed.transcribed_pieces:
                    for feature in piece.features:
                        # ignore all features on the opposite strand
                        if is_plus_strand == feature.is_plus_strand:
                            if is_plus_strand:
                                assert not (new_coords.start <= feature.position <= new_coords.end)
                            else:
                                assert not (new_coords.start - 1 <= feature.position <= new_coords.end - 1)


# todo, switch back to simple import if not changing...
class TranscribedHandler(geenuff.api.TranscribedHandler):
    pass


class TranslatedHandler(geenuff.api.TranslatedHandler):
    pass


class TranscribedPieceHandler(geenuff.api.TranscribedPieceHandler):
    pass


# todo, there is probably a nicer way to accomplish the following with multi inheritance...
def load_to_intervaltree(obj, trees):
    seqid = obj.data.coordinates.seqid
    if seqid not in trees:
        trees[seqid] = intervaltree.IntervalTree()
    tree = trees[seqid]
    py_start = obj.data.position
    tree[py_start:(py_start + 1)] = obj


class FeatureHandler(geenuff.api.FeatureHandler):
    def load_to_intervaltree(self, trees):
        load_to_intervaltree(self, trees)


class UpstreamFeatureHandler(geenuff.api.UpstreamFeatureHandler):
    def load_to_intervaltree(self, trees):
        load_to_intervaltree(self, trees)


class DownstreamFeatureHandler(geenuff.api.DownstreamFeatureHandler):
    def load_to_intervaltree(self, trees):
        load_to_intervaltree(self, trees)


class UpDownPairHandler(geenuff.api.UpDownPairHandler):
    pass


class FeatureVsCoords(object):
    """positions a feature (upstream, downstream, contained, or detached) relative to some coordinates"""

    def __init__(self, feature, slice_coordinates, is_plus_strand):
        self.feature = feature
        self.slice_coordinates = slice_coordinates
        self.is_plus_strand = is_plus_strand
        # precalculate shared data
        if is_plus_strand:
            self.sign = 1
        else:
            self.sign = -1

        # inclusive coordinates (classic python in for slice)
        if feature.bearing.value in [geenuff.types.START, geenuff.types.OPEN_STATUS, geenuff.types.POINT]:
            self.c_py_start = slice_coordinates.start
            self.c_py_end = slice_coordinates.end
        else:  # exclusive coordinates for a closing feature (not 1st upstream bp, one after downstream)
            self.c_py_start = slice_coordinates.start + self.sign
            self.c_py_end = slice_coordinates.end + self.sign

        if is_plus_strand:
            self.c_py_upstream = self.c_py_start
            self.c_py_downstream = self.c_py_end
        else:
            self.c_py_upstream = self.c_py_end
            self.c_py_downstream = self.c_py_start

    def is_detached(self):
        out = False
        if self.slice_coordinates.seqid != self.feature.coordinates.seqid:
            out = True
        elif self.is_plus_strand != self.feature.is_plus_strand:
            out = True
        return out

    def _is_lower(self):
        return self.c_py_start - self.feature.position > 0

    def _is_higher(self):
        return self.feature.position - self.c_py_end >= 0

    def is_upstream(self):
        if self.is_plus_strand:
            return self._is_lower()
        else:
            return self._is_higher()

    def is_downstream(self):
        if self.is_plus_strand:
            return self._is_higher()
        else:
            return self._is_lower()

    def is_contained(self):
        start_contained = self.c_py_start <= self.feature.position < self.c_py_end
        return start_contained


class PositionInterpreter(object):
    def __init__(self, feature, prev_feature, slice_coordinates, is_plus_strand):
        self.feature_vs_coord = FeatureVsCoords(feature, slice_coordinates, is_plus_strand)
        if prev_feature is not None:  # this marks a disjointed (diff piece) previous feature, todo cleanup/mk explicit
            self.previous_vs_coord = FeatureVsCoords(prev_feature, slice_coordinates, is_plus_strand)
        else:
            self.previous_vs_coord = None
        self.slice_coordinates = slice_coordinates
        self.is_plus_strand = is_plus_strand

    def is_detached(self):
        return self.feature_vs_coord.is_detached()

    def is_upstream(self):
        return self.feature_vs_coord.is_upstream()

    def is_downstream(self):
        return self.feature_vs_coord.is_downstream()

    def is_contained(self):
        return self.feature_vs_coord.is_contained()

    def just_passed_downstream(self):
        if self.previous_vs_coord is None:
            return False
        else:
            return self.feature_vs_coord.is_downstream() and self.previous_vs_coord.is_contained()


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
    def __init__(self, transcript, super_locus, sess, core_queue):
        super().__init__(transcript, super_locus=super_locus, session=sess)
        self.core_queue = core_queue
        #self.session = sess
        self.handlers = []

    def new_handled_data(self, template=None, new_type=geenuff.orm.Feature, **kwargs):
        data = new_type()
        # todo, simplify below...
        handler = self.transcript.data.super_locus.handler.handler_holder.mk_n_append_handler(data)
        if template is not None:
            template.handler.fax_all_attrs_to_another(another=handler)

        for key in kwargs:
            handler.set_data_attribute(key, kwargs[key])
        return handler

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
        new_piece = geenuff.orm.TranscribedPiece(super_locus=self.transcript.data.super_locus,
                                                 transcribed=self.transcript.data)
        new_handler = TranscribedPieceHandler()
        new_handler.add_data(new_piece)
        self.session.add(new_piece)
        self.session.commit()
        self.handlers.append(new_handler)
        return new_piece

    def modify4new_slice(self, new_coords, is_plus_strand=True, trees=None):
        """adjusts features and pieces of transcript to be artificially split across a new sub-coordinate"""
        if trees is None:
            trees = {}
        logging.debug('mod4slice, transcribed: {}, {}'.format(self.transcript.data.id, self.transcript.data.given_id))
        seen_one_overlap = False
        transition_gen = self.transition_5p_to_3p_with_new_pieces()
        previous_step = StepHolder()
        piece_at_border = None
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
            # within new_coords -> swap coordinates
            elif position_interp.is_contained():
                seen_one_overlap = True
                for f in current_step.features:
                    coord_swap = {'feat_id': f.id, 'coordinate_id_new': new_coords.id}
                    print(coord_swap, 'coord_swap')
                    self.core_queue.coord_swaps.append(coord_swap)
                    #f.coordinates = new_coords
                    #self.swap_piece(feature_handler=f.handler, new_piece=current_step.replacement_piece,
                    #                old_piece=current_step.old_piece)

            # handle pass end of coordinates between previous and current feature, [p] | [f]
            elif position_interp.just_passed_downstream():
                piece_at_border = current_step.old_piece
                # todo, make new_piece_after_border _here_ not in transition gen...
                new_piece_after_border = current_step.replacement_piece
                self.core_queue.execute_so_far()  # todo, rm from here once everything uses core
                if not seen_one_overlap:
                    raise NoFeaturesInSliceError("seen no features overlapping or contained in new piece '{}', can't "
                                                 "set downstream pass.\n  Last feature: '{}'\n  "
                                                 "Current feature: '{}'".format(
                                                     current_step.replacement_piece, previous_step.example_feature, f0,
                                                 ))
                print(new_coords, self.transcript.data.given_id, previous_step.status)
                self.set_status_downstream_border(new_coords=new_coords, old_coords=f0.coordinates,
                                                  is_plus_strand=is_plus_strand,
                                                  template_feature=previous_step.example_feature,
                                                  status=previous_step.status, old_piece=current_step.old_piece,
                                                  new_piece=new_piece_after_border, trees=trees)
                # the current features too, need the new_piece
                for f in current_step.features:
                    to_swap = {'piece_id_old': piece_at_border.id,
                               'piece_id_new': new_piece_after_border.id,
                               'feat_id': f.id}
                    self.core_queue.piece_swaps.append(to_swap)

            elif position_interp.is_downstream():
                if piece_at_border is not None:
                    if current_step.old_piece is piece_at_border:
                        print('real downstream')
                        # todo, swap from old piece to new_piece_after_border
                        for f in current_step.features:
                            to_swap = {'piece_id_old': piece_at_border.id,
                                       'piece_id_new': new_piece_after_border.id,
                                       'feat_id': f.id}
                            self.core_queue.piece_swaps.append(to_swap)
            else:
                print('ooops', f0)
                raise AssertionError('this code should be unreachable...? Check what is up!')

            # and step
            previous_step = copy.copy(current_step)

        # apply batch changes
        #self.core_queue.execute_so_far()

        # clean up any unused or abandoned pieces
        #for piece in self.transcript.data.transcribed_pieces:
        #    if piece.features == []:
        #        self.session.delete(piece)
        #self.session.commit()

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
                                     old_piece, trees):
        if is_plus_strand:
            up_at = down_at = new_coords.end
        else:
            up_at = down_at = new_coords.start - 1  # -1 so it's exclusive close (incl next open), in reverse direction

        at_least_one_link = False
        if status.genic:
            self._set_one_status_at_border(old_coords, template_feature, geenuff.types.TRANSCRIBED, up_at, down_at,
                                           new_piece, old_piece, trees)
            at_least_one_link = True
        if status.in_intron:
            self._set_one_status_at_border(old_coords, template_feature, geenuff.types.INTRON, up_at, down_at,
                                           new_piece, old_piece, trees)
            at_least_one_link = True
        if status.in_translated_region:
            self._set_one_status_at_border(old_coords, template_feature, geenuff.types.CODING, up_at,
                                           down_at, new_piece, old_piece, trees)
            at_least_one_link = True
        if status.in_trans_intron:
            self._set_one_status_at_border(old_coords, template_feature, geenuff.types.TRANS_INTRON, up_at, down_at,
                                           new_piece, old_piece, trees)
            at_least_one_link = True

        if status.erroneous:
            self._set_one_status_at_border(old_coords, template_feature, geenuff.types.ERROR, up_at, down_at, new_piece,
                                           old_piece, trees)
            at_least_one_link = True
        if not at_least_one_link:
            print('dying here.....')
            print('old piece: {}'.format(old_piece))
            print('old coords: {}'.format(old_coords))
            print(':::: {}\n'.format(old_piece.features))

            print('new piece: {}'.format(new_piece))
            print('new coords: {}'.format(new_coords))
            print(':::: {}'.format(new_piece.features))
            raise ValueError('Expected some sort of known status to set the status at border')

    def _set_one_status_at_border(self, old_coords, template_feature, status_type, up_at, down_at, new_piece,
                                  old_piece, trees):
        assert old_piece in template_feature.transcribed_pieces, "old id: ({}) not in feature {}'s pieces: {}".format(
            old_piece.id, template_feature.id, template_feature.transcribed_pieces)
        upstream = self.new_handled_data(template_feature, geenuff.orm.UpstreamFeature, position=up_at,
                                         given_id=None, type=status_type, bearing=geenuff.types.CLOSE_STATUS)
        upstream.load_to_intervaltree(trees)
        downstream = self.new_handled_data(template_feature, geenuff.orm.DownstreamFeature, position=down_at,
                                           given_id=None, type=status_type, coordinates=old_coords,
                                           bearing=geenuff.types.OPEN_STATUS)
        downstream.load_to_intervaltree(trees)
        # swap the new downstream feature to have new piece (which will be shared with further downstream features)
        self.swap_piece(feature_handler=downstream, new_piece=new_piece, old_piece=old_piece)
        self.new_handled_data(new_type=geenuff.orm.UpDownPair, upstream=upstream.data,
                              downstream=downstream.data, transcribed=self.transcript.data)
        self.session.add_all([upstream.data, downstream.data])
        self.session.commit()  # todo, figure out what the real rules are for committing, bc slower, but less buggy?


