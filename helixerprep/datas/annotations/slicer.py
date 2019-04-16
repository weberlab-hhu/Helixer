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
        table = geenuff.orm.association_transcribed_piece_to_feature
        out = table.update().\
            where(table.c.transcribed_piece_id == bindparam('piece_id_old')).\
            where(table.c.feature_id == bindparam('feat_id')).\
            values(transcribed_piece_id=bindparam('piece_id_new'))
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
        conn = self.engine.connect()
        for action, a_list in self.actions_lists:
            if a_list:
                conn.execute(action, a_list)
                del a_list[:]
        self.session.commit()


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
        ags = self.session.query(geenuff.orm.Genome).all()
        assert len(ags) == 1, "found # genomes != 1: {}".format(ags)
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

    def _slice_annotations_1way(self, slices, genome, is_plus_strand):
        for seqid, start, end, slice_id in slices:
            coordinates = geenuff.orm.Coordinate(seqid=seqid, start=start, end=end, genome=genome)
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
        try:
            tree = self.interval_trees[seqid]
        except KeyError as e:
            logging.warning('no annotations for {}'.format(e))
            return []
        intervals = tree[start:end]
        features = [x.data for x in intervals if x.data.data.is_plus_strand == is_plus_strand]
        return features

    def get_super_loci_frm_features(self, features):
        super_loci = set()
        for feature in features:
            for piece in feature.data.transcribed_pieces:
                super_loci.add(piece.transcribed.super_locus.handler)
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
               (FeatureHandler, geenuff.orm.Feature)]

        return self._get_paired_item(type(old_data), search_col=1, return_col=0, nested_list=key)


# todo, check everything moved successfully from seq info...
class CoordinateHandler(geenuff.api.CoordinateHandlerBase):
    def processing_set(self, sess):
        return sess.query(
            slice_dbmods.CoordinateSet
        ).filter(
            slice_dbmods.CoordinateSet.id == self.data.id
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
            current = slice_dbmods.CoordinateSet(coordinate=self.data, processing_set=processing_set)
        else:
            current.processing_set = processing_set
        sess.add(current)
        sess.commit()


class SuperLocusHandler(geenuff.api.SuperLocusHandlerBase):
    def __init__(self):
        super().__init__()
        self.handler_holder = HandleMaker(self)

    def make_all_handlers(self):
        self.handler_holder.make_all_handlers()

    def load_to_intervaltree(self, trees):
        self.make_all_handlers()
        features = self.features
        for f in features:
            try:
                feature = f.handler
            except AttributeError:
                feature = FeatureHandler()
                feature.add_data(f)
            feature.load_to_intervaltree(trees)

    @property
    def features(self):
        for transcript in self.data.transcribeds:
            for piece in transcript.transcribed_pieces:
                for feature in piece.features:
                    yield feature

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
class TranscribedHandler(geenuff.api.TranscribedHandlerBase):
    pass


class TranslatedHandler(geenuff.api.TranslatedHandlerBase):
    pass


class TranscribedPieceHandler(geenuff.api.TranscribedPieceHandlerBase):
    pass


# todo, there is probably a nicer way to accomplish the following with multi inheritance...
# todo, order py_start / py_end from start/end
class FeatureHandler(geenuff.api.FeatureHandlerBase):
    def load_to_intervaltree(self, trees):
        seqid = self.data.coordinate.seqid
        if seqid not in trees:
            trees[seqid] = intervaltree.IntervalTree()
        tree = trees[seqid]
        py_start, py_end = self.data.start, self.data.end
        if not self.data.is_plus_strand:
            py_start, py_end = py_end + 1, py_start + 1  # todo, clean up w/ similar redundant code elsewhere
        tree[py_start:py_end] = self


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

        if is_plus_strand:
            self.feature_py_start = feature.start
            self.feature_py_end = feature.end
        else:
            # ~ start, end = min(coords) + 1, max(coords) + 1, where +1 makes them incl, excl in the + direction
            self.feature_py_start = feature.end + 1
            self.feature_py_end = feature.start + 1

    def is_detached(self):
        out = False
        if self.slice_coordinates.seqid != self.feature.coordinate.seqid:
            out = True
        elif self.is_plus_strand != self.feature.is_plus_strand:
            out = True
        return out

    def _is_lower(self):
        return self.slice_coordinates.start - self.feature_py_end >= 0

    def _is_higher(self):
        return self.feature_py_start - self.slice_coordinates.end >= 0

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
        start_contained = self.slice_coordinates.start <= self.feature_py_start < self.slice_coordinates.end
        end_contained = self.slice_coordinates.start < self.feature_py_end <= self.slice_coordinates.end
        return start_contained and end_contained

    def _overlaps_lower(self):
        return self.feature_py_start < self.slice_coordinates.start < self.feature_py_end

    def _overlaps_higher(self):
        return self.feature_py_start < self.slice_coordinates.end < self.feature_py_end

    def overlaps_upstream(self):
        if self.is_plus_strand:
            return self._overlaps_lower()
        else:
            return self._overlaps_higher()

    def overlaps_downstream(self):
        if self.is_plus_strand:
            return self._overlaps_higher()
        else:
            return self._overlaps_lower()


class NoFeaturesInSliceError(Exception):
    pass


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript, super_locus, sess, core_queue):
        super().__init__(transcript, super_locus=super_locus, session=sess)
        self.core_queue = core_queue
        #self.session = sess
        self.handlers = []
        self._downstream_piece = None

    def downstream_piece(self, piece):
        if self._downstream_piece is None:
            self._downstream_piece = self.mk_new_piece(piece)
        return self._downstream_piece

    def new_handled_data(self, template=None, new_type=geenuff.orm.Feature, **kwargs):
        data = new_type()
        handler = self.transcript.data.super_locus.handler.handler_holder.mk_n_append_handler(data)
        if template is not None:
            template_dict = geenuff.api.db_attr_as_dict(template)
        else:
            template_dict = {}

        template_dict.update(kwargs)
        for key in template_dict:
            data.__setattr__(key, template_dict[key])
        return handler

    def mk_new_piece(self, piece):
        # increment position of any higher pieces
        old_pieces = self.transcript.data.transcribed_pieces
        for old_piece in old_pieces:
            if old_piece.position > piece.position:
                old_piece.position += 1

        new_piece = geenuff.orm.TranscribedPiece(transcribed=self.transcript.data, position=piece.position + 1)
        new_handler = TranscribedPieceHandler()
        new_handler.add_data(new_piece)
        self.session.add_all(old_pieces + [new_piece])
        self.session.commit()
        self.handlers.append(new_handler)
        return new_piece

    def modify4new_slice(self, new_coords, is_plus_strand=True, trees=None):
        """adjusts features and pieces of transcript to be artificially split across a new sub-coordinate"""
        # todo, this should be done in batch w/ a coordinate filter; not transcript by transcript...
        if trees is None:
            trees = {}
        logging.debug('mod4slice, transcribed: {}, {}'.format(self.transcript.data.id, self.transcript.data.given_id))
        seen_one_overlap = False
        transition_gen = list(self.transition_5p_to_3p())
        piece_at_border = None
        for aligned_features, piece in transition_gen:
            f0 = aligned_features[0]  # take first as all "aligned" features have the same coordinates
            old_coordinate = f0.coordinate
            position_interp = FeatureVsCoords(feature=f0, slice_coordinates=new_coords, is_plus_strand=is_plus_strand)
            # before or detached coordinates (already handled or good as-is, at least for now)
            if position_interp.is_detached():
                pass
            elif position_interp.is_upstream():
                pass  # todo, check that this _has_ been handled already
            elif position_interp.overlaps_upstream():
                pass  # todo, again, this should have already been handled, double check?
            # within new_coords -> swap coordinates
            elif position_interp.is_contained():
                seen_one_overlap = True
                for f in aligned_features:
                    coord_swap = {'feat_id': f.id, 'coordinate_id_new': new_coords.id}
                    self.core_queue.coord_swaps.append(coord_swap)

            # handle pass end of coordinates between previous and current feature, [p] | [f]
            elif position_interp.overlaps_downstream():
                seen_one_overlap = True
                # todo, make new_piece_after_border _here_ not in transition gen...
                piece_at_border = piece
                new_piece_after_border = self.downstream_piece(piece)
                self.core_queue.execute_so_far()  # todo, rm from here once everything uses core
                for feature in aligned_features:
                    self.set_status_downstream_border(new_coords=new_coords, old_coords=old_coordinate,
                                                      is_plus_strand=is_plus_strand,
                                                      template_feature=feature,
                                                      old_piece=piece_at_border,
                                                      new_piece=new_piece_after_border, trees=trees,)

            elif position_interp.is_downstream():
                if piece_at_border is not None:
                    if piece is piece_at_border:
                        # todo, swap from old piece to new_piece_after_border
                        for f in aligned_features:
                            to_swap = {'piece_id_old': piece_at_border.id,
                                       'piece_id_new': self.downstream_piece(piece).id,
                                       'feat_id': f.id}
                            self.core_queue.piece_swaps.append(to_swap)
            else:
                print('ooops', f0)
                raise AssertionError('this code should be unreachable...? Check what is up!')


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

        self._downstream_piece = None  # reset so the piece won't be saved for the next slice

    def get_rel_feature_position(self, feature, prev_feature, new_coords, is_plus_strand):
        pass

    @staticmethod
    def swap_list_item(template_list, replacement_item, item_to_remove):
        template_list = copy.copy(template_list)
        i = template_list.index(item_to_remove)
        template_list[i] = replacement_item
        return template_list

    def set_status_downstream_border(self, new_coords, old_coords, is_plus_strand, template_feature, new_piece,
                                     old_piece, trees):
        if is_plus_strand:
            down_at = new_coords.end
        else:
            down_at = new_coords.start - 1  # -1 so it's exclusive close (incl next open), in reverse direction

        assert old_piece in template_feature.transcribed_pieces, "old id: ({}) not in feature {}'s pieces: {}".format(
            old_piece.id, template_feature.id, template_feature.transcribed_pieces)

        downstream_pieces = self.swap_list_item(template_list=template_feature.transcribed_pieces,
                                                replacement_item=new_piece, item_to_remove=old_piece)

        # downstream feature setup to mark the 2nd half of the template feature outside of the new coordinates
        downstream = self.new_handled_data(template_feature, geenuff.orm.Feature, id=None, start=down_at,
                                           given_id=None, coordinate=old_coords,
                                           start_is_biological_start=False,
                                           transcribed_pieces=downstream_pieces)
        downstream.load_to_intervaltree(trees)

        # template feature is truncated until it's contained in the new coordiantes
        template_feature.end = down_at
        template_feature.end_is_biological_end = False
        template_feature.coordinate = new_coords

        self.session.add_all([downstream.data])
        self.session.commit()  # todo, figure out what the real rules are for committing, bc slower, but less buggy?


