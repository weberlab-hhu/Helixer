"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
from shutil import copyfile
import intervaltree

import annotations
import annotations_orm
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

    def copy_db(self):
        copyfile(self.db_path_in, self.db_path_sliced)

    def full_db_path(self):
        return 'sqlite:///{}'.format(self.db_path_sliced)

    def mk_session(self):
        if not os.path.exists(self.db_path_sliced):
            self.copy_db()
        self.engine = create_engine(self.full_db_path(), echo=False)  # todo, dynamic / real path
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
        sg = sequences.StructuredGenome()
        sg.from_json(self.sequences_path)
        self.structured_genome = sg

    def fill_intervaltrees(self):
        for sl in self.super_loci:
            sl.load_to_intervaltree(self.interval_trees)

    def slice_annotations(self):
        for seq in self.structured_genome.sequences:
            print(seq.meta_info.seqid)
            for slice in seq.slices:
                print('start: {}, end: {}, slice id: {}'.format(slice.start, slice.end, slice.slice_id))
                yield seq.meta_info.seqid, slice.start, slice.end, slice.slice_id
            # todo, setup slice as sequence_info in database
            # todo, get features & there by superloci in slice
            # todo, crop/reconcile superloci/transcripts/transcribeds/features with slice

    def get_super_loci_frm_slice(self, seqid, start, end):
        features = self.get_features_from_slice(seqid, start, end)
        super_loci = self.get_super_loci_frm_features(features)
        return super_loci

    def get_features_from_slice(self, seqid, start, end):
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


class SuperLocusHandler(annotations.SuperLocusHandler):

    def load_to_intervaltree(self, trees):
        features = self.data.features
        for f in features:
            feature = FeatureHandler()  # recreate feature handler post load (todo, mv elsewhere so it's always done?)
            feature.add_data(f)
            feature.load_to_intervaltree(trees)


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


class IndecipherableLinkageError(Exception):
    pass


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript):
        super().__init__(transcript)

    def transition_5p_to_3p(self):
        # todo,
        #  for each protein:
        #    _transition_w_prot
        # setup
        pass

    def _transition_w_prot(self, protein):
        # todo,
        #  get ordered pieces/features
        #  setup transcript status @ first feature
        pass

    @staticmethod
    def sorted_features(piece):
        features = piece.features
        # confirm strand & seqid
        assert all([f.coordinates == features[0].coordinates for f in features])
        assert all([f.is_plus_strand == features[0].is_plus_strand for f in features])
        features = sorted(features, key=lambda x: x.cmp_key())
        if not features[0].is_plus_strand:
            features.reverse()
        return features

    def sort_pieces(self, sess):
        pieces = self.transcript.data.transcribed_pieces
        # start with one piece, extend until both ends are reached
        ordered_pieces = pieces[0:1]
        self._extend_to_end(ordered_pieces, sess, downstream=True)
        self._extend_to_end(ordered_pieces, sess, downstream=False)
        assert set(ordered_pieces) == set(pieces)
        return ordered_pieces

    def _extend_to_end(self, ordered_pieces, sess, downstream=True):
        if downstream:
            next_fn = self.get_downstream_link
            latest_i = -1
            attr = 'downstream'
        else:
            next_fn = self.get_upstream_link
            latest_i = 0
            attr = 'upstream'

        while True:
            nextlink = next_fn(current_piece=ordered_pieces[latest_i], sess=sess)
            if nextlink is None:
                break
            nextstream = nextlink.__getattribute__(attr)
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
        assert len(matches) == 1  # todo; can we guarantee this?
        return matches[0]

    def get_upstream_link(self, current_piece, sess):
        downstreams = sess.query(annotations_orm.DownstreamFeature).all()
        # DownstreamFeature s of this pice
        downstreams_current = [x for x in downstreams if current_piece in x.transcribed_pieces]
        links = self._find_matching_links(updown_candidates=downstreams_current, get_upstreams=True)
        return self._links_list2link(links, direction='upstream', current_piece=current_piece)

    def get_downstream_link(self, current_piece, sess):
        upstreams = sess.query(annotations_orm.UpstreamFeature).all()
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

    @staticmethod
    def _links_list2link(links, direction, current_piece):
        if len(links) == 0:
            return None
        elif len(links) == 1:
            return links[0]
        else:
            raise IndecipherableLinkageError("Multiple possible within-transcript {} links found from {}, ({})".format(
                direction, current_piece, links
            ))

    def sort_all(self, sess):
        out = []
        for piece in self.sort_pieces(sess):
            out.append(self.sorted_features(piece))
        return out
