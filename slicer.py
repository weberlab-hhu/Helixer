"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
from shutil import copyfile
import intervaltree
import copy

import annotations
import annotations_orm
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


class SuperLocusHandler(annotations.SuperLocusHandler):
    def __init__(self):
        super().__init__()
        self.handle_holder = HandleMaker(self)

    def make_all_handlers(self):
        self.handle_holder.make_all_handlers()

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


class IndecipherableLinkageError(Exception):
    pass


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript, sess):
        super().__init__(transcript)
        self.session = sess
        self.handlers = []

    def transition_5p_to_3p(self):
        status = TranscriptStatus()
        for piece in self.sort_pieces():
            piece_features = self.sorted_features(piece)
            for aligned_features in self.stack_matches(piece_features):
                self.update_status(status, aligned_features)
                yield aligned_features, copy.deepcopy(status), piece

        # todo,
        #  get ordered pieces/features
        #  setup transcript status @ first feature

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
            # todo, trans-splicing features
            # status features
            elif ftype == type_enums.IN_RAW_TRANSCRIPT:
                status.saw_tts()
            elif ftype == type_enums.IN_TRANSLATED_REGION:
                status.saw_start(phase=feature.phase)
            elif ftype == type_enums.IN_INTRON:
                status.splice_open()
            elif ftype == type_enums.ERROR:
                pass
            else:
                raise ValueError('no implementation for updating status with feature of type {}'.format(ftype))

    def modify4new_slice(self, new_coords, is_plus_strand=True):
        if is_plus_strand:
            downstream_border = new_coords.end
            upstream_border = new_coords.start
        else:
            # todo, allow - pass of new_coords
            raise NotImplementedError

        new_piece = annotations_orm.TranscribedPiece(super_locus=self.transcript.data.super_locus,
                                                     transcribed=self.transcript.data)

        transition_gen = self.transition_5p_to_3p()
        prev_features, prev_status, prev_piece = next(transition_gen)  # todo, check and handle things for these!

        for aligned_features, status, piece in transition_gen:
            f0 = aligned_features[0]  # take first as all "aligned" features have the same coordinates
            same_seq = f0.coordinate.seqid == new_coords.seqid
            # before or detached coordinates (already handled or good as-is, at least for now)
            if not same_seq or f0.end < upstream_border:
                pass
            # it should never overlap start (because this should have been handled and split already)
            elif f0.start < upstream_border <= f0.end:
                raise ValueError('unhandled straddling of upstream boarder')
            # within new_coords -> swap coordinates
            elif f0.start >= upstream_border and f0.end <= downstream_border:
                for f in aligned_features:
                    f.coordinates = new_coords
            # handle feature [  |  ] straddling end of coordinates
            elif f0.start <= downstream_border < f0.end:
                # make new UpDownLink and status features to handle split
                self.set_status_at_border(new_coords, is_plus_strand, f0, status)
                self.split_feature_at_border(new_coords, is_plus_strand, f0)
            # handle pass end of coordinates between previous and current feature, [p] | [f]
            elif prev_features[0].end <= downstream_border < f0.start:
                self.set_status_at_border(new_coords, is_plus_strand, prev_features[0], prev_status)
                self.swap_piece(f0, new_piece, old_piece=piece)
            elif f0.start > downstream_border:
                self.swap_piece(f0, new_piece, old_piece=piece)
            else:
                raise AssertionError('this code should be unreachable...? Check what is up!')

            # and step
            prev_features = aligned_features
            prev_status = status
            prev_piece = piece

        if not new_piece.features:
            self.session.delete(new_piece)

    def swap_piece(self, feature, new_piece, old_piece):
        pass  # todo

    def set_status_at_border(self, new_coords, is_plus_strand, template_feature, status):
        pass  # todo!

    def split_feature_at_border(self, new_coords, is_plus_strand, feature):
        pass  # todo

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

    def sort_pieces(self):
        pieces = self.transcript.data.transcribed_pieces
        # start with one piece, extend until both ends are reached
        ordered_pieces = pieces[0:1]
        self._extend_to_end(ordered_pieces, downstream=True)
        self._extend_to_end(ordered_pieces, downstream=False)
        assert set(ordered_pieces) == set(pieces)
        return ordered_pieces

    def _extend_to_end(self, ordered_pieces, downstream=True):
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

    def sort_all(self):
        out = []
        for piece in self.sort_pieces():
            out.append(self.sorted_features(piece))
        return out

    @staticmethod
    def stack_matches(features):
        ifeatures = iter(features)
        prev = next(ifeatures)
        current = [prev]
        for feature in ifeatures:
            if feature.cmp_key()[0:4] == prev.cmp_key()[0:4]:
                current.append(feature)
            else:
                yield current
                current = [feature]
            prev = feature
        yield current
