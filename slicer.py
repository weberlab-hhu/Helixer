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

    def reconcile_with_slice(self, seqid, start, end):
        pass
#
#    def reconcile_with_slice(self, seqid, start, end, status, last_before_slice):
#        #overlap_status = OverlapStatus()
#        #overlap_status.set_status(self, seqid, start, end)
#        #status = overlap_status.status
#        if status == OverlapStatus.contained:
#            pass  # leave it alone
#        elif status == OverlapStatus.no_overlap:
#            # todo, if it is the last feature before the slice (aka, if the next one is contained)
#            if last_before_slice:
#                self.shift_phase(start, end)
#                pass  # todo, change to 1bp status_at (w/ phase if appropriate)
#            pass  # todo, delete (and from transcripts / super_locus)
#        elif status == OverlapStatus.overlaps_upstream:
#            self.shift_phase(start, end)
#            self.crop(start, end)
#        elif status == OverlapStatus.overlaps_downstream:
#            # just crop
#            self.crop(start, end)

#    def length_outside_slice(self, start, end):
#        if self.is_plus_strand():
#            length_outside_slice = start - self.start
#        else:
#            length_outside_slice = self.end - end
#        return length_outside_slice
#
#    def crop(self, start, end):
#        if self.start < start:
#            self.start = start
#        if self.end > end:
#            self.end = end
#
#    def shift_phase(self, start, end):
#        if self.phase is not None:
#            l_out = self.length_outside_slice(start, end)
#            self.phase = (l_out - self.phase) % 3


class TranscribedHandler(annotations.TranscribedHandler):
    def reconcile_with_slice(self, seqid, start, end):
        pass


class TranslatedHandler(annotations.TranslatedHandler):
    def reconcile_translated_with_slice(self, seqid, start, end):
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


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript):
        super().__init__(transcript)

    def crop_to_slice(self, seqid, start, end):
        """crops transcript in place"""
        pass

    def transition_5p_to_3p(self):
        # setup
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

    def sort_pieces(self):
        pass


    def sort_all(self):
        pass