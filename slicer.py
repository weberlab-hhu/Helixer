"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
from shutil import copyfile

import annotations
import annotations_orm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import os
import sequences

from gff_2_annotations import TranscriptStatus  # todo, move to helpers?


class SliceController(object):

    def __init__(self, db_path_in=None, db_path_sliced=None, sequences_path=None):
        self.db_path_in = db_path_in
        self.db_path_sliced = db_path_sliced
        self.sequences_path = sequences_path
        self.structured_genome = None
        self.engine = None
        self.session = None
        self.super_loci = []

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

    def slice_annotations(self):
        for seq in self.structured_genome.sequences:
            print(seq.meta_info.seqid)
            for slice in seq.slices:
                print('start: {}, end: {}, slice id: {}'.format(slice.start, slice.end, slice.slice_id))
            # todo, setup slice as sequence_info in database
            # todo, get features & there by superloci in slice
            # todo, crop/reconcile superloci/transcripts/transcribeds/features with slice


class SuperLocusHandler(annotations.SuperLocusHandler):
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