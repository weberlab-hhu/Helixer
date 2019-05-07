import os
import intervaltree

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import geenuff
from geenuff.base.orm import Coordinate, Genome
from geenuff.base.helpers import full_db_path
from helixerprep.core.partitions import CoordinateGenerator, choose_set
from .numerify import ExampleMaker
from ..core.handlers import CoordinateHandler


class ExportController(object):
    def __init__(self, db_path_in=None, h5_path_out):
        self.db_path_in = db_path_in
        self.h5_path_out = h5_path_out
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    def export(self, chunk_size, shuffle, seed):
        """Fetches all Coordinates, calls on functions in numerify.py to split
        and encode them and then saves the (possibly shuffled) sequences to the
        specified .h5 file.
        """
        chunks = []
        all_coords = self.session.query(Coordinate).all()
        example_maker = ExampleMaker()
        for coord in all_coords:
            chunks.append(list(example_maker.examples_from_coord(coord, True, chunk_size)))

        # output to .h5

    def gen_slices(self, genome, train_size, dev_size, chunk_size, seed):
        """Returns a list of all slices of all coordinates of a genome
        with their assigned processing sets"""
        self.slices = []
        coords = self.session.query(geenuff.orm.Coordinate).all()
        cg = CoordinateGenerator(train_size, dev_size, chunk_size, seed)
        for coord in coords:
            length = coord.end - coord.start
            for start, end, pset in cg.divvy_coordinates(length, coord.sha1):
                sliced_coord = (
                    coord.seqid,
                    coord.sequence[start:end],
                    start,
                    end,
                    pset,
                )
                self.slices.append(sliced_coord)

    def fill_intervaltrees(self):
        self.intervaltrees = {}
        for sl in self.super_loci:
            sl.load_to_intervaltree(self.interval_trees)

    def load_super_loci(self):
        self.super_loci = []
        sl_data = self.session.query(geenuff.orm.SuperLocus).all()
        for sl in sl_data:
            super_locus = SuperLocusHandler()
            super_locus.add_data(sl)
            self.super_loci.append(super_locus)

    def slice_db(self, train_size, dev_size, chunk_size, seed):
        self.load_super_loci()
        self.fill_intervaltrees()

        genome = self.get_one_genome()
        self.gen_slices(genome, train_size, dev_size, chunk_size, seed)
        self.slice_annotations(genome)

    def slice_annotations(self, genome):
        """Artificially slices annotated genome to match sequence slices
        and adjusts transcripts as appropriate.
        Enum, """
        self._slice_annotations_1way(self.slices, genome, is_plus_strand=True)
        self._slice_annotations_1way(self.slices[::-1], genome, is_plus_strand=False)

    def _slice_annotations_1way(self, slices, genome, is_plus_strand):
        for seqid, sequence, start, end, pset in slices:
            coordinate = geenuff.orm.Coordinate(seqid=seqid,
                                                sequence=sequence,
                                                start=start,
                                                end=end,
                                                genome=genome)
            coordinate_set = CoordinateSet(coordinate=coordinate, processing_set=pset)
            self.session.add_all([coordinate, coordinate_set])
            self.session.commit()

            overlapping_super_loci = self.get_super_loci_frm_slice(seqid, start, end,
                                                                   is_plus_strand=is_plus_strand)
            for super_locus in overlapping_super_loci:
                super_locus.make_all_handlers()
                super_locus.modify4slice(new_coords=coordinate, is_plus_strand=is_plus_strand,
                                         session=self.session, trees=self.interval_trees,
                                         core_queue=self.core_queue)
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
            raise ValueError('No, interval trees defined. The method .fill_intervaltrees '
                             'must be called first')
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

    def reshuffle_train_dev_sets(self, genome, train_size, dev_size, seed):
        """Reshuffles the train and dev annotation of processing sets for Coordinate individually.
        Due to limitiation with the inheritance in Sqlalchemy, a is_slice flag could not
        conveniently be added to the Coordinate class, which leads to much more complicated queries
        used here.
        """
        import random
        sliced_coord_id_query = self.session.query(CoordinateSet).with_entities(CoordinateSet.id)
        main_coords = (
            self.session.query(Coordinate)
                .filter(Coordinate.genome == genome)
                .filter(Coordinate.id.notin_(sliced_coord_id_query))
                .all()
        )
        for main_coord in main_coords:
            main_coord_handler = CoordinateHandler(main_coord)
            sub_coords = main_coord_handler.get_slices(self.session, ['train', 'dev'])

            random.seed(main_coord.sha1 + seed)
            for sub_coord in sub_coords:
                new_set = choose_set(train_size, dev_size)
                while new_set == 'test':
                    new_set = choose_set(train_size, dev_size)
                sub_coord.processing_set = new_set
        self.session.commit()


