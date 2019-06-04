import os
import multiprocessing
import random
from shutil import copyfile
from sqlalchemy.orm import load_only
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import geenuff
from geenuff.base.helpers import full_db_path
from geenuff.base.orm import Coordinate
from helixerprep.core.orm import Mer
from helixerprep.core.helpers import MerCounter


class MerController(object):
    def __init__(self, db_path_in, db_path_out):
        self._setup_db(db_path_in, db_path_out)
        self._mk_session()

    def _setup_db(self, db_path_in, db_path_out):
        self.db_path = db_path_out
        if db_path_out != '':
            if os.path.exists(db_path_out):
                print('overriding the helixer output db at {}'.format(db_path_out))
            copyfile(db_path_in, db_path_out)
        else:
            print('adding the helixer additions directly to input db at {}'.format(db_path_in))
            self.db_path = db_path_in

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path), echo=False)
        # add Helixer specific table to the input db if it doesn't exist yet
        if not self.engine.dialect.has_table(self.engine, 'mer'):
            geenuff.orm.Base.metadata.tables['mer'].create(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    @staticmethod
    def _count_mers(coord, min_k, max_k):
        mer_counters = []
        # setup all counters
        for k in range(min_k, max_k + 1):
            mer_counters.append(MerCounter(k))

        # count all 'mers
        for i in range(len(coord.sequence)):
            for k in range(min_k, max_k + 1):
                if i + 1 >= k:
                    substr = coord.sequence[i-(k-1):i+1]
                    mer_counters[k - 1].add_count(substr)
        print('done with {}'.format(coord))
        return coord, mer_counters

    def add_mers(self, min_k, max_k, n_processes=8):
        all_mers = self.session.query(Mer).options(load_only('id')).all()
        coords_without_mers = self.session.query(Coordinate).\
                                  filter(Coordinate.id.notin_(all_mers)).all()
        random.shuffle(coords_without_mers)

        input_data = [[c, min_k, max_k] for c in coords_without_mers]
        with multiprocessing.Pool(processes=n_processes) as pool:
            mer_counters = pool.starmap(MerController._count_mers, input_data)
        self._add_mer_counters_to_db(mer_counters)

    def _add_mer_counters_to_db(self, mer_counters_all_coords):
        # convert to canonical and setup db entries
        for coord, mer_counters in mer_counters_all_coords:
            for mer_counter in mer_counters:
                for mer_sequence, count in mer_counter.export().items():
                    mer = Mer(coordinate_id=coord.id,
                              mer_sequence=mer_sequence,
                              count=count,
                              length=mer_counter.k)
                    self.session.add(mer)
        self.session.commit()


