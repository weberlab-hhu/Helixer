import os
from shutil import copyfile
from sqlalchemy.orm import load_only
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import geenuff
from geenuff.base.helpers import full_db_path
from geenuff.base.orm import Coordinate
from helixerprep.core.orm import Mer
from helixerprep.core.handlers import CoordinateHandler

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

    def add_mers(self, min_k, max_k):
        """Adds kmers in the specified range for all Coordinates that do not have an
        entry in Mer of that length. Can not handle partial kmer addition
        (e.g. adding kmers in a new range when some already exist for a Coordinate)
        """
        all_mers = self.session.query(Mer).options(load_only('id')).all()
        coords_without_mers = self.session.query(Coordinate).\
                                  filter(Coordinate.id.notin_(all_mers)).all()
        for coord in coords_without_mers:
            coord_handler = CoordinateHandler(coord)
            coord_handler.add_mer_counts_to_db(min_k, max_k, self.session)


