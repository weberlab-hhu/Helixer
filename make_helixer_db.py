import os
import argparse
from shutil import copyfile
from sqlalchemy.orm import load_only
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import geenuff
from geenuff.base.helpers import full_db_path
from geenuff.base.orm import Genome, Coordinate
import helixerprep.datas.annotations.slicer as slicer
from helixerprep.datas.annotations.slice_dbmods import Mer


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
        (e.g. adding kmers in a new range for existing Coordinates)
        """
        all_mers = self.session.query(Mer).options(load_only('id')).all()
        coords_without_mers = self.session.query(Coordinate).\
                                  filter(Coordinate.id.notin_(all_mers)).all()
        for coord in coords_without_mers:
            coord_handler = slicer.CoordinateHandler(coord)
            coord_handler.add_mer_counts_to_db(min_k, max_k, self.session)


def main(args):
    controller = MerController(args.db_path_in, args.db_path_out)  # inserts Mer table
    if args.max_k > 0:
        controller.add_mers(args.min_k, args.max_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db_path_in', type=str, required=True,
                    help=('Path to the GeenuFF SQLite input database.'))
    io.add_argument('--db_path_out', type=str, default='',
                    help=('Output path of the new Helixer SQLite database. If not provided '
                          'the input database will be replaced.'))

    fasta_specific = parser.add_argument_group("Controlling the kmer generation:")
    fasta_specific.add_argument('--min_k', help='minumum size kmer to calculate from sequence',
                                default=0, type=int)
    fasta_specific.add_argument('--max_k', help='maximum size kmer to calculate from sequence',
                                default=0, type=int)

    args = parser.parse_args()

    assert args.min_k <= args.max_k, 'min_k can not be greater than max_k'
    if args.max_k > 0 and args.min_k == 0:
        args.min_k = 1
        print('min_k parameter set to 1')

    main(args)
