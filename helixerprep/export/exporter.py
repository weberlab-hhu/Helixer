import numpy as np
import deepdish as dd
import sklearn

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from geenuff.base.orm import Coordinate
from geenuff.base.helpers import full_db_path
from .numerify import CoordNumerifier


class ExportController(object):
    def __init__(self, db_path_in, h5_path_out):
        self.db_path_in = db_path_in
        self.h5_path_out = h5_path_out
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    def export(self, chunk_size, shuffle, seed, approx_file_size=100*2**20):
        """Fetches all Coordinates, calls on functions in numerify.py to split
        and encode them and then saves the sequences in possibly multiply files
        of about the size of approx_file_size.
        """
        def get_empty_data_dict():
            d = {
                'inputs': [],
                'labels': [],
                'label_masks': [],
                'config': {
                    'chunk_size': chunk_size,
                    'shuffle': shuffle,
                    'seed': seed,
                },
            }
            return d

        def save_data(data, file_chunk_count, shuffle):
            if shuffle:
                x, y, m = data['inputs'], data['labels'], data['label_masks']
                x, y, m = sklearn.utils.shuffle(x, y, m)
                data['inputs'], data['labels'], data['label_masks'] = x, y, m
            dd.io.save(self.h5_path_out.split('.')[0] + str(file_chunk_count) + '.h5',
                       data, compression=None)
            data = get_empty_data_dict()

        file_chunk_count = 0
        current_size = 0  # if this is > approx_file_size make new file chunk
        data = get_empty_data_dict()
        all_coords = self.session.query(Coordinate).all()
        print('{} coordinates choosen to numerify'.format(len(all_coords)))
        for coord in all_coords:
            if coord.features:
                n_masked_bases = 0
                for is_plus_strand in [True, False]:
                    numerifier = CoordNumerifier(coord, is_plus_strand, chunk_size)
                    coord_data = numerifier.numerify()
                    for key in ['inputs', 'labels', 'label_masks']:
                        data[key] += coord_data[key]
                    current_size += coord.end
                    n_masked_bases += sum([np.count_nonzero(m) for m in coord_data['label_masks']])
                    # break data up in file chunks
                    if current_size * 12 > approx_file_size:
                        save_data(data, file_chunk_count, shuffle)
                        file_chunk_count += 1
                        current_size = 0
                masked_bases_percent = n_masked_bases / (coord.end * 2) * 100
                print('Numerified {} of species {} with base level error rate of {:.2f}%'.format(
                    coord, coord.genome.species, masked_bases_percent))
            else:
                print('Skipping {} as it has no features'.format(coord))
        # numerify the left over data
        save_data(data, file_chunk_count, shuffle)
