import os
import h5py
import numpy as np
import deepdish as dd
import sklearn

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from geenuff.base.orm import Coordinate
from geenuff.base.helpers import full_db_path
from .numerify import CoordNumerifier


class ExportController(object):
    def __init__(self, db_path_in, out_dir):
        self.db_path_in = db_path_in
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        elif os.listdir(self.out_dir):
            print('WARNING: target dir {} is not empty'.format(self.out_dir))
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

        def get_file_name(i):
            return os.path.join(self.out_dir, 'data' + str(i) + '.h5')

        def save_data(data, i, shuffle):
            if shuffle:
                x, y, m = data['inputs'], data['labels'], data['label_masks']
                x, y, m = sklearn.utils.shuffle(x, y, m)
                data['inputs'], data['labels'], data['label_masks'] = x, y, m
            dd.io.save(get_file_name(i), data, compression=None)

        n_file_chunks = 0
        n_chunks_total = 0  # total number of chunks
        current_size = 0  # if this is > approx_file_size make new file chunk
        data = get_empty_data_dict()
        all_coords = self.session.query(Coordinate).all()[:10]
        print('{} coordinates choosen to numerify'.format(len(all_coords)))
        for i, coord in enumerate(all_coords):
            if coord.features:
                n_masked_bases = 0
                for is_plus_strand in [True, False]:
                    numerifier = CoordNumerifier(coord, is_plus_strand, chunk_size)
                    coord_data = numerifier.numerify()
                    for key in ['inputs', 'labels', 'label_masks']:
                        data[key] += coord_data[key]
                    current_size += coord.end
                    n_chunks_total += len(data['label_masks'])
                    n_masked_bases += sum([len(m) - np.count_nonzero(m)
                                           for m
                                           in coord_data['label_masks']])
                    # break data up in file chunks
                    if current_size * 12 > approx_file_size:
                        save_data(data, n_file_chunks, shuffle)
                        data = get_empty_data_dict()
                        n_file_chunks += 1
                        current_size = 0
                masked_bases_percent = n_masked_bases / (coord.end * 2) * 100
                print(('{}/{} Numerified {} of species {} with {} features '
                       'and a base level error rate of {:.2f}%').format(
                    i + 1, len(all_coords), coord, coord.genome.species, len(coord.features),
                    masked_bases_percent))
            else:
                print('{}/{} Skipping {} of species {} as it has no features'.format(
                    i + 1, len(all_coords), coord, coord.genome.species))

        # numerify the left over data
        if current_size > 0:
            print(('Numerified the left over data with a base level '
                   'error rate of {:.2f}%').format(masked_bases_percent))
            save_data(data, n_file_chunks, shuffle)
        else:
            print('Skipping the left over data as it is empty')

        # add the total n_chunks to the first file
        file_name = get_file_name(0)
        f = h5py.File(file_name, 'r+')
        config = f['config'].attrs
        config['n_chunks_total'] = n_chunks_total
        f.close()
