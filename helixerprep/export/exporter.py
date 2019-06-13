import os
import h5py
import numpy as np
import sklearn
import configparser

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from geenuff.base.orm import Coordinate
from geenuff.base.helpers import full_db_path
from .numerify import CoordNumerifier


class ExportController(object):
    def __init__(self, db_path_in, h5_out):
        self.db_path_in = db_path_in
        self.h5_out = h5_out
        if os.path.isfile(self.h5_out):
            print('WARNING: target file {} already exists. Overriding.\n'.format(self.h5_out))
        self.h5_file = h5py.File(self.h5_out, 'w')
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    def export(self, chunk_size):
        """Fetches all Coordinates, calls on functions in numerify.py to split
        and encode them and then saves the sequences in possibly multiply files
        of about the size of approx_file_size.
        """
        def save_data(inputs, labels, label_masks):
            # zero-pad each sequence to chunk_size
            # this is inefficient if there could be a batch with only sequences smaller than
            # chunk_size, but taking care of that introduces a lot of extra complexity
            n_seq = len(inputs)
            X = np.zeros((n_seq, chunk_size, 4), dtype=inputs[0].dtype)
            y = np.zeros((n_seq, chunk_size, 3), dtype=labels[0].dtype)
            sample_weights = np.zeros((n_seq, chunk_size), dtype=label_masks[0].dtype)
            for j in range(n_seq):
                sample_len = len(inputs[j])
                X[j, :sample_len, :] = inputs[j]
                y[j, :sample_len, :] = labels[j]
                sample_weights[j, :sample_len] = label_masks[j]

            # check if this is the first batch to save
            dset_keys = ['X', 'y', 'sample_weights']
            if '/data/X' in self.h5_file:
                for dset_key in dset_keys:
                    dset = self.h5_file['/data/' + dset_key]
                    old_len = dset.shape[0]
                    dset.resize(old_len + n_seq, axis=0)
            else:
                old_len = 0
                self.h5_file.create_dataset('/data/X',
                                            shape=(n_seq, chunk_size, 4),
                                            maxshape=(None, chunk_size, 4),
                                            chunks=(1, chunk_size, 4),
                                            dtype='float16',
                                            compression='lzf',
                                            shuffle=True)  # only for the compression
                self.h5_file.create_dataset('/data/y',
                                            shape=(n_seq, chunk_size, 3),
                                            maxshape=(None, chunk_size, 3),
                                            chunks=(1, chunk_size, 3),
                                            dtype='int8',
                                            compression='lzf',
                                            shuffle=True)
                self.h5_file.create_dataset('/data/sample_weights',
                                            shape=(n_seq, chunk_size),
                                            maxshape=(None, chunk_size),
                                            chunks=(1, chunk_size),
                                            dtype='int8',
                                            compression='lzf',
                                            shuffle=True)
            # add new data
            for dset_key, data in zip(dset_keys, [X, y, sample_weights]):
                self.h5_file['/data/' + dset_key][old_len:] = data
            self.h5_file.flush()

        n_seq_total = 0  # total number of individual sequences
        all_coords = self.session.query(Coordinate).all()[:5]
        print('{} coordinates chosen to numerify'.format(len(all_coords)))
        for i, coord in enumerate(all_coords):
            if coord.features:
                inputs, labels, label_masks = [], [], []
                n_masked_bases = 0
                for is_plus_strand in [True, False]:
                    numerifier = CoordNumerifier(coord, is_plus_strand, chunk_size)
                    coord_data = numerifier.numerify()
                    # add data
                    inputs += coord_data['inputs']
                    labels += coord_data['labels']
                    label_masks += coord_data['label_masks']
                    # keep track of variables
                    n_seq_total += len(coord_data['inputs'])
                    n_masked_bases += sum([len(m) - np.count_nonzero(m)
                                           for m
                                           in coord_data['label_masks']])
                save_data(inputs, labels, label_masks)
                masked_bases_percent = n_masked_bases / (coord.end * 2) * 100
                print(('{}/{} Numerified {} of species {} with {} features '
                       'and a base level error rate of {:.2f}%').format(
                    i + 1, len(all_coords), coord, coord.genome.species, len(coord.features),
                    masked_bases_percent))
            else:
                print('{}/{} Skipping {} of species {} as it has no features'.format(
                    i + 1, len(all_coords), coord, coord.genome.species))
        self.h5_file.close()
