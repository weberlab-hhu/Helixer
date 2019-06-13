import os
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import random

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from geenuff.base.orm import Coordinate
from geenuff.base.helpers import full_db_path
from .numerify import CoordNumerifier


class ExportController(object):
    def __init__(self, db_path_in, data_dir):
        self.db_path_in = db_path_in
        if os.listdir(data_dir):
            print('WARNING: directory not empty')
        self.h5_train = h5py.File(os.path.join(data_dir, 'training_data.h5'), 'w')
        self.h5_val = h5py.File(os.path.join(data_dir, 'validation_data.h5'), 'w')
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    def export(self, chunk_size, seed=42):
        """Fetches all Coordinates, calls on functions in numerify.py to split
        and encode them and then saves the sequences in possibly multiply files
        of about the size of approx_file_size.
        """

        def save_data(h5_file, inputs, labels, label_masks):
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
            if '/data/X' in h5_file:
                for dset_key in dset_keys:
                    dset = h5_file['/data/' + dset_key]
                    old_len = dset.shape[0]
                    dset.resize(old_len + n_seq, axis=0)
            else:
                old_len = 0
                h5_file.create_dataset('/data/X',
                                       shape=(n_seq, chunk_size, 4),
                                       maxshape=(None, chunk_size, 4),
                                       chunks=(1, chunk_size, 4),
                                       dtype='float16',
                                       compression='lzf',
                                       shuffle=True)  # only for the compression
                h5_file.create_dataset('/data/y',
                                       shape=(n_seq, chunk_size, 3),
                                       maxshape=(None, chunk_size, 3),
                                       chunks=(1, chunk_size, 3),
                                       dtype='int8',
                                       compression='lzf',
                                       shuffle=True)
                h5_file.create_dataset('/data/sample_weights',
                                       shape=(n_seq, chunk_size),
                                       maxshape=(None, chunk_size),
                                       chunks=(1, chunk_size),
                                       dtype='int8',
                                       compression='lzf',
                                       shuffle=True)
            # add new data
            for dset_key, data in zip(dset_keys, [X, y, sample_weights]):
                h5_file['/data/' + dset_key][old_len:] = data
            h5_file.flush()

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
                    n_masked_bases += sum(
                        [len(m) - np.count_nonzero(m) for m in coord_data['label_masks']])
                # split and save
                split_data = train_test_split(inputs, labels, label_masks, test_size=0.2,
                                              random_state=seed)
                save_data(self.h5_train, split_data[0], split_data[2], split_data[4])
                save_data(self.h5_val, split_data[1], split_data[3], split_data[5])
                masked_bases_percent = n_masked_bases / (coord.end * 2) * 100
                print(
                    ('{}/{} Numerified {} of species {} with {} features '
                     'and a base level error rate of {:.2f}%').format(i + 1, len(all_coords), coord,
                                                                      coord.genome.species,
                                                                      len(coord.features),
                                                                      masked_bases_percent))
            else:
                print('{}/{} Skipping {} of species {} as it has no features'.format(
                    i + 1, len(all_coords), coord, coord.genome.species))
        self.h5_train.close()
        self.h5_val.close()
