import os
import h5py
import copy
import numpy as np
import random
from itertools import compress

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from geenuff.base.orm import Coordinate, Genome
from geenuff.base.helpers import full_db_path
from .numerify import CoordNumerifier


class ExportController(object):
    def __init__(self, db_path_in, data_dir, only_test_set=False):
        self.db_path_in = db_path_in
        self.only_test_set = only_test_set
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        elif os.listdir(data_dir):
            print('Output directory must be empty or not existing')
            exit()
        if self.only_test_set:
            print('Exporting all data into test_data.h5')
            self.h5_test = h5py.File(os.path.join(data_dir, 'test_data.h5'), 'w')
        else:
            print('Exporting all data into test_data.h5')
            self.h5_train = h5py.File(os.path.join(data_dir, 'training_data.h5'), 'w')
            self.h5_val = h5py.File(os.path.join(data_dir, 'validation_data.h5'), 'w')
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    @staticmethod
    def _split_data(data_arrays, test_size=0.2):
        """Basically does the same as sklearn.model_selection.train_test_split except
        it does not always fill the test arrays with at least one element.
        Expects the arrays to be in the order: inputs, labels, label_masks
        """
        train_arrays, val_arrays = [[], [], []], [[], [], []]
        for i in range(len(data_arrays[0])):
            if random.random() > test_size:
                for j in range(3):
                    train_arrays[j].append(data_arrays[j][i])
            else:
                for j in range(3):
                    val_arrays[j].append(data_arrays[j][i])
        return train_arrays, val_arrays

    def _save_data(self, h5_file, inputs, labels, label_masks, chunk_size, timestep_len):
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

    def _fetch_coords(self, genomes):
        if genomes:
            print('Selecting the following genomes: {}'.format(genomes))
            all_coords = (self.session.query(Coordinate)
                         .join(Genome, Genome.id == Coordinate.genome_id)
                         .filter(Genome.species.in_(genomes))
                         .all())
        else:
            print('Selecting all genomes from {}'.format(self.db_path_in))
            all_coords = self.session.query(Coordinate).all()
        return all_coords

    def _close_files(self):
        if self.only_test_set:
            self.h5_test.close()
        else:
            self.h5_train.close()
            self.h5_val.close()

    def export(self, chunk_size, timestep_len=1, genomes=''):
        """timestep_len is the number of bp inputs at each timestep (and also how much to output)"""
        assert chunk_size % timestep_len == 0

        all_coords = self._fetch_coords(genomes)
        print('{} coordinates chosen to numerify'.format(len(all_coords)))

        for i, coord in enumerate(all_coords):
            if coord.features:
                inputs, labels, label_masks = [], [], []
                n_masked_bases = 0
                for is_plus_strand in [True, False]:
                    numerifier = CoordNumerifier(coord, is_plus_strand, chunk_size)
                    coord_data = numerifier.numerify()
                    # keep track of variables
                    n_masked_bases += sum(
                        [len(m) - np.count_nonzero(m) for m in coord_data['label_masks']])
                    # filter out sequences that are completely masked as error
                    valid_data = [s.any() for s in coord_data['label_masks']]
                    coord_data['inputs'] = list(compress(coord_data['inputs'], valid_data))
                    coord_data['labels'] = list(compress(coord_data['labels'], valid_data))
                    coord_data['label_masks'] = list(compress(coord_data['label_masks'], valid_data))
                    # add data
                    inputs += coord_data['inputs']
                    labels += coord_data['labels']
                    label_masks += coord_data['label_masks']

                masked_bases_percent = n_masked_bases / (coord.length * 2) * 100
                # no need to split
                if self.only_test_set:
                    self._save_data(self.h5_test, inputs, labels, label_masks, chunk_size,
                                    timestep_len)
                    print(('{}/{} Numerified {} of species {} with {} features in {} chunks '
                           'and a base level error rate of {:.2f}%').format(
                               i + 1, len(all_coords), coord, coord.genome.species,
                               len(coord.features), len(inputs), masked_bases_percent))
                else:
                    # split and save
                    train_data, val_data = _split_data([inputs, labels, label_masks], test_size=0.2)
                    if train_data[0]:
                        self._save_data(self.h5_train, *train_data, chunk_size, timestep_len)
                    if val_data[0]:
                        self._save_data(self.h5_val, *val_data, chunk_size, timestep_len)
                    print(('{}/{} Numerified {} of species {} with {} features in {} chunks '
                           '(train: {}, test: {}) and a base level error rate of {:.2f}%').format(
                               i + 1, len(all_coords), coord, coord.genome.species,
                               len(coord.features), len(inputs), len(train_data[0]),
                               len(val_data[0]), masked_bases_percent))
            else:
                print('{}/{} Skipping {} of species {} as it has no features'.format(
                    i + 1, len(all_coords), coord, coord.genome.species))
        self.close_files()
