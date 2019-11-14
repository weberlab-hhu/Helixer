import os
import h5py
import copy
import time
import numpy as np
import random
import datetime
import subprocess
from itertools import compress
from collections import defaultdict
from sklearn.model_selection import train_test_split

import geenuff, helixerprep
from geenuff.base.orm import Coordinate, Genome, Feature
from geenuff.base.helpers import full_db_path
from geenuff.applications.exporter import GeenuffExportController
from .numerify import CoordNumerifier


class HelixerExportController(object):
    def __init__(self, db_path_in, data_dir, only_test_set=False):
        self.db_path_in = db_path_in
        self.only_test_set = only_test_set
        self.geenuff_exporter = GeenuffExportController(self.db_path_in, longest=True)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        elif os.listdir(data_dir):
            print('Output directory must be empty or not existing')
            exit()
        if self.only_test_set:
            print('Exporting all data into test_data.h5')
            self.h5_test = h5py.File(os.path.join(data_dir, 'test_data.h5'), 'w')
        else:
            print('Splitting data into training_data.h5 and validation_data.h5')
            self.h5_train = h5py.File(os.path.join(data_dir, 'training_data.h5'), 'w')
            self.h5_val = h5py.File(os.path.join(data_dir, 'validation_data.h5'), 'w')

    @staticmethod
    def _split_sequences(flat_data, val_size):
        """Basically does the same as sklearn.model_selection.train_test_split except
        it does not always fill the test arrays with at least one element.
        Expects the arrays to be in the order: inputs, labels, label_masks
        """
        train_arrays, val_arrays = {}, {}
        for key in flat_data:
            train_arrays[key] = []
            val_arrays[key] = []

        for i in range(len(flat_data['inputs'])):
            if random.random() > val_size:
                for key in flat_data:
                    train_arrays[key].append(flat_data[key][i])
            else:
                for key in flat_data:
                    val_arrays[key].append(flat_data[key][i])
        return train_arrays, val_arrays

    def _save_data(self, h5_file, flat_data, chunk_size, n_y_cols):
        inputs = flat_data['inputs']
        labels = flat_data['labels']
        label_masks = flat_data['label_masks']
        transitions = flat_data['transitions']
        # convert to numpy arrays

        # zero-pad each sequence to chunk_size
        # this is inefficient if there could be a batch with only sequences smaller than
        # chunk_size, but taking care of that introduces a lot of extra complexity
        n_seq = len(inputs)
        X = np.zeros((n_seq, chunk_size, 4), dtype=inputs[0].dtype)
        y = np.zeros((n_seq, chunk_size, n_y_cols), dtype=labels[0].dtype)
        y_transitions = np.zeros((n_seq, chunk_size, 6), dtype=transitions[0].dtype)
        sample_weights = np.zeros((n_seq, chunk_size), dtype=label_masks[0].dtype)

        for j in range(n_seq):
            sample_len = len(inputs[j])
            X[j, :sample_len, :] = inputs[j]
            y[j, :sample_len, :] = labels[j]
            y_transitions[j, :sample_len, :] = transitions[j]
            sample_weights[j, :sample_len] = label_masks[j]

        err_samples = np.any(sample_weights == 0, axis=1)
        # just one entry per chunk
        if n_y_cols > 3:
            fully_intergenic_samples = np.all(y[:, :, 0] == 1, axis=1)
        else:
            fully_intergenic_samples = np.all(y[:, :, 0] == 0, axis=1)
        gc_contents = np.array(flat_data['gc_contents'], dtype=np.uint64)
        coord_lengths = np.array(flat_data['coord_lengths'], dtype=np.uint64)
        start_ends = np.array(flat_data['start_ends'], dtype=np.int64)

        # check if this is the first batch to save
        dset_keys = [
            'X', 'y', 'sample_weights', 'gc_contents', 'coord_lengths', 'err_samples',
            'fully_intergenic_samples', 'start_ends', 'species', 'seqids', 'transitions'
        ]
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
                                   shape=(n_seq, chunk_size, n_y_cols),
                                   maxshape=(None, chunk_size, n_y_cols),
                                   chunks=(1, chunk_size, n_y_cols),
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
            h5_file.create_dataset('/data/gc_contents',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='uint64',
                                   compression='lzf')
            h5_file.create_dataset('/data/coord_lengths',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='uint64',
                                   compression='lzf')
            h5_file.create_dataset('/data/err_samples',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='bool',
                                   compression='lzf')
            h5_file.create_dataset('/data/fully_intergenic_samples',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='bool',
                                   compression='lzf')
            h5_file.create_dataset('/data/species',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='S25',
                                   compression='lzf')
            h5_file.create_dataset('/data/seqids',
                                   shape=(n_seq,),
                                   maxshape=(None,),
                                   dtype='S50',
                                   compression='lzf')
            h5_file.create_dataset('/data/start_ends',
                                   shape=(n_seq, 2),
                                   maxshape=(None, 2),
                                   dtype='int64',
                                   compression='lzf')
            h5_file.create_dataset('/data/transitions',
                                   shape=(n_seq, chunk_size, 6),
                                   maxshape=(None, chunk_size, 6),
                                   chunks=(1, chunk_size, 6),
                                   dtype='int8',  # guess we'll stick to int8
                                   compression='lzf',
                                   shuffle=True)
            # add new data
        dsets = [X, y, sample_weights, gc_contents, coord_lengths, err_samples,
                 fully_intergenic_samples, start_ends, flat_data['species'], flat_data['seqids'],
                 y_transitions]
        for dset_key, data in zip(dset_keys, dsets):
            h5_file['/data/' + dset_key][old_len:] = data
        h5_file.flush()

    def _split_coords_by_N90(self, genome_coords, val_size):
        """Splits the given coordinates in a train and val set. It does so by doing it individually for
        each the coordinates < N90 and >= N90 of each genome."""
        def N90_index(coords):
            len_90_perc = int(sum([c[1] for c in coords]) * 0.9)
            len_sum = 0
            for i, coord in enumerate(coords):
                len_sum += coord[1]
                if len_sum >= len_90_perc:
                    return i

        train_coord_ids, val_coord_ids = [], []
        for coords in genome_coords.values():
            n90_idx = N90_index(coords) + 1
            coord_ids = [c[0] for c in coords]
            for n90_split in [coord_ids[:n90_idx], coord_ids[n90_idx:]]:
                if len(n90_split) < 2:
                    # if there is no way to split a half only add to training data
                    train_coord_ids += n90_split
                else:
                    genome_train_coord_ids, genome_val_coord_ids = train_test_split(n90_split,
                                                                                    test_size=val_size)
                    train_coord_ids += genome_train_coord_ids
                    val_coord_ids += genome_val_coord_ids
        return train_coord_ids, val_coord_ids

    def _add_data_attrs(self, genomes, exclude, keep_errors):
        attrs = {
            'timestamp': str(datetime.datetime.now()),
            'genomes': ','.join(genomes),
            'exclude': ','.join(exclude),
            'keep_errors': str(keep_errors),
        }
        # get GeenuFF and Helixer commit hashes
        for module in [geenuff, helixerprep]:
            os.chdir(os.path.dirname(module.__file__))
            cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
            attrs[module.__name__ + '_commit'] = subprocess.check_output(cmd).strip().decode()
        # insert attrs into .h5 file
        for key, value in attrs.items():
            if self.only_test_set:
                self.h5_test.attrs[key] = value
            else:
                self.h5_train.attrs[key] = value
                self.h5_val.attrs[key] = value

    def _close_files(self):
        if self.only_test_set:
            self.h5_test.close()
        else:
            self.h5_train.close()
            self.h5_val.close()

    def _numerify_coord(self, coord, coord_features, chunk_size, keep_errors):
        coord_data = CoordNumerifier.numerify(self.geenuff_exporter, coord, coord_features, chunk_size)
        # keep track of variables
        n_seqs = len(coord_data['labels'])
        n_masked_bases = sum([np.count_nonzero(m == 0) for m in coord_data['label_masks']])
        n_ig_bases = sum([np.count_nonzero(l[:, 0] == 1) for l in coord_data['labels']])
        # filter out sequences that are completely masked as error
        if not keep_errors:
            valid_data = [s.any() for s in coord_data['label_masks']]
            n_invalid_seqs = n_seqs - valid_data.count(True)
            if n_invalid_seqs > 0:
                for key in coord_data.keys():
                    coord_data[key] = list(compress(coord_data[key], valid_data))
        else:
            n_invalid_seqs = 0
        masked_bases_perc = n_masked_bases / (coord.length * 2) * 100
        ig_bases_perc = n_ig_bases / (coord.length * 2) * 100
        invalid_seqs_perc = n_invalid_seqs / n_seqs * 100
        return coord_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc

    def export(self, chunk_size, genomes, exclude, val_size, keep_errors):
        genome_coord_features = self.geenuff_exporter.genome_query(genomes, exclude)
        # make version without features for shorter downstream code
        genome_coords = {g_id: list(values.keys()) for g_id, values in genome_coord_features.items()}
        n_coords = sum([len(coords) for genome_id, coords in genome_coords.items()])
        print('\n{} coordinates chosen to numerify'.format(n_coords))

        train_coords, val_coords = self._split_coords_by_N90(genome_coords, val_size)
        n_coords_done = 1
        n_y_cols = 4
        for genome_id, coords in genome_coords.items():
            for (coord_id, coord_len) in coords:
                coord = self.geenuff_exporter.get_coord_by_id(coord_id)
                coord_features = genome_coord_features[genome_id][(coord_id, coord_len)]
                numerify_outputs = self._numerify_coord(coord, coord_features, chunk_size, keep_errors)

                flat_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc = numerify_outputs
                if self.only_test_set:
                    self._save_data(self.h5_test, flat_data, chunk_size, n_y_cols)
                    assigned_set = 'test'
                else:
                    if coord_id in train_coords:
                        self._save_data(self.h5_train, flat_data, chunk_size, n_y_cols)
                        assigned_set = 'train'
                    else:
                        self._save_data(self.h5_val, flat_data, chunk_size, n_y_cols)
                        assigned_set = 'val'
                print((f'{n_coords_done}/{n_coords} Numerified {coord} of {coord.genome.species} '
                       f"with {len(coord.features)} features in {len(flat_data['inputs'])} chunks, "
                       f'err rate: {masked_bases_perc:.2f}%, ig rate: {ig_bases_perc:.2f}%, '
                       f'fully err seqs: {invalid_seqs_perc:.2f}% ({assigned_set})'))
                n_coords_done += 1
                # free all datasets so we don't keep two all the time
                dset_keys = list(flat_data.keys())
                for key in dset_keys:
                    del flat_data[key]

        self._add_data_attrs(genomes, exclude, keep_errors)
        self._close_files()
