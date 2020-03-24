import os
import h5py
import numpy as np
import random
import datetime
import subprocess
from itertools import compress
from sklearn.model_selection import train_test_split

import geenuff, helixerprep
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

    @staticmethod
    def _create_dataset(h5_file, key, matrix, dtype):
        shape = list(matrix.shape)
        shuffle = len(shape) > 1

        h5_file.create_dataset(key,
                               shape=shape,
                               maxshape=tuple([None] + shape[1:]),
                               chunks=tuple([1] + shape[1:]),
                               dtype=dtype,
                               compression='lzf',
                               shuffle=shuffle)  # only for the compression

    def _save_data(self, h5_file, flat_data, h5_group='/data/'):
        # todo, pull from flat_data dict
        dset_keys = [
            'X', 'y', 'sample_weights', 'gene_lengths', 'transitions', 'err_samples',
            'fully_intergenic_samples', 'start_ends', 'species', 'seqids'
        ]
        dsets = {key: None for key in dset_keys}

        # keys of the arrays that need to be padded

        assert len(set(mat_info.matrix.shape[0] for mat_info in flat_data)) == 1, 'unequal data lengths'

        # setup keys
        n_seqs = flat_data[0].matrix.shape[0]  # should be y, but should also not matter
        # append to or create datasets
        if h5_group + 'y' in h5_file:
            for dset_key in dsets.keys():
                dset = h5_file[h5_group + dset_key]
                old_len = dset.shape[0]
                dset.resize(old_len + n_seqs, axis=0)
        else:
            old_len = 0
            for mat_info in flat_data:
                self._create_dataset(h5_file, h5_group + mat_info.key, mat_info.matrix, mat_info.dtype)

        # writing to the h5 file
        for mat_info in flat_data:
            h5_file[h5_group + mat_info.key][old_len:] = mat_info.matrix
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
            try:
                attrs[module.__name__ + '_commit'] = subprocess.check_output(cmd).strip().decode()
            except subprocess.CalledProcessError:
                attrs[module.__name__ + '_commit'] = 'error'
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

    def _numerify_coord(self, coord, coord_features, chunk_size, keep_errors, one_hot):
        print(chunk_size, 'chunk size')
        coord_data = CoordNumerifier.numerify(coord, coord_features, chunk_size,
                                              one_hot)
        # keep track of variables
        y = [cd.matrix for cd in coord_data if cd.key == 'y'][0]
        sample_weights = [cd.matrix for cd in coord_data if cd.key == 'sample_weights'][0]
        n_seqs = y.shape[0]
        n_masked_bases = sum([np.count_nonzero(m == 0) for m in sample_weights])  # todo, use numpy
        n_ig_bases = sum([np.count_nonzero(l[:, 0] == 1) for l in y])
        # filter out sequences that are completely masked as error
        if not keep_errors:
            valid_data = np.any(sample_weights, axis=1)
            n_invalid_seqs = n_seqs - np.sum(valid_data)
            if n_invalid_seqs > 0:
                for mat_info in coord_data:
                    mat_info.matrix = mat_info.matrix[valid_data]
                #for key in coord_data.keys():  # todo, different looping method
                #    coord_data[key] = list(compress(coord_data[key], valid_data))  # todo, should still be using numpy
        else:
            n_invalid_seqs = 0
        masked_bases_perc = n_masked_bases / (coord.length * 2) * 100
        ig_bases_perc = n_ig_bases / (coord.length * 2) * 100
        invalid_seqs_perc = n_invalid_seqs / n_seqs * 100
        return coord_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc

    def export(self, chunk_size, genomes, exclude, val_size, keep_errors, one_hot=True,
               all_transcripts=False):
        genome_coord_features = self.geenuff_exporter.genome_query(genomes, exclude,
                                                                   all_transcripts=all_transcripts)
        # make version without features for shorter downstream code
        genome_coords = {g_id: list(values.keys()) for g_id, values in genome_coord_features.items()}
        n_coords = sum([len(coords) for genome_id, coords in genome_coords.items()])
        print('\n{} coordinates chosen to numerify'.format(n_coords))

        train_coords, val_coords = self._split_coords_by_N90(genome_coords, val_size)
        n_coords_done = 1
        for genome_id, coords in genome_coords.items():
            for (coord_id, coord_len) in coords:
                coord = self.geenuff_exporter.get_coord_by_id(coord_id)
                coord_features = genome_coord_features[genome_id][(coord_id, coord_len)]
                numerify_outputs = self._numerify_coord(coord, coord_features, chunk_size, keep_errors,
                                                        one_hot)

                flat_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc = numerify_outputs
                if self.only_test_set:
                    self._save_data(self.h5_test, flat_data)
                    assigned_set = 'test'
                else:
                    if coord_id in train_coords:
                        self._save_data(self.h5_train, flat_data)
                        assigned_set = 'train'
                    else:
                        self._save_data(self.h5_val, flat_data)
                        assigned_set = 'val'
                print((f'{n_coords_done}/{n_coords} Numerified {coord} of {coord.genome.species} '
                       f"with {len(coord.features)} features in {flat_data[0].matrix.shape[0]} chunks, "
                       f'err rate: {masked_bases_perc:.2f}%, ig rate: {ig_bases_perc:.2f}%, '
                       f'fully err seqs: {invalid_seqs_perc:.2f}% ({assigned_set})'))
                n_coords_done += 1
                # free all datasets so we don't keep two all the time
                del flat_data

        self._add_data_attrs(genomes, exclude, keep_errors)
        self._close_files()
