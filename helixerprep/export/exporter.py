import os
import h5py
import numpy as np
import random
import datetime
import subprocess
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

        self.h5_coord_offset = 0

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

    def _save_data(self, h5_file, flat_data, h5_coords, n_chunks, first_round_for_coordinate, h5_group='/data/'):
        assert len(set(mat_info.matrix.shape[0] for mat_info in flat_data)) == 1, 'unequal data lengths'

        if first_round_for_coordinate:
            self._create_or_expand_datasets(h5_file, h5_group, flat_data, n_chunks)

        # h5_coords are relative for the coordinate/chromosome, so offset by previous length
        old_len = self.h5_coord_offset
        start = old_len + h5_coords[0]
        end = old_len + h5_coords[1]

        # writing to the h5 file
        for mat_info in flat_data:
            h5_file[h5_group + mat_info.key][start:end] = mat_info.matrix
        h5_file.flush()

    def _create_or_expand_datasets(self, h5_file, h5_group, flat_data, n_chunks):
        # append to or create datasets
        if h5_group + 'y' not in h5_file:
            for mat_info in flat_data:
                self._create_dataset(h5_file, h5_group + mat_info.key, mat_info.matrix, mat_info.dtype)

        old_len = h5_file[h5_group + flat_data[0].key].shape[0]
        self.h5_coord_offset = old_len
        for mat_info in flat_data:
            dset = h5_file[h5_group + mat_info.key]
            dset.resize(old_len + n_chunks, axis=0)

    @staticmethod
    def _create_dataset(h5_file, key, matrix, dtype):
        shape = list(matrix.shape)
        shuffle = len(shape) > 1
        shape[0] = 0  # create w/o size
        h5_file.create_dataset(key,
                               shape=shape,
                               maxshape=tuple([None] + shape[1:]),
                               chunks=tuple([1] + shape[1:]),
                               dtype=dtype,
                               compression='lzf',
                               shuffle=shuffle)  # only for the compression

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

    def _numerify_coord(self, coord, coord_features, chunk_size, one_hot, keep_featureless):
        """filtering and stats"""
        coord_data_gen = CoordNumerifier.numerify(coord, coord_features, chunk_size,
                                                  one_hot)

        # the following will all be used to calculated a percentage, which is yielded but ignored until the end
        n_chunks = 0
        n_invalid_chunks = 0
        n_featureless_chunks = 0
        n_bases = 0
        n_ig_bases = 0
        n_masked_bases = 0

        for coord_data, h5_coord in coord_data_gen:
            if not keep_featureless and not bool(coord_features):
                print('continuing w/o exporting one super-chunk {} on {}'.format(h5_coord, coord))
                continue  # don't process or export featureless coordinates unless explicitly requested
            # easy access to matrices
            y = [cd.matrix for cd in coord_data if cd.key == 'y'][0]
            x = [cd.matrix for cd in coord_data if cd.key == 'X'][0]
            sample_weights = [cd.matrix for cd in coord_data if cd.key == 'sample_weights'][0]

            # count things
            n_chunks += y.shape[0]
            n_masked_bases += np.sum(sample_weights == 0)  # sample weights already 0 where there's padding, ignore

            padded_bases = np.sum(1 - np.sum(x, axis=2))
            n_bases += np.prod(y.shape[:2]) - padded_bases

            if not one_hot:
                n_ig_bases += np.sum(y[:, :, 0])
            else:
                n_ig_bases += np.sum(1 - y[:, :, 0])  # where transcript is 1, it's genic, but this counts padding
                n_ig_bases -= padded_bases  # subtract padding

            masked_bases_perc = n_masked_bases / n_bases * 100
            ig_bases_perc = n_ig_bases / n_bases * 100
            invalid_chunks_perc = n_invalid_chunks / n_chunks * 100  # todo, rm, this is always 0% now
            featureless_chunks_perc = n_featureless_chunks / n_chunks * 100

            yield coord_data, coord, masked_bases_perc, ig_bases_perc, invalid_chunks_perc, featureless_chunks_perc, \
                  h5_coord

    def export(self, chunk_size, genomes, exclude, val_size, one_hot=True,
               all_transcripts=False, keep_featureless=False, h5_group='/data/'):
        keep_errors = True
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
                # calculate how many chunks will be produced
                n_chunks = coord_len // chunk_size
                if coord_len % chunk_size:
                    n_chunks += 1  # bc pad to size
                n_chunks *= 2  # for + & - strand

                coord = self.geenuff_exporter.get_coord_by_id(coord_id)
                coord_features = genome_coord_features[genome_id][(coord_id, coord_len)]
                numerify_outputs = self._numerify_coord(coord, coord_features, chunk_size,
                                                        one_hot, keep_featureless)
                first_round_for_coordinate = True
                for flat_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc, \
                    featureless_chunks_perc, h5_coord in numerify_outputs:
                    y = flat_data[0]
                    if self.only_test_set:
                        self._save_data(self.h5_test, flat_data, h5_coords=h5_coord, n_chunks=n_chunks,
                                        first_round_for_coordinate=first_round_for_coordinate, h5_group=h5_group)
                        assigned_set = 'test'
                    else:
                        if coord_id in train_coords:
                            self._save_data(self.h5_train, flat_data, h5_coords=h5_coord, n_chunks=n_chunks,
                                            first_round_for_coordinate=first_round_for_coordinate, h5_group=h5_group)
                            assigned_set = 'train'
                        else:
                            self._save_data(self.h5_val, flat_data, h5_coords=h5_coord, n_chunks=n_chunks,
                                            first_round_for_coordinate=first_round_for_coordinate, h5_group=h5_group)
                            assigned_set = 'val'
                    first_round_for_coordinate = False
                try:
                    print((f'{n_coords_done}/{n_coords} Numerified {coord} of {coord.genome.species} '
                           f"with {len(coord.features)} features in {flat_data[0].matrix.shape[0]} chunks, "
                           f'masked rate: {masked_bases_perc:.2f}%, ig rate: {ig_bases_perc:.2f}%, '
                           f'filtered fully err chunks: {invalid_seqs_perc:.2f}% ({assigned_set}), '
                           f'filtered chunks from featureless coordinates {featureless_chunks_perc:.2f}%'))
                except UnboundLocalError as e:
                    print('please fix me so I do not throw e at featureless coordinates.... anyway, swallowing:', e)
                n_coords_done += 1

        self._add_data_attrs(genomes, exclude, keep_errors)
        self._close_files()
