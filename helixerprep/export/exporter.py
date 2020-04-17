import os
import h5py
import numpy as np
import random
import datetime
import subprocess
from sklearn.model_selection import train_test_split

import geenuff
import helixerprep
from geenuff.applications.exporter import GeenuffExportController
from .numerify import CoordNumerifier

from collections import defaultdict


class HelixerExportController(object):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'

    def __init__(self, db_path_in, data_dir, only_test_set=False, match_existing=False, h5_group='/data/'):
        self.db_path_in = db_path_in
        self.only_test_set = only_test_set
        self.geenuff_exporter = GeenuffExportController(self.db_path_in, longest=True)

        # h5 export details
        self.match_existing = match_existing
        self.h5_group = h5_group
        if match_existing:
            mode = 'a'
            # confirm files exist
            if self.only_test_set:
                assert os.path.exists(os.path.join(data_dir, 'test_data.h5')), 'data_dir lacks expected test_data.h5'
            else:
                for h5 in ['training_data.h5', 'validation_data.h5']:
                    assert os.path.exists(os.path.join(data_dir, h5)), 'data_dir lacks expected {}'.format(h5)
        else:
            mode = 'w'
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            elif os.listdir(data_dir):
                print('Output directory must be empty or not existing')
                exit()
        self.h5 = {}
        if self.only_test_set:
            print('Exporting all data into test_data.h5')
            self.h5[HelixerExportController.TEST] = h5py.File(os.path.join(data_dir, 'test_data.h5'), mode)
        else:
            print('Splitting data into training_data.h5 and validation_data.h5')
            self.h5[HelixerExportController.TRAIN] = h5py.File(os.path.join(data_dir, 'training_data.h5'), mode)
            self.h5[HelixerExportController.VAL] = h5py.File(os.path.join(data_dir, 'validation_data.h5'), mode)

        self.h5_coord_offset = {HelixerExportController.TEST: 0,
                                HelixerExportController.VAL: 0,
                                HelixerExportController.TRAIN: 0}

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

    def _save_data(self, flat_data, h5_coords, n_chunks, first_round_for_coordinate,
                   assigned_set, h5_group='/data/'):
        h5_file = self.h5[assigned_set]
        assert len(set(mat_info.matrix.shape[0] for mat_info in flat_data)) == 1, 'unequal data lengths'

        if first_round_for_coordinate:
            self._create_or_expand_datasets(h5_file, h5_group, flat_data, n_chunks, assigned_set)

        # h5_coords are relative for the coordinate/chromosome, so offset by previous length
        old_len = self.h5_coord_offset[assigned_set]
        start = old_len + h5_coords[0]
        end = old_len + h5_coords[1]

        # sanity check sort order
        if self.match_existing:
            # take species, seqids, and start_ends and assert the match those under '/data/'
            for key in ['seqids', 'start_ends', 'species']:
                expected = h5_file['/data/' + key][start:end]
                for mat_info in flat_data:
                    if mat_info.key == key:
                        assert np.all(mat_info.matrix == expected), '{} != {} :-('.format(mat_info.matrix, expected)

        # writing to the h5 file
        for mat_info in flat_data:
            h5_file[h5_group + mat_info.key][start:end] = mat_info.matrix
        h5_file.flush()

    def _create_or_expand_datasets(self, h5_file, h5_group, flat_data, n_chunks, assigned_set):
        # append to or create datasets
        if h5_group + 'y' not in h5_file:
            for mat_info in flat_data:
                self._create_dataset(h5_file, h5_group + mat_info.key, mat_info.matrix, mat_info.dtype)

        old_len = h5_file[h5_group + flat_data[0].key].shape[0]
        self.h5_coord_offset[assigned_set] = old_len
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

    def _split_coords_by_existing(self, genome_coords):
        if self.only_test_set:
            val_coord_ids = []
            test_in_h5 = self._get_sp_seqids_from_h5(HelixerExportController.TEST)
            for g_id, coord_id, sp, seqid in self._gen_sp_seqid(genome_coords):
                if seqid in test_in_h5:
                    val_coord_ids.append(coord_id)
                else:
                    print('{} not found in existing h5s, maybe featureless in existing, '
                          'but worry if you see a whole lot of this warning...'.format(seqid))
            return [], val_coord_ids
        else:
            train_coord_ids = []
            val_coord_ids = []
            train_in_h5 = self._get_sp_seqids_from_h5(HelixerExportController.TRAIN)
            val_in_h5 = self._get_sp_seqids_from_h5(HelixerExportController.VAL)
            # prep all coordinate IDs so that they'll match that from h5
            for g_id, coord_id, sp, seqid in self._gen_sp_seqid(genome_coords):
                if seqid in val_in_h5[sp]:
                    val_coord_ids.append(coord_id)
                elif seqid in train_in_h5[sp]:
                    train_coord_ids.append(coord_id)
                else:
                    print('{} not found in existing h5s, maybe featureless in existing, '
                          'but worry if you see a whole lot of this warning...'.format(seqid))
            return train_coord_ids, val_coord_ids

    def _gen_sp_seqid(self, genome_coords):
        for g_id in genome_coords:
            first4genome = True
            for coord_id, coord_len in genome_coords[g_id]:
                # get coord (to later access species name and sequence name)
                coord = self.geenuff_exporter.get_coord_by_id(coord_id)
                if first4genome:
                    sp = coord.genome.species.encode('ASCII')
                    first4genome = False
                seqid = coord.seqid.encode('ASCII')
                yield g_id, coord_id, sp, seqid

    def _get_sp_seqids_from_h5(self, assigned_set, by=1000):
        """from flat h5 datasets to dict with {species: [seqid, seqid2, ...], ...}"""
        h5 = self.h5[assigned_set]
        out = defaultdict(set)
        for i in range(0, h5['data/y'].shape[0], by):
            species = h5['data/species'][i:(i + by)]
            seqids = h5['data/seqids'][i:(i + by)]
            for j in range(len(seqids)):
                out[species[j]].add(seqids[j])
        return dict(out)

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
            for assigned_set in self.h5:
                self.h5[assigned_set].attrs[key] = value

    def _close_files(self):
        for key in self.h5:
            self.h5[key].close()

    def _numerify_coord(self, coord, coord_features, chunk_size, one_hot, keep_featureless, write_by, modes):
        """filtering and stats"""
        coord_data_gen = CoordNumerifier.numerify(coord, coord_features, chunk_size,
                                                  one_hot, write_by=write_by, mode=modes)

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
            sample_weights = [cd.matrix for cd in coord_data if cd.key == 'sample_weights'][0]

            # count things
            n_chunks += y.shape[0]
            n_masked_bases += np.sum(sample_weights == 0)  # sample weights already 0 where there's padding, ignore

            padded_bases = np.sum(1 - np.sum(y, axis=2))  # todo, this fails on one-hot...
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
               all_transcripts=False, keep_featureless=False, write_by=10_000_000_000,
               modes=('X', 'y', 'anno_meta', 'transitions')):
        h5_group = self.h5_group
        keep_errors = True
        genome_coord_features = self.geenuff_exporter.genome_query(genomes, exclude,
                                                                   all_transcripts=all_transcripts)
        # make version without features for shorter downstream code
        genome_coords = {g_id: list(values.keys()) for g_id, values in genome_coord_features.items()}

        n_coords = sum([len(coords) for genome_id, coords in genome_coords.items()])
        print('\n{} coordinates chosen to numerify'.format(n_coords))
        if self.match_existing:
            train_coords, val_coords = self._split_coords_by_existing(genome_coords=genome_coords)
        else:
            train_coords, val_coords = self._split_coords_by_N90(genome_coords, val_size)
        n_coords_done = 1
        n_writing_chunks = 0
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
                                                        one_hot, keep_featureless, write_by=write_by,
                                                        modes=modes)
                first_round_for_coordinate = True
                for flat_data, coord, masked_bases_perc, ig_bases_perc, invalid_seqs_perc, \
                    featureless_chunks_perc, h5_coord in numerify_outputs:
                    if self.only_test_set:
                        if self.match_existing:
                            if coord_id in val_coords + train_coords:
                                assigned_set = HelixerExportController.TEST
                            else:
                                continue
                        else:
                            assigned_set = HelixerExportController.TEST
                    else:
                        if coord_id in val_coords:
                            assigned_set = HelixerExportController.VAL
                        elif coord_id in train_coords:
                            assigned_set = HelixerExportController.TRAIN
                        else:
                            print('set could not be assigned for {}, continuing without saving'.format(coord.seqid))
                            assert self.match_existing
                            continue

                    self._save_data(flat_data, h5_coords=h5_coord, n_chunks=n_chunks,
                                    first_round_for_coordinate=first_round_for_coordinate, h5_group=h5_group,
                                    assigned_set=assigned_set)
                    first_round_for_coordinate = False
                    n_writing_chunks += 1
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
        return n_writing_chunks  # for testing only atm
