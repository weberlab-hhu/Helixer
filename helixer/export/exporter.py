import os
import time
import h5py
import glob
import numpy as np
import random
import sqlite3
import datetime
import subprocess
import pkg_resources
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split

import geenuff
import helixer
from geenuff.applications.exporter import GeenuffExportController
from .numerify import CoordNumerifier
from ..core.helpers import get_sp_seq_ranges, file_stem

from collections import defaultdict


class HelixerExportController(object):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'

    def __init__(self, main_db_path, data_dir, only_test_set=False, match_existing=False, h5_group='/data/'):

        def check_db(path):
            genome_name_file = file_stem(path)
            conn = sqlite3.connect(path)
            c = conn.cursor()
            c.execute('''SELECT species from genome;''')
            genome_name_db = c.fetchall()
            conn.close()

            assert len(genome_name_db) == 1, f'{path} is not a valid db as it contains more than one genome'
            msg = f'the name of {path} must match the species name inside:'
            assert genome_name_file.lower() == genome_name_db[0][0].lower(), msg
            print(f'{path} looks good')

        self.only_test_set = only_test_set
        db_paths = {file_stem(db_path):db_path for db_path in glob.glob(os.path.join(main_db_path, '*.sqlite3'))}
        # open connections to all dbs in path
        start_time = time.time()
        with ThreadPool(8) as p:
            p.map(check_db, list(db_paths.values()))
        self.exporters = {genome:GeenuffExportController(path, longest=True) for genome, path in db_paths.items()}
        print(f'Checked and opened connections to {len(db_paths)} dbs in {time.time() - start_time:.2f} seconds')

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
        self.species_seq_ranges = {}
        self.current_sp_start_ends = {}
        if self.match_existing:
            for key in self.h5:
                self.species_seq_ranges[key] = get_sp_seq_ranges(self.h5[key])

    def _set_current_sp_start_ends(self, species):
        for key in self.h5:
            sp_start_ends = {}  # {seqid: {'seqid': np.array([(0, 20k), (20k, 40k), ...]),
            #                              'seqid2': ...}}
            sp_ranges = self.species_seq_ranges[key][species]
            sp_se_array = self.h5[key]['data/start_ends'][sp_ranges['start']:sp_ranges['end']]
            sp_start = sp_ranges['start']
            for seqid, ranges in sp_ranges['seqids'].items():
                rel_start, rel_end = [i - sp_start for i in ranges]
                length = rel_end - rel_start
                tuple_array = np.zeros(shape=(length,), dtype=tuple)
                for i, se in enumerate(sp_se_array[rel_start:rel_end]):
                    tuple_array[i] = tuple(se)
                sp_start_ends[seqid] = tuple_array
            self.current_sp_start_ends[key] = sp_start_ends

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
            # find out exactly where things align with existing
            start_ends = [x.matrix for x in flat_data if x.key == "start_ends"][0]
            seqid = [x.matrix for x in flat_data if x.key == "seqids"][0][0]
            tuple_start_ends = np.zeros(shape=(start_ends.shape[0],), dtype=tuple)
            for i, pair in enumerate(start_ends):
                tuple_start_ends[i] = tuple(pair)

            _, idx_existing, idx_new = np.intersect1d(
                self.current_sp_start_ends[assigned_set][seqid],
                tuple_start_ends,
                return_indices=True,
                assume_unique=True
            )
            start = min(idx_existing) + old_len
            end = max(idx_existing) + old_len + 1

            mask = sorted(idx_new)

            for mat_info in flat_data:
                mat_info.matrix = mat_info.matrix[mask]

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
        if self.match_existing:
            # due to potential filtering in existing we likely won't write the full n_chunks
            sp = [x.matrix for x in flat_data if x.key == 'species'][0][0]
            seqid = [x.matrix for x in flat_data if x.key == 'seqids'][0][0]
            seq_coordinates = self.species_seq_ranges[assigned_set][sp]['seqids'][seqid]
            n_chunks = seq_coordinates[1] - seq_coordinates[0]

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
            for coord_id, sp, seqid, _ in self._gen_sp_seqid(genome_coords):
                if seqid in test_in_h5[sp]:
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
            for coord_id, sp, seqid, _ in self._gen_sp_seqid(genome_coords):
                if seqid in val_in_h5[sp]:
                    val_coord_ids.append(coord_id)
                elif seqid in train_in_h5[sp]:
                    train_coord_ids.append(coord_id)
                else:
                    print('{} not found in existing h5s, maybe featureless in existing, '
                          'but worry if you see a whole lot of this warning...'.format(seqid))
            return train_coord_ids, val_coord_ids

    def _gen_sp_seqid(self, genome_coords):
        for genome_name in genome_coords:
            for i, (coord_id, coord_len) in enumerate(genome_coords[genome_name]):
                # get coord (to later access species name and sequence name)
                coord = self.exporters[genome_name].get_coord_by_id(coord_id)
                if i == 0:
                    sp = genome_name.encode('ASCII')
                seqid = coord.seqid.encode('ASCII')
                yield coord_id, sp, seqid, coord_len

    def _get_sp_seqids_from_h5(self, assigned_set, by=1000):
        """from flat h5 datasets to dict with {species: [seqid, seqid2, ...], ...}"""
        # todo, is this fully redundant with get_sp_seq_ranges?
        h5 = self.h5[assigned_set]
        out = defaultdict(dict)
        for i in range(0, h5['data/y'].shape[0], by):
            species = h5['data/species'][i:(i + by)]
            seqids = h5['data/seqids'][i:(i + by)]
            for j in range(len(seqids)):
                out[species[j]][seqids[j]] = 0
        out = dict(out)
        for key in out:
            out[key] = list(out[key].keys())
        return dict(out)

    def _resort_genome_coords_from_existing(self, genome_coords):
        # setup keys to go from the h5 naming to the db naming
        pre_decode = self._gen_sp_seqid(genome_coords)
        ckey = defaultdict(dict)
        for coord_id, sp, seqid, coord_len in pre_decode:
            sp = sp.decode()
            ckey[sp][seqid] = (coord_id, coord_len)

        # pull ordering from h5s, but replace with ids from db
        genome_coords = defaultdict(list)
        for assigned_set in self.h5:
            in_h5 = self._get_sp_seqids_from_h5(assigned_set)
            for sp in in_h5:
                sp_str = sp.decode()
                for seqid in in_h5[sp]:
                    coord_tuple = ckey[sp_str][seqid]
                    genome_coords[sp_str].append(coord_tuple)
        return genome_coords

    def _add_data_attrs(self, genomes, keep_errors):
        attrs = {
            'timestamp': str(datetime.datetime.now()),
            'genomes': ','.join(genomes),
            'keep_errors': str(keep_errors),
        }
        # get GeenuFF and Helixer commit hashes
        pwd = os.getcwd()
        for module in [geenuff, helixer]:
            os.chdir(os.path.dirname(module.__file__))
            cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
            try:
                attrs[module.__name__ + '_commit'] = subprocess.check_output(cmd, stderr=subprocess.STDOUT).\
                    strip().decode()
            except subprocess.CalledProcessError:
                attrs[module.__name__ + '_commit'] = 'commit not found, version: {}'.format(
                    pkg_resources.require(module.__name__)[0].version
                )
                print('logged installed version in place of git commit for {}'.format(module.__name__))
        os.chdir(pwd)
        # insert attrs into .h5 file
        for key, value in attrs.items():
            for assigned_set in self.h5:
                self.h5[assigned_set].attrs[key] = value

    def _close_files(self):
        for key in self.h5:
            self.h5[key].close()

    def _numerify_coord(self, coord, coord_features, chunk_size, one_hot, keep_featureless,
                        write_by, modes, multiprocess):
        """filtering and stats"""
        coord_data_gen = CoordNumerifier.numerify(coord, coord_features, chunk_size, one_hot,
                                                  write_by=write_by, mode=modes, multiprocess=multiprocess)

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

    def export(self, chunk_size, genomes, val_size, one_hot=True,
               all_transcripts=False, keep_featureless=False, write_by=10_000_000_000,
               modes=('X', 'y', 'anno_meta', 'transitions'), multiprocess=True):
        h5_group = self.h5_group
        keep_errors = True
        genome_coord_features = {genome_name: self.exporters[genome_name].genome_query(all_transcripts=all_transcripts)
                                 for genome_name in genomes}
        # make version without features for shorter downstream code
        genome_coords = {genome_name: list(values.keys()) for genome_name, values in genome_coord_features.items()}

        n_coords = sum([len(coords) for genome_name, coords in genome_coords.items()])
        print('\n{} coordinates chosen to numerify'.format(n_coords))
        if self.match_existing:
            train_coords, val_coords = self._split_coords_by_existing(genome_coords=genome_coords)
            # resort coordinates to match existing as well (bc length sort doesn't work on ties)
            genome_coords = self._resort_genome_coords_from_existing(genome_coords)
        else:
            train_coords, val_coords = self._split_coords_by_N90(genome_coords, val_size)
        n_coords_done = 1
        n_writing_chunks = 0
        prev_genome_name = None
        for genome_name, coords in genome_coords.items():
            if genome_name != prev_genome_name and self.match_existing:
                # once per species, setup dicts with {seqid: [(start,end), ...], ...} for each h5
                # this avoids excessive random disk reads
                self._set_current_sp_start_ends(genome_name.encode('ASCII'))

            for (coord_id, coord_len) in coords:
                start_time = time.time()
                # calculate how many chunks will be produced
                n_chunks = coord_len // chunk_size
                if coord_len % chunk_size:
                    n_chunks += 1  # bc pad to size
                n_chunks *= 2  # for + & - strand

                coord = self.exporters[genome_name].get_coord_by_id(coord_id)
                coord_features = genome_coord_features[genome_name][(coord_id, coord_len)]
                numerify_outputs = self._numerify_coord(coord, coord_features, chunk_size,
                                                        one_hot, keep_featureless, write_by=write_by,
                                                        modes=modes, multiprocess=multiprocess)
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
                    print(f'{n_coords_done}/{n_coords} Numerified {coord} of {genome_name} '
                          f"with {len(coord.features)} features in {flat_data[0].matrix.shape[0]} chunks, "
                          f'masked rate: {masked_bases_perc:.2f}%, ig rate: {ig_bases_perc:.2f}%, '
                          f'filtered fully err chunks: {invalid_seqs_perc:.2f}% ({assigned_set}), '
                          f'filtered chunks from featureless coordinates {featureless_chunks_perc:.2f}% '
                          f'({time.time() - start_time:.2f} secs)', end='\n\n')
                except UnboundLocalError as e:
                    print('please fix me so I do not throw e at featureless coordinates.... anyway, swallowing:', e)
                n_coords_done += 1
            prev_genome_name = genome_name
        self._add_data_attrs(genomes, keep_errors)
        self._close_files()
        print('Export from geenuff db to h5 file(s) with numeric matrices finished successfully.')
        return n_writing_chunks  # for testing only atm

