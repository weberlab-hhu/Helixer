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
from geenuff.applications.importer import FastaImporter
from .numerify import CoordNumerifier
from ..core.helpers import get_sp_seq_ranges, file_stem

from collections import defaultdict


class HelixerExportControllerBase(object):

    def __init__(self, input_path, output_path, match_existing=False):
        self.input_path = input_path
        self.output_path = output_path
        self.match_existing = match_existing

    @staticmethod
    def calc_n_chunks(coord_len, chunk_size):
        """calculates the number of chunks resulting from a coord len and chunk size"""
        n_chunks = coord_len // chunk_size
        if coord_len % chunk_size:
            n_chunks += 1  # bc pad to size
        n_chunks *= 2  # for + & - strand
        return n_chunks

    @staticmethod
    def _create_dataset(h5_file, key, matrix, dtype, compression='gzip', create_empty=True):
        shape = list(matrix.shape)
        shuffle = len(shape) > 1
        if create_empty:
            shape[0] = 0  # create w/o size
        h5_file.create_dataset(key,
                               shape=shape,
                               maxshape=tuple([None] + shape[1:]),
                               chunks=tuple([1] + shape[1:]),
                               dtype=dtype,
                               compression=compression,
                               shuffle=shuffle)  # only for the compression

    def _create_or_expand_datasets(self, h5_group, flat_data, n_chunks, compression='gzip'):
        if h5_group not in self.h5 or len(self.h5[h5_group].keys()) == 0:
            for mat_info in flat_data:
                self._create_dataset(self.h5, h5_group + mat_info.key, mat_info.matrix, mat_info.dtype, compression)

        old_len = self.h5[h5_group + flat_data[0].key].shape[0]
        self.h5_coord_offset = old_len
        for mat_info in flat_data:
            self.h5[h5_group + mat_info.key].resize(old_len + n_chunks, axis=0)

    def _save_data(self, flat_data, h5_coords, n_chunks, first_round_for_coordinate, compression='gzip',
                   h5_group='/data/'):
        assert len(set(mat_info.matrix.shape[0] for mat_info in flat_data)) == 1, 'unequal data lengths'

        if first_round_for_coordinate:
            self._create_or_expand_datasets(h5_group, flat_data, n_chunks, compression)

        # h5_coords are relative for the coordinate/chromosome, so offset by previous length
        old_len = self.h5_coord_offset
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
                self.current_sp_start_ends[seqid],
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
                expected = self.h5['/data/' + key][start:end]
                for mat_info in flat_data:
                    if mat_info.key == key:
                        assert np.all(mat_info.matrix == expected), '{} != {} :-('.format(mat_info.matrix, expected)

        # writing to the h5 file
        for mat_info in flat_data:
            self.h5[h5_group + mat_info.key][start:end] = mat_info.matrix
        self.h5.flush()

    def _add_data_attrs(self, genomes):
        attrs = {
            'timestamp': str(datetime.datetime.now()),
            'genomes': ','.join(genomes),
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
            self.h5.attrs[key] = value


class HelixerFastaToH5Controller(HelixerExportControllerBase):

    class CoordinateSurrogate(object):
        """Mimics some functionatity of the Coordinate orm class, so we can go directly from FASTA to H5"""
        def __init__(self, seqid, seq):
            self.seqid = seqid
            self.sequence = seq
            self.length = len(seq)

        def __repr__(self):
            return f'Fasta only Coordinate (seqid: {self.seqid}, len: {self.length})'

    def export_fasta_to_h5(self, chunk_size, genome, compression, multiprocess):
        fasta_importer = FastaImporter(genome)
        fasta_seqs = fasta_importer.parse_fasta(self.input_path)
        self.h5 = h5py.File(self.output_path, 'w')
        print(f'Effectively set --modes to "X" and --write-by to infinite due to '
              f'--direct-fasta-to-h5-path being set')

        for i, (seqid, seq) in enumerate(fasta_seqs):
            start_time = time.time()
            coord = HelixerFastaToH5Controller.CoordinateSurrogate(seqid, seq)
            n_chunks = HelixerExportControllerBase.calc_n_chunks(coord.length, chunk_size)
            data_gen = CoordNumerifier.numerify_only_fasta(coord, chunk_size, genome, multiprocess=multiprocess)
            for j, data in enumerate(data_gen):
                self._save_data(data, h5_coords=(0, len(data[0].matrix)), n_chunks=n_chunks,
                                first_round_for_coordinate=(j == 0), compression=compression)
            print(f'{i + 1} Numerified {coord} of {genome} in {time.time() - start_time:.2f} secs', end='\n\n')
        self._add_data_attrs([genome])
        self.h5.close()


class HelixerExportController(HelixerExportControllerBase):

    def __init__(self, input_path, output_path, match_existing=False, h5_group='/data/'):
        super().__init__(input_path, output_path, match_existing)
        self.h5_group = h5_group
        main_db_path = self.input_path

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

        db_paths = {file_stem(db_path): db_path for db_path in glob.glob(os.path.join(main_db_path, '*.sqlite3'))}
        assert os.path.exists(main_db_path) and db_paths, "main_db_paths, must be a folder containing one or more " \
                                                          "*.sqlite3 files, where the * matches the names from the " \
                                                          "--genomes parameter"
        # open connections to all dbs in path
        start_time = time.time()
        with ThreadPool(8) as p:
            p.map(check_db, list(db_paths.values()))
        self.exporters = {genome: GeenuffExportController(path, longest=True) for genome, path in db_paths.items()}
        print(f'Checked and opened connections to {len(db_paths)} dbs in {time.time() - start_time:.2f} seconds')

        if match_existing:
            # confirm files exist
            assert os.path.exists(self.output_path), 'output_path not existing'
            self.h5 = h5py.File(output_path, 'a')
            self.species_seq_ranges = get_sp_seq_ranges(self.h5)
        else:
            self.h5 = h5py.File(output_path, 'w')

        print(f'Exporting all data to {output_path}')
        self.h5_coord_offset = 0

    def _set_current_sp_start_ends(self, species):
        sp_start_ends = {}  # {seqid: {'seqid': np.array([(0, 20k), (20k, 40k), ...]),
        #                              'seqid2': ...}}
        sp_ranges = self.species_seq_ranges[species]
        sp_se_array = self.h5['data/start_ends'][sp_ranges['start']:sp_ranges['end']]
        sp_start = sp_ranges['start']
        for seqid, ranges in sp_ranges['seqids'].items():
            rel_start, rel_end = [i - sp_start for i in ranges]
            length = rel_end - rel_start
            tuple_array = np.zeros(shape=(length,), dtype=tuple)
            for i, se in enumerate(sp_se_array[rel_start:rel_end]):
                tuple_array[i] = tuple(se)
            sp_start_ends[seqid] = tuple_array
        self.current_sp_start_ends = sp_start_ends

    def _gen_sp_seqid(self, genome_coords):
        for genome_name in genome_coords:
            for i, (coord_id, coord_len) in enumerate(genome_coords[genome_name]):
                # get coord (to later access species name and sequence name)
                coord = self.exporters[genome_name].get_coord_by_id(coord_id)
                if i == 0:
                    sp = genome_name.encode('ASCII')
                seqid = coord.seqid.encode('ASCII')
                yield coord_id, sp, seqid, coord_len

    def _get_sp_seqids_from_h5(self, by=1000):
        """from flat h5 datasets to dict with {species: [seqid, seqid2, ...], ...}"""
        # todo, is this fully redundant with get_sp_seq_ranges?
        out = defaultdict(dict)
        for i in range(0, self.h5['data/y'].shape[0], by):
            species = self.h5['data/species'][i:(i + by)]
            seqids = self.h5['data/seqids'][i:(i + by)]
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
        in_h5 = self._get_sp_seqids_from_h5()
        for sp in in_h5:
            sp_str = sp.decode()
            for seqid in in_h5[sp]:
                coord_tuple = ckey[sp_str][seqid]
                genome_coords[sp_str].append(coord_tuple)
        return genome_coords

    def _numerify_coord(self, coord, coord_features, chunk_size, one_hot, write_by, modes, multiprocess):
        """filtering and stats"""
        coord_data_gen = CoordNumerifier.numerify(coord, coord_features, chunk_size, one_hot,
                                                  write_by=write_by, mode=modes, multiprocess=multiprocess)
        # the following will all be used to calculated a percentage, which is yielded but ignored until the end
        n_chunks = n_bases = n_ig_bases = n_masked_bases = 0

        for coord_data, h5_coord in coord_data_gen:
            # easy access to matrices
            y = [cd.matrix for cd in coord_data if cd.key == 'y'][0]
            sample_weights = [cd.matrix for cd in coord_data if cd.key == 'sample_weights'][0]

            # count things, only works properly for one hot encodings
            n_chunks += y.shape[0]
            n_ig_bases += np.count_nonzero(y[:, :, 0] == 1)
            padded_bases = np.count_nonzero(np.all(y == 0, axis=2))
            n_bases += np.prod(y.shape[:2]) - padded_bases
            n_masked_bases += np.count_nonzero(sample_weights == 0) - padded_bases

            masked_bases_perc = n_masked_bases / n_bases * 100
            ig_bases_perc = n_ig_bases / n_bases * 100

            yield coord_data, coord, masked_bases_perc, ig_bases_perc, h5_coord

    def export(self, chunk_size, genomes, one_hot=True, all_transcripts=False, write_by=10_000_000_000,
               modes=('X', 'y', 'anno_meta', 'transitions'), compression='gzip', multiprocess=True):
        genome_coord_features = {genome_name: self.exporters[genome_name].genome_query(all_transcripts=all_transcripts)
                                 for genome_name in genomes}
        # make version without features for shorter downstream code
        genome_coords = {genome_name: list(values.keys()) for genome_name, values in genome_coord_features.items()}

        n_coords = sum([len(coords) for genome_name, coords in genome_coords.items()])
        print('\n{} coordinates chosen to numerify'.format(n_coords))
        if self.match_existing:
            # resort coordinates to match existing as well (bc length sort doesn't work on ties)
            genome_coords = self._resort_genome_coords_from_existing(genome_coords)
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
                n_chunks = HelixerExportControllerBase.calc_n_chunks(coord_len, chunk_size)
                coord = self.exporters[genome_name].get_coord_by_id(coord_id)
                coord_features = genome_coord_features[genome_name][(coord_id, coord_len)]
                numerify_outputs = self._numerify_coord(coord, coord_features, chunk_size, one_hot, write_by=write_by,
                                                        modes=modes, multiprocess=multiprocess)

                for i, (flat_data, coord, masked_bases_perc, ig_bases_perc, h5_coord) in enumerate(numerify_outputs):
                    self._save_data(flat_data, h5_coords=h5_coord, n_chunks=n_chunks, first_round_for_coordinate=(i == 0),
                                    compression=compression, h5_group=self.h5_group)
                    n_writing_chunks += 1

                print(f'{n_coords_done}/{n_coords} Numerified {coord} of {genome_name} '
                      f"with {len(coord.features)} features in {flat_data[0].matrix.shape[0]} chunks, "
                      f'masked rate: {masked_bases_perc:.2f}%, ig rate: {ig_bases_perc:.2f}%, '
                      f'({time.time() - start_time:.2f} secs)', end='\n\n')
                n_coords_done += 1
            prev_genome_name = genome_name
        self._add_data_attrs(genomes)
        self.h5.close()
        print('Export from geenuff db to h5 file(s) with numeric matrices finished successfully.')
        return n_writing_chunks  # for testing only atm
