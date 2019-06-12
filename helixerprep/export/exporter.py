import os
import numpy as np
import sklearn
import configparser

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
            print('WARNING: target dir {} is not empty\n'.format(self.out_dir))
        self._mk_session()

    def _mk_session(self):
        self.engine = create_engine(full_db_path(self.db_path_in), echo=False)
        self.session = sessionmaker(bind=self.engine)()

    def export(self, chunk_size, shuffle_in_file, approx_file_size=100*2**20):
        """Fetches all Coordinates, calls on functions in numerify.py to split
        and encode them and then saves the sequences in possibly multiply files
        of about the size of approx_file_size.
        """
        def get_file_name(i):
            return os.path.join(self.out_dir, 'data' + str(i))

        def save_data(i, inputs, labels, label_masks):
            if shuffle_in_file:
                inputs, labels, label_masks = sklearn.utils.shuffle(inputs, labels, label_masks)
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
            np.savez(get_file_name(i), X=X, y=y, sample_weights=sample_weights)
            # print('saved {}.npz with {} sequences'.format(get_file_name(i), len(inputs)))

        n_file_chunks = 0
        n_chunks_total = 0  # total number of chunks
        current_size = 0  # if this is > approx_file_size make new file chunk
        inputs, labels, label_masks = [], [], []
        all_coords = self.session.query(Coordinate).all()
        print('{} coordinates chosen to numerify'.format(len(all_coords)))
        for i, coord in enumerate(all_coords):
            if coord.features:
                n_masked_bases = 0
                for is_plus_strand in [True, False]:
                    numerifier = CoordNumerifier(coord, is_plus_strand, chunk_size)
                    coord_data = numerifier.numerify()
                    # add data
                    inputs += coord_data['inputs']
                    labels += coord_data['labels']
                    label_masks += coord_data['label_masks']
                    # keep track of variables
                    current_size += coord.end
                    n_chunks_total += len(coord_data['inputs'])
                    n_masked_bases += sum([len(m) - np.count_nonzero(m)
                                           for m
                                           in coord_data['label_masks']])
                    # break data up in file chunks
                    if current_size * 12 > approx_file_size:
                        save_data(n_file_chunks, inputs, labels, label_masks)
                        inputs, labels, label_masks = [], [], []
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
            save_data(n_file_chunks, inputs, labels, label_masks)
        else:
            print('Skipping the left over data as it is empty')

        # write config file
        config = configparser.ConfigParser()
        config.add_section('data')
        config.set('data', 'chunk_size', str(chunk_size))
        config.set('data', 'n_chunks_total', str(n_chunks_total))
        config.set('data', 'shuffle_in_file', str(shuffle_in_file))

        config_file = open(os.path.join(self.out_dir, 'data_config.ini'), 'w')
        config.write(config_file)
        config_file.close()
