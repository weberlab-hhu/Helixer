from abc import ABC, abstractmethod
import os
import sys
import inspect
import click

import helixer.core.helpers

try:
    import nni
except ImportError:
    pass
import time
import glob
import zarr
import numcodecs
import datetime
from importlib.metadata import version
import subprocess
import numpy as np
from sklearn.utils import shuffle
from terminaltables import AsciiTable

import torch
from torch import nn
from torch.utils.data import Dataset
from lightning.fabric import Fabric

from helixer.prediction.Metrics import Metrics
from helixer.core import overlap
from helixer.core.strs import *
#from helixer.cli.cli_formatter import ClsMethodClickCommand

#class ConfusionMatrixTrain(Callback):
class ConfusionMatrixTrain():
    def __init__(self, save_model_path, train_generator, val_generator, large_eval_folder, patience, calc_H=False,
                 report_to_nni=False, check_every_nth_batch=1_000_000, save_every_check=False):
        self.save_model_path = save_model_path
        self.save_dir = os.path.dirname(save_model_path)
        self.save_every_check = save_every_check
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.large_eval_folder = large_eval_folder
        self.patience = patience
        self.calc_H = calc_H
        self.report_to_nni = report_to_nni
        self.best_val_genic_f1 = 0.0
        self.checks_without_improvement = 0
        self.check_every_nth_batch = check_every_nth_batch  # high default for ~ 1 / epoch
        self.epoch = 0
        print(self.save_model_path, 'SAVE MODEL PATH')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f'training took {(time.time() - self.epoch_start) / 60:.2f}m')
        self.check_in()
        self.epoch += 1

    def on_train_batch_end(self, batch, logs=None):
        if not (batch + 1) % self.check_every_nth_batch:
            print(f'\nvalidation and checkpoint at batch {batch}')
            self.check_in(batch)

    # todo: find other way to freeze layers
    #def freeze_layers(self, model):
    #    # thank you https://github.com/keras-team/keras/issues/13279#issuecomment-527705263
    #    for i in model.layers:
    #        i.trainable = False
    #        if isinstance(i, Model):
    #           self.freeze_layers(i)
    #    return model

    def check_in(self, batch=None):
        _, _, val_genic_f1 = HelixerModel.run_metrics(self.val_generator, self.model, calc_H=self.calc_H)
        if self.report_to_nni:
            nni.report_intermediate_result(val_genic_f1)
        if val_genic_f1 > self.best_val_genic_f1:
            self.best_val_genic_f1 = val_genic_f1
            self.freeze_layers(self.model)
            self.model.save(self.save_model_path, save_format='h5')
            print('saved new best model with genic f1 of {} at {}'.format(self.best_val_genic_f1,
                                                                          self.save_model_path))
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1
            if self.checks_without_improvement >= self.patience:
                self.model.stop_training = True
        if batch is None:
            b_str = 'epoch_end'
        else:
            b_str = f'b{batch:06}'
        if self.save_every_check:
            path = os.path.join(self.save_dir, f'model_e{self.epoch}_{b_str}.h5')
            self.model.save(path, save_format='h5')
            print(f'saved model at {path}')

    def on_train_end(self, logs=None):
        # TODO rewrite entire function later
        if os.path.isdir(self.large_eval_folder):
            # load best model
            # TODO custom save and load to retain some info in the dict (not just model state dict)
            best_model = load_model(self.save_model_path)
            # double check that we loaded the correct model, can be remove if confirmed this works
            print('\nValidation set again:')
            _, _, val_genic_f1 = HelixerModel.run_metrics(self.val_generator, best_model, print_to_stdout=True,
                                                          calc_H=self.calc_H)
            assert val_genic_f1 == self.best_val_genic_f1

            training_species = self.train_generator.h5_file.attrs['genomes']
            median_f1 = HelixerModel.run_large_eval(self.large_eval_folder, best_model, self.val_generator, training_species)

            if self.report_to_nni:
                nni.report_final_result(median_f1)

        elif self.report_to_nni:
            nni.report_final_result(self.best_val_genic_f1)


class HelixerSequence(Dataset):
    def __init__(self, model, zarr_files, mode, batch_size, rank, world_size):
        # todo: handle batch_size and shuffle in the dataloader
        assert mode in [TEST, TRAIN, VAL]
        self.model = model
        self.zarr_files = zarr_files  # generators, opened in read mode by HelixerModel class
        self.mode = mode
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self._cp_into_namespace(['float_precision', 'class_weights', 'transition_weights', 'input_coverage',
                                 'coverage_count', 'coverage_norm', 'overlap', 'overlap_offset', 'core_length',
                                 'predict_phase', 'only_predictions', 'num_devices', 'debug'])

        if self.mode == TEST:
            assert len(self.zarr_files) == 1, "predictions and eval should be applied to individual files only"

        # set chunk size and overlap parameters
        # this is first done here, because its pulled dynamically from zarr files
        chunk_sizes = [zf[DATA_X].shape[1] for zf in self.zarr_files]
        for cs in chunk_sizes[1:]:
            assert cs == chunk_sizes[0], f'Not all subsequence lengths match in zarr files: {chunk_sizes}'
        self.chunk_size = chunk_sizes[0]
        # once we have the chunk_size, we can find defaults for overlapping
        if self.overlap_offset is None:
            self.overlap_offset = self.chunk_size // 2
        if self.core_length is None:
            self.core_length = int(self.chunk_size * 3 / 4)

        self.data_list_names = [DATA_X]
        if not self.only_predictions:
            self.data_list_names += [DATA_Y, DATA_SAMPLE_WEIGHTS]
            if self.load_predictions:
                self.data_list_names.append('data/predictions')  # deprecated
            if self.predict_phase:
                self.data_list_names.append(DATA_PHASES)
            if self.mode == 'train':
                if self.transition_weights is not None:
                    self.data_list_names.append(DATA_TRANSITIONS)
                if self.coverage_weights:
                    self.data_list_names.append('scores/by_bp')

        if self.overlap:
            assert self.mode == TEST, "overlapping currently only works for test (predictions & eval)"
            # todo: available for training?, if yes, this needs to be integrated into the sharding logic
            # can take [0] below bc we've asserted that test means len(self.zarr_files) == 1 above
            contiguous_ranges = helixer.core.helpers.get_contiguous_ranges(self.zarr_files[0])
            self.ol_helper = overlap.OverlapSeqHelper(contiguous_ranges=contiguous_ranges,
                                                      chunk_size=self.chunk_size,
                                                      max_batch_size=self.batch_size,
                                                      overlap_offset=self.overlap_offset,
                                                      core_length=self.core_length)

        if self.input_coverage:
            self.data_list_names += ['evaluation/rnaseq_coverage', 'evaluation/rnaseq_spliced_coverage']

        self.data_lists = [[] for _ in range(len(self.data_list_names))]
        self.data_dtypes = [self.zarr_files[0][name].dtype for name in self.data_list_names]
        # TODO: maybe convert X.dtype=float16 (less RAM usage while compressed)
        self.compressor = numcodecs.blosc.Blosc(cname='blosclz', clevel=4, shuffle=2)  # use BITSHUFFLE

        if self.num_devices > 1 and self.mode != TEST: # todo: != PREDICT
            # shard dataset across devices
            # get all samples/chunks per zarr file to distribute them evenly to the different processes
            self.sample_coordinates = self._load_zarr_indices()
            self.shard_lengths = self._create_shards(self.sample_coordinates.shape[0], self.world_size)
            self.shard_start_idx = sum(self.shard_lengths[:self.rank])
            self.shard_end_idx = sum(self.shard_lengths[:(self.rank+1)])
            self.shard_coordinates = self.sample_coordinates[self.shard_start_idx:self.shard_end_idx]
            self._load_shard()
        else:
            print(f'\nstarting to load {self.mode} data into memory..')
            # todo: transfer chunk down below into its own logic?
            for zarr_file in self.zarr_files:
                self._load_one_zarr(zarr_file)

        for name, data_list in zip(self.data_list_names, self.data_lists):
            comp_data_size = sum([sys.getsizeof(e) for e in data_list])
            print(f'Compressed data size of {name} is at least {comp_data_size / 2 ** 30:.4f} GB\n')

        self.n_seqs = len(self.data_lists[0])
        print(f'setting self.n_seqs to {self.n_seqs}, bc that is len of {self.data_list_names[0]}')

        if self.mode == TEST:
            if self.class_weights is not None:
                print(f'ignoring the class_weights of {self.class_weights} in mode "test"')
                self.class_weights = None
            if self.transition_weights is not None:
                print(f'ignoring the transition_weights of {self.transition_weights} in mode "test"')
                self.transition_weights = None

    def _load_zarr_indices(self):
        if self.mode == TRAIN or self.mode == VAL:
            mask = [np.logical_and(zf[DATA_IS_ANNOTATED], zf[DATA_ERR_SAMPLES]) for zf in self.zarr_files]
            idxs = [np.where(m == True)[0] for m in mask]
            # entire dataset, todo: print only on rank zero OR print somewhere else?
            print(f'\nmasking {sum([np.sum(m) for m in mask])} completely un-annotated or completely erroneous sequences')
        else:
            idxs = [np.arange(zf[DATA_X].shape[0]) for zf in self.zarr_files]
        file_idxs = [np.full((idxs[i].shape[0],), fill_value=i) for i in range(len(idxs))]
        # concatenate indices and zarr file indices, and then stack them up
        # result: [[0,0], [0,1], [0,2], ..., [3,2], ...]
        return np.stack([np.hstack(file_idxs), np.hstack(idxs)], axis=1)

    @staticmethod
    def _create_shards(total_dataset_length, world_size):
        count, remainder = divmod(total_dataset_length, world_size)
        return [count + 1 if i < remainder else count for i in range(world_size)]

    def _load_shard(self):
        file_idxs = np.unique(self.shard_coordinates[:,0])
        for i in file_idxs:
            zarr_file = self.zarr_files[i]
            indices = self.shard_coordinates[:,1][self.shard_coordinates[:,0]==i]
            n_seqs = len(indices)
            max_at_once = min(2000, n_seqs)
            for name, data_list in zip(self.data_list_names, self.data_lists):
                for offset in range(0, n_seqs, max_at_once):
                    data_slice = zarr_file[name][indices[offset:offset + max_at_once]]
                    data_list.extend([self.compressor.encode(e) for e in data_slice])

    def _load_one_zarr(self, zarr_file):
        print(f'For zarr file starting with species = {zarr_file[DATA_SPECIES][0]}:')
        x_dset = zarr_file[DATA_X]
        print(f'x shape: {x_dset.shape}')
        if not self.only_predictions:
            y_dset = zarr_file[DATA_Y]
            print(f'y shape: {y_dset.shape}')
        # todo: exclude debug and make mini_zarr the debug stand-ins, debug script/routine instead?
        if self.debug:
            # so that total sequences between all files add to ~1000
            n_seqs = max(1000 // len(self.zarr_files), 1)
        else:
            n_seqs = x_dset.shape[0]

        if self.mode == TRAIN or self.mode == VAL:
            mask = np.logical_and(zarr_file[DATA_IS_ANNOTATED],
                                  zarr_file[DATA_ERR_SAMPLES])
            n_masked = x_dset.shape[0] - np.sum(mask)
            print(f'\nmasking {n_masked} completely un-annotated or completely erroneous sequences')

        else:
            mask = np.ones(zarr_file[DATA_X].shape[0], dtype=bool)
            n_masked = 0

        # load at most 2000 uncompressed samples at a time in memory
        # todo: make this dependent on the sequence length as well
        max_at_once = min(2000, n_seqs)
        for name, data_list in zip(self.data_list_names, self.data_lists):
            start_time_dset = time.time()
            for offset in range(0, n_seqs, max_at_once):
                step_mask = mask[offset:offset + max_at_once]
                if name == 'data/predictions':  # deprecated
                    data_slice = zarr_file[name][0, offset:offset + max_at_once][step_mask]  # only use one prediction for now
                else:
                    data_slice = zarr_file[name][offset:offset + max_at_once][step_mask]
                data_list.extend([self.compressor.encode(e) for e in data_slice])
            print(f'Data loading of {n_seqs - n_masked} (total so far {len(data_list)}) samples of {name} '
                  f'into memory took {time.time() - start_time_dset:.2f} secs')

    #@staticmethod
    #def _zero_out_utrs(y):
        # merge UTR and IG labels and zero out the UTR column
        # still keep 4 columns for simplicity of downstream code and (maybe) more transfer learning potential
        #y[..., 0] = np.logical_or(y[..., 0], y[..., 1])
        #y[..., 1] = 0

    # todo: shuffler needs then to be independent of fabric's seed_everything
    def preshuffle_data(self):
        start_time = time.time()
        # todo: don't use data_lists for multi GPU training, but zarr_indices
        #  don't preshuffle when training on 1 GPU at all
        self.data_lists = shuffle(*self.data_lists)  # todo: add random_state=seed -> deterministic
        print(f'Reshuffled {self.mode} data in {time.time() - start_time:.2f} secs')

    def _cp_into_namespace(self, names):
        """Moves class properties from self.model into this class for brevity"""
        for name in names:
            self.__dict__[name] = self.model.__dict__[name]

    def _get_batch_data(self, batch_idx):
        batch = []
        # batch must have one thing for everything unpacked by __getitem__ (and in order)
        for name in [DATA_X, DATA_Y, DATA_SAMPLE_WEIGHTS, DATA_TRANSITIONS, DATA_PHASES,
                     'data/predictions', 'scores/by_bp']:
            if name not in self.data_list_names:
                batch.append(None)
            else:
                decoded_list = self.get_batch_of_one_dataset(name, batch_idx)

                # append coverage to X directly, might be clearer elsewhere once working, but this needs little code...
                # TODO delete/comment out and rework (be strict, not just accept rnaseq as prefix)
                if name == DATA_X and self.input_coverage:
                    decode_coverage = self.get_batch_of_one_dataset('evaluation/rnaseq_coverage', batch_idx)
                    decode_coverage = [self._cov_norm(x.reshape(-1, self.coverage_count)).astype(np.float16) for x in decode_coverage]
                    decode_spliced = self.get_batch_of_one_dataset('evaluation/rnaseq_spliced_coverage', batch_idx)
                    decode_spliced = [self._cov_norm(x.reshape(-1, self.coverage_count)).astype(np.float16) for x in decode_spliced]
                    decoded_list = [np.concatenate((x, y, z), axis=1) for x, y, z in
                                    zip(decoded_list, decode_coverage, decode_spliced)]

                decoded = np.stack(decoded_list, axis=0)
                if self.overlap and name == DATA_X:
                    decoded = self.ol_helper.make_input(batch_idx, decoded)

                batch.append(decoded)

        return tuple(batch)

    def get_batch_of_one_dataset(self, name, batch_idx):
        """returns single batch (the Nth where N=batch_idx) from dataset '{name}'"""
        # setup indices based on overlapping or not
        if self.overlap:
            zarr_indices = self.ol_helper.zarr_indices_of_batch(batch_idx)
        else:
            end = min(self.n_seqs, (batch_idx + 1) * self.batch_size)
            zarr_indices = np.arange(batch_idx * self.batch_size, end)

        return self._decode_one(name, zarr_indices)

    def _decode_one(self, name, zarr_indices):
        """decode batch delineated by zarr_indices from compressed data originally from dataset {name}"""
        i = self.data_list_names.index(name)
        dtype = self.data_dtypes[i]
        data_list = self.data_lists[i]
        decoded_list = [np.frombuffer(self.compressor.decode(data_list[idx]), dtype=dtype)
                        for idx in zarr_indices]
        if len(decoded_list[0]) > self.chunk_size:
            decoded_list = [e.reshape(self.chunk_size, -1) for e in decoded_list]
        return decoded_list

    def _cov_norm(self, x):
        method = self.coverage_norm
        if method is None:
            return x
        elif method == 'log':
            return np.log(x + 1.1)
        elif method == 'linear':
            return x / 100
        else:
            raise ValueError(f'unrecognized method: {method} for normalizing coverage data')

    def _update_sw_with_transition_weights(self):
        pass

    def _update_sw_with_coverage_weights(self):
        pass

    def _mk_timestep_pools(self, matrix):
        """reshape matrix to have multiple bp per timestep (in last dim)"""
        # assumes input shape
        # [0] = batch_size            --> don't touch
        # [1] = data's chunk_size     --> divide by pool size
        # [2:] = collapsable          --> -1, remaining, AKA np.prod(shape[2:]) * pool_size
        pool_size = self.model.pool_size
        if matrix is None:
            return None
        shape = list(matrix.shape)
        shape[1] = shape[1] // pool_size
        shape[-1] = -1
        matrix = matrix.reshape((
            shape
        ))
        return matrix

    def _mk_timestep_pools_class_last(self, matrix):
        """reshape matrix to have multiple bp per timestep, w/ classes as last dim for softmax"""
        if matrix is None:
            return None
        pool_size = self.model.pool_size
        assert len(matrix.shape) == 3
        # assumes input shape
        # [0] = batch_size            --> don't touch
        # [1] = data's chunk_size     --> divide by pool size
        # [2] = labels                --> pooling inserted before, retained as last dimension
        matrix = matrix.reshape((
            matrix.shape[0],
            matrix.shape[1] // pool_size,
            pool_size,  # make labels 2d, so we can use the standard softmax / loss functions
            matrix.shape[-1],
        ))
        return matrix

    def _aggregate_timestep_pools(self, matrix, aggr_function=np.mean):
        pass

    def compress_tw(self, transitions):
        return self._squish_tw_to_sw(transitions, self.transition_weights, self.stretch_transition_weights)

    @staticmethod
    def _squish_tw_to_sw(transitions, transition_weights, stretch):
        # basically completes a max pool of transitions, from their pre-timestep pooled form
        max_pooled_trns = np.max(transitions, axis=2).astype(np.int8)

        # multiply by weights, and sum classes
        pooled_weighted_trns = np.multiply(max_pooled_trns, transition_weights)
        summed_trns = np.sum(pooled_weighted_trns, axis=2)

        # replace 0's (where weighting shouldn't change) with 1's as this will be multiplied later
        summed_trns[summed_trns == 0] = 1

        return summed_trns

    @staticmethod
    def to_torch_tensor(data):
        # todo: can this be accelerated by creating the final tensor on the GPU directly with init_tensor()
        #  or will this mess up prefetching and pin_memory?
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
           if all(isinstance(d, np.ndarray) for d in data):
               return [torch.from_numpy(d).float() for d in data]
           else:
               raise Exception(f'expected list of numpy.ndarrays, got {type(data[0])}')
        else:
            raise Exception(f'expected numpy.ndarray, got {type(data)}')

    def __len__(self):
        """how many total samples"""
        if self.debug:
            # if self.debug and self.mode == 'train':
            return 3
        elif self.overlap:
            return self.ol_helper.adjusted_epoch_length()
        else:
            return self.n_seqs

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _generic_get_item(self, idx):
        """covers the data preprocessing (reshape, trim, weighting, etc.) common to all models"""
        X, y, sw, transitions, phases, _, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size

        if pool_size > 1:
            if X.shape[1] % pool_size != 0:
                # clip to maximum size possible with the pooling length
                overhang = X.shape[1] % pool_size
                X = X[:, :-overhang]
                if not self.only_predictions:
                    y = y[:, :-overhang]
                    sw = sw[:, :-overhang]
                    if self.predict_phase:
                        phases = phases[:, :-overhang]
                    if self.mode == TRAIN and self.transition_weights is not None:
                        transitions = transitions[:, :-overhang]

            if not self.only_predictions:
                y = self._mk_timestep_pools_class_last(y)
                sw = sw.reshape((sw.shape[0], -1, pool_size))
                sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.int8)

            if self.mode == TRAIN:
                if self.class_weights is not None:
                    # class weights are additive for the individual timestep predictions
                    # giving even more weight to transition points
                    # class weights without pooling not supported yet
                    # cw = np.array([1.0, 1.2, 1.0, 0.8], dtype=np.float32)
                    cls_arrays = [np.any((y[:, :, :, col] == 1), axis=2) for col in range(4)]
                    cls_arrays = np.stack(cls_arrays, axis=2).astype(np.int8)
                    # add class weights to applicable timesteps
                    cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:2] + (1,)))
                    cw = np.sum(cw_arrays, axis=2)
                    sw = np.multiply(cw, sw)

                # todo, while now compressed, the following is still 1:1 with LSTM model... --> HelixerModel
                if self.transition_weights is not None:
                    transitions = self._mk_timestep_pools_class_last(transitions)
                    # more reshaping and summing  up transition weights for multiplying with sample weights
                    sw_t = self.compress_tw(transitions)
                    sw = np.multiply(sw_t, sw)

                if self.coverage_weights:
                    coverage_scores = coverage_scores.reshape((coverage_scores.shape[0], -1, pool_size))
                    # maybe offset coverage scores [0,1] by small number (bc RNAseq has issues too), default 0.0
                    if self.coverage_offset > 0.:
                        coverage_scores = np.add(coverage_scores, self.coverage_offset)
                    coverage_scores = np.mean(coverage_scores, axis=2)
                    sw = np.multiply(coverage_scores, sw)

            if self.predict_phase and not self.only_predictions:
                y_phase = self._mk_timestep_pools_class_last(phases)
                y = [y, y_phase]

            return [self.to_torch_tensor(d) for d in (X, y, sw, transitions, phases, _, coverage_scores)]


class HelixerModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_hparams(self):
        init_parameters = inspect.signature(self.__init__).parameters
        hyperparameters = {arg: getattr(self, arg) for arg, value in init_parameters.items()}
        return hyperparameters


class HelixerBaseModelRunner:
    """Not an actual Lightning trainer class"""
    def __init__(self):
        pass

    @staticmethod
    def setup_fabric(mode, device, num_devices, callbacks, precision):
        if mode == "train":
            strategy = "auto" if num_devices == 1 else "ddp"
        else:
            strategy = "auto"
        # callbacks need to go in here as well BEFORE context
        # launch fabric after general setup but before loading the dataset or model
        # when to seed everything/recover the random states? initial seed_everything BEFORE model setup
        # in setup dataloader: use distributed sampler false
        precisions = {'float32': '32-true', 'float16': '16-true'}
        fabric = Fabric(accelerator=device, devices=num_devices, strategy=strategy,
                        precision=precisions[precision], callbacks=callbacks)
        fabric.launch()
        return fabric
        # init model directly on GPU with fabric.init_module()

    def generate_callbacks(self, train_generator):
        pass
        # TODO torch mvp/runner specific, create callback script
    #    callbacks = [ConfusionMatrixTrain(self.save_model_path, train_generator, self.gen_validation_data(),
    #                                      self.large_eval_folder, self.patience, calc_H=self.calculate_uncertainty,
    #                                      report_to_nni=self.nni, check_every_nth_batch=self.check_every_nth_batch,
    #                                      save_every_check=self.save_every_check),
    #                 PreshuffleCallback(train_generator)]
    #    return callbacks

    def init_model(self, model_class, model_checkpoint, **model_kwargs):
        if model_checkpoint:
            # load from checkpoint
            pass
        else:
            model = model_class(**model_kwargs)
        return model

    # todo: to metrics callback
    @staticmethod
    def run_metrics(generator, model, print_to_stdout=True, calc_H=False):
        start = time.time()
        metrics_calculator = Metrics(generator, print_to_stdout=print_to_stdout,
                                     skip_uncertainty=not calc_H)
        metrics = metrics_calculator.calculate_metrics(model)
        genic_metrics = metrics['genic_base_wise']['genic']
        if np.isnan(genic_metrics['f1']):
            genic_metrics['f1'] = 0.0
        print('\nmetrics calculation took: {:.2f} minutes\n'.format(int(time.time() - start) / 60))
        return genic_metrics['precision'], genic_metrics['recall'], genic_metrics['f1']

    @abstractmethod
    def sequence_cls(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    @staticmethod
    def sum_shapes(datasets):
        shapes = [ds.shape for ds in datasets]
        return [sum(x[0] for x in shapes)] + list(shapes[0][1:])

    def open_data_files(self):
        def get_n_correct_seqs(zarr_files):
            sum_n_correct = 0
            for zarr_file in zarr_files:
                if DATA_ERR_SAMPLES in zarr_file.keys():
                    err_samples = np.array(zarr_file[DATA_ERR_SAMPLES])
                    n_correct = np.count_nonzero(err_samples == False)
                    if n_correct == 0:
                        print('WARNING: no fully correct sample found')
                else:
                    print('No err_samples dataset found, correct samples will be set to 0')
                    n_correct = 0
                sum_n_correct += n_correct
            return sum_n_correct

        def get_n_intergenic_seqs(zarr_files):
            sum_n_fully_ig = 0
            for zarr_file in zarr_files:
                if DATA_FULLY_INTERGENIC_SAMPLES in zarr_file.keys():
                    ic_samples = np.array(zarr_file[DATA_FULLY_INTERGENIC_SAMPLES])
                    n_fully_ig = np.count_nonzero(ic_samples == True)
                    if n_fully_ig == 0:
                        print('WARNING: no fully intergenic samples found')
                else:
                    print('No fully_intergenic_samples dataset found, fully intergenic samples will be set to 0')
                    n_fully_ig = 0
                sum_n_fully_ig += n_fully_ig
            return sum_n_fully_ig

        if self.run_purpose == TRAIN:
            self.zarr_trains = [zarr.open(f, mode='r') for f in glob.glob(os.path.join(self.data_dir, 'training_data*.zarr'))]
            self.zarr_vals = [zarr.open(f, mode='r') for f in glob.glob(os.path.join(self.data_dir, 'validation_data*.zarr'))]
            try:
                self.shape_train = self.sum_shapes([zf[DATA_X] for zf in self.zarr_trains])
            except IndexError as e:
                print('debugging info: self.zarr_trains = {}, self.data_dir = {}'.format(self.zarr_trains, self.data_dir),
                      file=sys.stderr)
                raise e
            try:
                self.shape_val = self.sum_shapes([zf[DATA_X] for zf in self.zarr_vals])
            except IndexError as e:
                print('debugging info: self.zarr_vals = {}, self.data_dir = {}'.format(self.zarr_vals, self.data_dir),
                      file=sys.stderr)
                raise e

            n_train_correct_seqs = get_n_correct_seqs(self.zarr_trains)
            n_val_correct_seqs = get_n_correct_seqs(self.zarr_vals)

            n_train_seqs = self.shape_train[0]
            n_val_seqs = self.shape_val[0]  # always validate on all

            n_intergenic_train_seqs = get_n_intergenic_seqs(self.zarr_trains)
            n_intergenic_val_seqs = get_n_intergenic_seqs(self.zarr_vals)
        else:
            self.zarr_tests = [zarr.open(self.test_data_path, mode='r')]  # list for consistency with train/val
            self.shape_test = self.zarr_tests[0][DATA_X].shape

            n_test_correct_seqs = get_n_correct_seqs(self.zarr_tests)
            n_test_seqs_with_intergenic = self.shape_test[0]
            n_intergenic_test_seqs = get_n_intergenic_seqs(self.zarr_tests)

        if self.verbose:
            print('\nData config: ')
            if self.run_purpose == TRAIN:
                print([dict(x.attrs) for x in self.zarr_trains])
                print('\nTraining {} shape: {}'.format(DATA_X, self.shape_train[:2]))
                print('Validation {} shape: {}'.format(DATA_X, self.shape_val[:2]))
                print('\nTotal est. training sequences: {}'.format(n_train_seqs))
                print('Total est. val sequences: {}'.format(n_val_seqs))
                print('\nEst. intergenic train/val seqs: {:.2f}% / {:.2f}%'.format(
                    n_intergenic_train_seqs / n_train_seqs * 100,
                    n_intergenic_val_seqs / n_val_seqs * 100))
                print('Fully correct train/val seqs: {:.2f}% / {:.2f}%\n'.format(
                    n_train_correct_seqs / self.shape_train[0] * 100,
                    n_val_correct_seqs / self.shape_val[0] * 100))
            else:
                print([dict(x.attrs) for x in self.zarr_tests])
                print('\nTest data shape: {}'.format(self.shape_test[:2]))
                print('\nIntergenic test seqs: {:.2f}%'.format(
                    n_intergenic_test_seqs / n_test_seqs_with_intergenic * 100))
                print('Fully correct test seqs: {:.2f}%\n'.format(
                    n_test_correct_seqs / self.shape_test[0] * 100))

    def _print_model_info(self):
        # todo: add model summary; use lightning as an example (leave out parameter size computation for now)
        pwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        try:
            cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            branch = subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip().decode()
            cmd = ['git', 'describe', '--always']  # show tag or hash if no tag available
            commit = subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip().decode()
            print(f'Current Helixer branch: {branch} ({commit})')
        except subprocess.CalledProcessError:
            print(f'Current Helixer version: {version("helixer")}')

        try:
            if os.path.isfile(self.load_model_path):
                cmd = ['md5sum', self.load_model_path]
                self.loaded_model_hash = subprocess.check_output(cmd).strip().decode()
                print(f'Md5sum of the loaded model: {self.loaded_model_hash}')
        except subprocess.CalledProcessError:
            print('An error occurred while running a subprocess, unable to record loaded_model_hash')
            self.loaded_model_hash = 'error'

        print()
        if self.verbose:
            print(self.model)
        else:
            print('Total params: {:,}'.format(self.count_params()))
        os.chdir(pwd)  # return to previous directory
    
    def eval_epoch(self, data):
        size = len(data.data)
        num_batches = len(data.loader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data.loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size * y.shape[1]
        print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # TODO hook to actual metrics, masking and all that jazz
        # todo: repackage into tnt units (class, let HelixerModel inherit from that class)

    def train_epoch(self, training_data):
        size = len(training_data.data)
        self.model.train()
        for batch, (X, y) in enumerate(training_data.loader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #model.fit(train_generator,
            #          epochs=self.epochs,
            #          workers=self.workers,
            #          callbacks=self.generate_callbacks(train_generator),
            #          verbose=True)

    def save_model(self):
        # todo: save all hyperparameters and callback states (seed, loop state, ranks, global step?)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, self.save_model_path)

    def load_model(self):
        checkpoint = torch.load(self.load_model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # todo: set self.optimizer in init
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def fit(self, training_data, eval_data):
        # todo, this needs to grow to be s.t. with early stopping, call backs, etc... (torch.tnt or better yet fabric)
        epochs = 3
        for epoch in range(epochs):
            self.epoch = epoch + 1
            print(f"Epoch {self.epoch}\n-------------------------------")
            self.train_epoch(training_data)
            self.eval_epoch(eval_data)
            # todo: save self.loss here

        # todo: if save every so and so interval, don't save here again if the interval is hit (callback candidate)
        # todo: still save the best model only or give more options? e.g. at least every epoch for training
        # todo (continued): in case the training stops for non-early stop reasons
        self.save_model()


    def run(self):
        def load_model_strategy():
            if not self.input_coverage:
                model = load_model(self.load_model_path)
            else:
                # for whatever reason, the fine tuning method is not saving the full model
                # in an entirely valid h5 file (depending on if you ask h5py or h5ls). puh.
                # thus loading the original model is both the easiest way to get architecture
                # setup and seems safer to make sure _all_ and not just _new_ weights are there
                oldmodel = load_model(self.pretrained_model_path)
                # repeat everything done setting up training to get exact architecture
                # freeze weights and replace everything from the dense layer
                dense_at = [l.name for l in oldmodel.layers].index('dense')
                for layer in oldmodel.layers:
                    layer.trainable = False

                model = self.insert_coverage_before_hat(oldmodel, dense_at)
                model.load_weights(self.load_model_path)
            return model

        self.set_resources()
        #self.open_data_files()
        #if self.input_coverage:
        #    # preview first h5 file to find number of bam files and set 'coverage_count'
        #    # which is used to calculate model input size +
        #    try:
        #        h5_files = self.h5_tests
        #    except AttributeError:
        #        h5_files = self.h5_trains
        #    self.coverage_count = h5_files[0]['evaluation/rnaseq_coverage'].shape[2]

        if self.run_purpose == TRAIN:
            if self.resume_training:
                self.load_model()
                pass  
            #    if not self.fine_tune:
            #        model = load_model(self.load_model_path)
            #    else:
            #        oldmodel = load_model(self.load_model_path)
            #        assert oldmodel.input.shape[2] == 4, \
            #            f"input shape of trained model != 4 ({oldmodel.input.shape[2]}); " \
            #            f"fine tuning is only supported on models trained without coverage"
            #        # freeze weights and replace everything from the dense layer
            #        dense_at = [l.name for l in oldmodel.layers].index('dense')
            #        for layer in oldmodel.layers:
            #            layer.trainable = False
            #        # the following assumes the base-model is trained without coverage
            #        if not self.input_coverage:
            #            inp = oldmodel.input
            #            output = self.model_hat((oldmodel.layers[dense_at - 1].output, None))
            #            model = Model(inp, output)
            #        else:
            #            model = self.insert_coverage_before_hat(oldmodel, dense_at)

            else:
                self.model = self.setup_model()
            self._print_model_info()

            #if self.optimizer.lower() == 'adam':
            #    self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clip_norm)
            #elif self.optimizer.lower() == 'adamw':
            #    self.optimizer = AdamW(learning_rate=self.learning_rate, clipnorm=self.clip_norm,
            #                           weight_decay=self.weight_decay)


            self.compile_model()
            self.fit(self.training_data, self.validation_data)
        else:
            self.model = self.setup_model() 
            self.load_model()
            self.compile_model()  # todo, this is likely optional once metrics are in and loss not reported
            self._print_model_info()

            if self.run_purpose == EVAL:
                # TODO, next line only, mvp
                #_, _, _ = HelixerModel.run_metrics(test_generator, model, calc_H=self.calculate_uncertainty)
                self.eval_epoch(self.test_data)
                #if self.large_eval_folder:
                #    assert self.data_dir != '', 'need training data of the model for training genome names'
                #    training_species = h5py.File(os.path.join(self.data_dir, 'training_data.h5'), 'r').attrs['genomes']
                #    _ = HelixerModel.run_large_eval(self.large_eval_folder, model, test_generator, training_species,
                #                                    print_to_stdout=True, calc_H=self.calculate_uncertainty)
            elif self.run_purpose == PREDICT:
                if os.path.isfile(self.prediction_output_path):
                    print(f'{self.prediction_output_path} already exists and will be overwritten.')
                self._make_predictions(self.model)  # callback candidate?
            else:
                assert ValueError, f"run_purpose should be in {TRAIN}, {EVAL}, {PREDICT}"


#    def insert_coverage_before_hat(self, oldmodel, dense_at):
#        """splits input in half, feeds CATG to the main model, and coverage in before tuning layers"""
#        # hacking RNAseq coverage in.
#        values_per_bp = 4 + self.coverage_count * 2
#
#        raw_input = Input(shape=(None, values_per_bp), dtype=self.float_precision,
#                          name='main_input')
#        # make a callable model that gives the intermediate output, with first 4 of input (CATG)
#        excerpt_model = Model(oldmodel.input, oldmodel.layers[dense_at - 2].output)
#        x = excerpt_model(raw_input[:, :, :4])
#        # add hat, including coverage back on
#        output = self.model_hat((x, raw_input[:, :, 4:]))
#
#        model = Model(raw_input, output)
#        return model

    # # check if model timestep width fits the subsequence length (has to be evenly divisible)
    # # todo: adapt to pytorch's/fabric's ckpt files
    # with h5py.File(args.model_filepath, 'r') as model:
    #     # todo, safer way to find this, i.e. add to ckpt model infos
    #     try:
    #         timestep_width = model['/model_weights/dense_1/dense_1/bias:0'].shape[0] // 8
    #     except KeyError:
    #         try:
    #             timestep_width = model['/model_weights/dense/dense/bias:0'].shape[0] // 8
    #         except KeyError:
    #             print("WARNING could not parse timestep width from model, assuming it is 9")
    #             timestep_width = 9
    #     msg = (f'subsequence length (currently {args.subsequence_length}) '
    #            f'has to be evenly divisible by {timestep_width}')
    #     assert args.subsequence_length % timestep_width == 0, msg


class HelixerTrainer(HelixerBaseModelRunner):
    def __init__(self, data_dir, save_model_path, epochs, batch_size, val_batch_size,
                 patience, check_every_nth_batch, optimizer, clip_norm, learning_rate, weight_decay,
                 class_weights, transition_weights, resume_training, load_model_path, save_every_check,
                 verbose, debug, fine_tune, pretrained_model_path,
                 input_coverage, coverage_norm, add_hidden_layer, model_class, **model_kwargs):
        super().__init__()
        self.fabric = self.setup_fabric()
        pass

    def generate_callbacks(self):
        pass

    def train_epoch(self, batch):
        pass

    def val_epoch(self, batch):
        pass

    def train(self, batch):
        pass


class HelixerTester(HelixerBaseModelRunner):
    def __init__(self, test_data_path, load_model_path, batch_size, overlap, overlap_offset,
                 overlap_core_length, model_class):
        super().__init__()
        self.fabric = self.setup_fabric()
        pass

    def generate_callbacks(self):
        pass

    def test_epoch(self, batch):
        pass

    def test(self, batch):
        pass


class HelixerPredictor(HelixerBaseModelRunner):
    def __init__(self, input_data_path, load_model_path, batch_size, prediction_output_path, overlap,
                 overlap_offset, overlap_core_length, model_class):
        super().__init__()
        self.fabric = self.setup_fabric()
        pass

    def generate_callbacks(self):
        pass

    def predict_epoch(self, batch):
        pass

    def predict(self, batch):
        pass


if __name__ == '__main__':
    print(click.secho("ERROR: 'HelixerModel.py' is not meant to be executed by the user. "
                      "Please use 'Helixer.py' or 'HybridModel.py'.", fg='red', bold=True))
