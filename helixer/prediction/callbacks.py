import os
import click
import random
import time
import numpy as np
import torch
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric import Fabric

from helixer.core.helpers import rank_zero_click_secho
from helixer.prediction.Callback import Callback


class ModelCheckpoint(Callback):
    def __init__(self, save_model_path, batch_size, save_every_check=False):
        self.local_rank = None
        self.save_model_path = save_model_path
        self.save_every_check = save_every_check
        self.train_batch_count = 0
        self.batch_size = batch_size

    def on_fit_start(self, fabric: Fabric):
        self.local_rank = fabric.local_rank

    def on_train_batch_end(self, runner):
        if runner.validation_interval is not None and not (self.train_batch_count + 1) % runner.validation_interval:
            print(f'\nvalidation and checkpoint at batch {self.train_batch_count}')
            self.check_in()

    def on_validation_epoch_end(self, runner):
        if self.save_every_check:
            self.save_checkpoint()
        # check if the model is better than the previous one
        if runner.current_genic_f1 > runner.best_genic_f1:
            self.save_checkpoint() # todo: add diff. name than each check epochs

    def save_checkpoint(self):
        pass

    def check_in(self, batch=None):
        # # todo: run own special val loop/metric check with the confusion matrix, best to runner and here just metric report
        # _, _, val_genic_f1 = HelixerModel.run_metrics(self.val_generator, self.model, calc_H=self.calc_H)
        # if self.report_to_nni:
        #     nni.report_intermediate_result(val_genic_f1)
        # if val_genic_f1 > self.best_val_genic_f1:
        #     self.best_val_genic_f1 = val_genic_f1
        #     self.freeze_layers(self.model)
        #     self.model.save(self.save_model_path, save_format='h5')
        #     print('saved new best model with genic f1 of {} at {}'.format(self.best_val_genic_f1,
        #                                                                   self.save_model_path))
        #     self.checks_without_improvement = 0
        # else:
        #     self.checks_without_improvement += 1
        #     if self.checks_without_improvement >= self.patience:
        #         self.model.stop_training = True
        # if batch is None:
        #     b_str = 'epoch_end'
        # else:
        #     b_str = f'b{batch:06}'
        # if self.save_every_check:
        #     path = os.path.join(self.save_dir, f'model_e{self.epoch}_{b_str}.h5')
        #     self.model.save(path, save_format='h5')
        #     print(f'saved model at {path}')
        pass


class TimeHelixerCallback:
    def __init__(self):
        super().__init__()
        self.start = 0
        self.last_epoch = 0
        self.local_rank = None
        self.divide = 0

    def on_fit_start(self, fabric: Fabric):
        self.local_rank = fabric.local_rank

    def on_train_start(self):
        self.start = time.time()

    def on_train_epoch_end(self):
        pass

    def on_train_end(self):
        pass

    def on_test_start(self):
        self.start = time.time()

    def on_test_end(self):
        rank_zero_click_secho(self.local_rank,
                              f"Testing took {round(((time.time() - self.start) / 60), ndigits=2)} min.")

    def on_predict_start(self):
        self.start = time.time()

    def on_predict_end(self):
        rank_zero_click_secho(self.local_rank,
                              f"Prediction took {round(((time.time() - self.start) / 60), ndigits=2)} min.")


class SeedCallback:
    def __init__(self, seed: int, resume: bool, model_path, device, num_workers):
        self.resume = resume
        self.include_cuda = True if device == 'gpu' else False
        self.workers = True if num_workers > 0 else False

        if not self.resume:
            self.seed = self.set_seed(seed, self.workers)
            self.state = self.collect_seed_state(self.include_cuda)
        else:
            # load model checkpoint callback on CPU
            # todo: can this be like this or fabric load?
            seed_callback_dict = torch.load(model_path,
                                            map_location=lambda storage, loc: storage)['callbacks']['SeedCallback']

            # retrieve seed and state dict from model checkpoint
            self.seed = seed_callback_dict['seed']
            self.state = seed_callback_dict['rng_states']
            click.secho(f'The seed provided by the model is: {self.seed}.')
            if 'torch.cuda' in self.state.keys() and not self.include_cuda:
                click.secho('You are resuming training of a GPU trained model on the CPU. '
                            'This is unintended. The training might not be reproducible.')
            if 'torch.cuda' not in self.state.keys() and self.include_cuda:
                click.secho('You are resuming training of a CPU trained model on the GPU. '
                            'This is unintended. The training might not be reproducible.')

            # set os variables (seed_everything() does this automatically)
            # distributed sampler (used by ddp to train on multiple devices) uses PL_GLOBAL_SEED
            # to recover the seed to shuffle and subsample the training dataset
            os.environ['PL_GLOBAL_SEED'] = str(self.seed)
            os.environ['PL_SEED_WORKERS'] = f'{int(self.workers)}'

    def on_train_start(self):
        if self.resume:
            self.set_seed_state(self.state)

    def on_train_epoch_end(self):
        self.state.update(self.collect_seed_state(self.include_cuda))

    def load_state_dict(self, state_dict):
        self.seed = state_dict['seed']
        self.state.update(state_dict['rng_states'])

    def state_dict(self):
        return {'seed': self.seed, 'rng_states': self.state.copy()}

    @staticmethod
    def set_seed(seed, workers):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min

        if seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            click.secho(f"A seed wasn't provided. Choosing random seed: {seed}")
        elif not (min_seed_value <= seed <= max_seed_value):
            seed = random.randint(min_seed_value, max_seed_value)
            click.secho(f'Invalid seed given. Choosing random seed: {seed}')
        else:
            click.secho(f'Chosen seed: {seed}')
        seed_everything(seed=seed, workers=workers)  # seed for reproducibility
        return seed

    @staticmethod
    def collect_seed_state(include_cuda):
        """ Function adapted from lightning.fabric.utilities.seed._collect_rng_states to collect
        the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python (random)."""
        states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if include_cuda:
            states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return states

    @staticmethod
    def set_seed_state(rng_state_dict):
        """Function adapted from lightning.fabric.utilities.seed._set_rng_states to set the global random
        state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python (random) in the current process."""
        torch.set_rng_state(rng_state_dict["torch"])
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
        np.random.set_state(rng_state_dict["numpy"])
        version, state, gauss = rng_state_dict["random"]
        random.setstate((version, tuple(state), gauss))


class PredictCallback:
    def __init__(self, input_file, output_file, batch_size):
        self.input_file = input_file
        self.output_file = output_file
        self.start = 0
        self.stop = None
        self.batch_size = batch_size

    def on_predict_start(self):
        pass

    def on_predict_batch_end(self, outputs, batch):
        pass
    # todo: this should pass stuff to rust for quick writing

# todo: add early stopping for training
