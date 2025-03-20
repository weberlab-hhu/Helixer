import os
import random
import time
import numpy as np
import torch
from lightning.fabric.utilities.seed import seed_everything

from helixer.core.helpers import rank_zero_click_secho


class ConfusionMatrixCallback:
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def on_train_epoch_end(self):
        pass

    def on_train_batch_end(self, batch_idx):
        pass

    def on_test_end(self):
        pass

    def save_metrics(self):
        pass


class TimeHelixerCallback:
    def __init__(self, local_rank):
        super().__init__()
        self.start = 0
        self.last_epoch = 0
        self.local_rank = local_rank
        self.divide = 0

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


# try shorter set_seed, leave random seed to lightning
class SeedCallback:
    def __init__(self, seed: int, resume: bool, model_path, include_cuda: bool, num_workers, local_rank):
        self.resume = resume
        self.include_cuda = include_cuda
        self.workers = True if num_workers > 0 else False
        self.local_rank = local_rank

        if not self.resume:
            # todo: seed everything here then?
            #  or create a before model init method
            self.seed = self.set_seed(seed, self.workers)
            self.state = self.collect_seed_state(self.include_cuda)
        else:
            # load model checkpoint callback on CPU
            seed_callback_dict = torch.load(model_path,
                                            map_location=lambda storage, loc: storage)['callbacks']['SeedCallback']

            # retrieve seed and state dict from model checkpoint
            self.seed = seed_callback_dict['seed']
            self.state = seed_callback_dict['rng_states']
            rank_zero_click_secho(self.local_rank,f'The seed provided by the model is: {self.seed}.')
            if 'torch.cuda' in self.state.keys() and not include_cuda:
                rank_zero_click_secho(self.local_rank,
                                      'You are resuming training of a GPU trained model on the CPU. '
                                      'This is unintended. The training might not be reproducible.')
            if 'torch.cuda' not in self.state.keys() and include_cuda:
                rank_zero_click_secho(self.local_rank,
                                      'You are resuming training of a CPU trained model on the GPU. '
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

    def set_seed(self, seed, workers):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min

        if seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            rank_zero_click_secho(self.local_rank,
                                  f"A seed wasn't provided by the user. The random seed is: {seed}.")
        elif not (min_seed_value <= seed <= max_seed_value):
            seed = random.randint(min_seed_value, max_seed_value)
            rank_zero_click_secho(self.local_rank,
                                  f'Invalid seed given. The new random seed is: {seed}.')
        else:
            rank_zero_click_secho(self.local_rank,f'The seed provided by the user is: {seed}.')
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
    def __init__(self, output_file, input_file, batch_size):
        self.output_file = output_file
        self.input_file = input_file
        self.start = 0
        self.stop = None
        self.batch_size = batch_size

    def on_predict_start(self):
        pass

    def on_predict_batch_end(self, outputs, batch):
        pass
