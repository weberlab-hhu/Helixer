import os
import click
import numpy as np
import torch


def validate_path_fragment(ctx, param, path):
    """Check if the path exists up until the final filename. The file itself will be created by Helixer."""
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f'{dir_path} given to option {param.name} does not exist')


def validate_weights(ctx, param, weights):
    weights = eval(weights)
    if not isinstance(weights, (list, np.ndarray, type(None))):
        raise ValueError(f'{param.name} evaluated to {weights} of type {type(weights)}; '
                         f'this commonly means you need to remove nested quotes (if not starting with nni)')
    if type(weights) is list:
        return np.array(weights, dtype=np.float32)
    return weights


def validate_device(ctx, param, device):
    if device == 'gpu' and not torch.cuda.is_available():
        msg = ('No GPUs available on your machine. Please check if your machine has a GPU '
               'and if PyTorch is installed correctly. Switching to CPU.')
        click.secho(msg, fg='yellow', bold=True)
        return 'cpu'
    return device
# todo: add model validation, load and merge parameters and so on
