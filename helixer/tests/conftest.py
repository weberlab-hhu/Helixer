import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--helixer-model-path',
        action='store',
        default=None,
        metavar='PATH',
        help=(
            'Path to a Helixer model .h5 file used in inference tests. '
            'When omitted, the land_plant model is used from the default Helixer '
            'model directory (~/.local/share/Helixer/models/land_plant on Linux).'
        ),
    )
    parser.addoption(
        '--training-logs',
        action='store_true',
        help=(
            'Write the output of each HybridModel.py run of test_training.py to a .log file in the '
            'test\'s working directory, the progress bar reduced to the last line of each epoch.'
        ),
    )
