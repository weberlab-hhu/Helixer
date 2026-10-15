"""Training pipeline integration tests for Helixer.

These tests cover the full training workflow using Arabidopsis lyrata chromosomes
downloaded from Ensembl Genomes FTP:
  1. Data download from FTP
  2. Geenuff import (train chromosome uses uncompressed FASTA + zip GFF3 to exercise
     Geenuff's multi-format support; val/test chromosomes stay as .gz)
  3. H5 export
  4. Training (HybridModel.py, 5 epochs)
  5. Tuning (resume training, 2 epochs)
  6. Eval with and without overlap (reports metrics, does not generate predictions)

Prediction output format is tested separately in test_integration.py.

Network access (FTP) and significant compute time (GPU recommended) are required.
All tests are marked pytest.mark.training and are not part of the default suite.

Run from the Helixer/helixer directory with:
    pytest --verbose tests/test_training.py
Add --training-logs to write the output of each HybridModel.py run to training.log, tuning.log,
eval.log and eval_overlap.log in the test's working directory (under pytest's basetemp).
"""
import gzip
import os
import pathlib
import re
import shutil
import subprocess
import urllib.request
import zipfile

import h5py
import pytest

from geenuff.applications.importer import ImportController
from helixer.export.exporter import HelixerExportController

# Paths and constants
# ---------------------------------------------------------------------------

SPECIES = 'Arabidopsis_lyrata'
_FTP_FA = 'ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna'
_FTP_GFF = 'ftp://ftp.ensemblgenomes.org/pub/plants/release-47/gff3/arabidopsis_lyrata'

CHROMOSOME_URLS = {
    'train': {
        'fa': f'{_FTP_FA}/Arabidopsis_lyrata.v.1.0.dna.chromosome.1.fa.gz',
        'gff': f'{_FTP_GFF}/Arabidopsis_lyrata.v.1.0.47.chromosome.1.gff3.gz',
    },
    'validation': {
        'fa': f'{_FTP_FA}/Arabidopsis_lyrata.v.1.0.dna.chromosome.7.fa.gz',
        'gff': f'{_FTP_GFF}/Arabidopsis_lyrata.v.1.0.47.chromosome.7.gff3.gz',
    },
    'test': {
        'fa': f'{_FTP_FA}/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz',
        'gff': f'{_FTP_GFF}/Arabidopsis_lyrata.v.1.0.47.chromosome.8.gff3.gz',
    },
}

REQUIRED_H5_KEYS = (
    'data/X',
    'data/y',
    'data/seqids',
    'data/species',
    'data/start_ends',
    'data/sample_weights',
)


# Helpers
# ---------------------------------------------------------------------------

def _gunzip(src: str, dest: str) -> None:
    """Decompress a gzip file to dest."""
    with gzip.open(src, 'rb') as f_in, open(dest, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def check_cwd() -> None:
    """Abort if tests are not run from the expected working directory."""
    if not os.getcwd().endswith('Helixer/helixer'):
        pytest.exit('Training integration tests must be run from the Helixer/helixer directory')


@pytest.fixture(scope='session')
def wdir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Session-scoped directory for all training test artifacts."""
    return tmp_path_factory.mktemp('helixer_training')


@pytest.fixture(scope='session')
def downloaded_data(wdir: pathlib.Path) -> dict[str, dict[str, str]]:
    """Download Arabidopsis lyrata chromosomes from Ensembl Genomes FTP.

    The train chromosome is decompressed to plain FASTA and zip-archived GFF3
    to test geenuff's support for those formats. Val and test chromosomes are
    kept as the downloaded .gz files.

    Returns a dict mapping role ('train', 'validation', 'test') to
    {'fa': fasta_path, 'gff': gff_path}.
    """
    paths = {}
    for role, urls in CHROMOSOME_URLS.items():
        role_dir = f'{wdir}/{role}'
        os.makedirs(role_dir)
        fa_gz = f'{role_dir}/genome.fa.gz'
        gff_gz = f'{role_dir}/annotation.gff3.gz'
        urllib.request.urlretrieve(urls['fa'], fa_gz)
        urllib.request.urlretrieve(urls['gff'], gff_gz)
        if role == 'train':
            fa = f'{role_dir}/genome.fa'
            gff_plain = f'{role_dir}/annotation.gff3'
            gff_zip = f'{role_dir}/annotation.zip'
            _gunzip(fa_gz, fa)
            _gunzip(gff_gz, gff_plain)
            with zipfile.ZipFile(gff_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(gff_plain, 'annotation.gff3')
            os.remove(gff_plain)
            paths[role] = {'fa': fa, 'gff': gff_zip}
        else:
            paths[role] = {'fa': fa_gz, 'gff': gff_gz}
    return paths


@pytest.fixture(scope='session')
def geenuff_dbs(wdir: pathlib.Path, downloaded_data: dict[str, dict[str, str]]) -> dict[str, str]:
    """Import each chromosome into its own geenuff SQLite database.

    Returns a dict mapping role to the database path.
    """
    dbs = {}
    for role, file_paths in downloaded_data.items():
        db_path = f'{wdir}/{role}/geenuff.sqlite3'
        controller = ImportController(database_path=f'sqlite:///{db_path}', config={})
        controller.add_genome(
            file_paths['fa'], file_paths['gff'],
            genome_args={'species': SPECIES},
        )
        dbs[role] = db_path
    return dbs


@pytest.fixture(scope='session')
def train_h5_dir(wdir: pathlib.Path, geenuff_dbs: dict[str, str]) -> str:
    """Export training and validation chromosomes to h5 in a shared directory.

    HybridModel.py training mode expects files named training_data*.h5 and
    validation_data*.h5 in the directory passed via --data-dir.

    Returns the directory path.
    """
    h5_dir = f'{wdir}/h5s_train'
    os.makedirs(h5_dir)
    name_map = {'train': 'training_data.h5', 'validation': 'validation_data.h5'}
    for role, h5_name in name_map.items():
        controller = HelixerExportController(geenuff_dbs[role], f'{h5_dir}/{h5_name}')
        controller.export(chunk_size=21384, write_by=5_346_000, one_hot=True)
    return h5_dir


@pytest.fixture(scope='session')
def test_h5_path(wdir: pathlib.Path, geenuff_dbs: dict[str, str]) -> str:
    """Export the test chromosome to a standalone h5 file; return the path."""
    h5_path = f'{wdir}/test_data.h5'
    controller = HelixerExportController(geenuff_dbs['test'], h5_path)
    controller.export(chunk_size=21384, write_by=5_346_000, one_hot=True)
    return h5_path


@pytest.fixture(scope='session')
def log_dir(pytestconfig: pytest.Config, wdir: pathlib.Path) -> pathlib.Path | None:
    """The directory to write the HybridModel.py logs to, given --training-logs; None otherwise."""
    return wdir if pytestconfig.getoption('--training-logs') else None


def readable_output(output: str) -> str:
    """Keras' progress bar reduced to the last line of each epoch: the bar's updates, each ending in
    backspaces and some running straight into the next log message, are dropped."""
    output = output.replace('\b', '')
    output = re.sub(r'(?<=\S)(?=\d{4}-\d\d-\d\d \d\d:\d\d:\d\d, [A-Z]: )', '\n', output)
    return '\n'.join(line for line in output.split('\n') if ' - ETA: ' not in line)


def run_logged(cmd: list[str], log_dir: pathlib.Path | None, name: str) -> subprocess.CompletedProcess[str]:
    """Runs a command; with a log_dir, writes its stdout and stderr to <log_dir>/<name>.log."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log_dir is not None:
        with open(log_dir / f'{name}.log', 'w') as f:
            f.write(f'$ {" ".join(cmd)}\n\n--- stdout ---\n{readable_output(result.stdout)}\n'
                    f'--- stderr ---\n{readable_output(result.stderr)}')
    return result


@pytest.fixture(scope='session')
def training_result(wdir: pathlib.Path, train_h5_dir: str, log_dir: pathlib.Path | None
                    ) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Train HybridModel.py for 5 epochs; return (model_path, CompletedProcess)."""
    model_path = f'{wdir}/training_model.h5'
    result = run_logged(
        ['HybridModel.py',
         '--data-dir', train_h5_dir,
         '--class-weights', '[0.7, 1.6, 1.2, 1.2]',
         '--transition-weights', '[1, 12, 3, 1, 12, 3]',
         '--epochs', '5',
         '--save-model-path', model_path],
        log_dir, 'training',
    )
    return model_path, result


@pytest.fixture(scope='session')
def tuning_result(wdir: pathlib.Path, train_h5_dir: str, training_result: tuple[str, subprocess.CompletedProcess[str]],
                  log_dir: pathlib.Path | None) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Resume training (tune) for 2 epochs; return (model_path, CompletedProcess)."""
    trained_model_path, _ = training_result
    tuned_model_path = f'{wdir}/tuned_model.h5'
    result = run_logged(
        ['HybridModel.py',
         '--data-dir', train_h5_dir,
         '--class-weights', '[0.7, 1.6, 1.2, 1.2]',
         '--transition-weights', '[1, 12, 3, 1, 12, 3]',
         '--epochs', '2',
         '--load-model-path', trained_model_path,
         '--patience', '10',
         '--check-every-nth-batch', '150',
         '--resume-training',
         '--save-model-path', tuned_model_path],
        log_dir, 'tuning',
    )
    return tuned_model_path, result


@pytest.fixture(scope='session')
def eval_result(test_h5_path: str, tuning_result: tuple[str, subprocess.CompletedProcess[str]],
                log_dir: pathlib.Path | None) -> subprocess.CompletedProcess[str]:
    """Run eval (no overlap) with the tuned model; return CompletedProcess."""
    tuned_model_path, _ = tuning_result
    return run_logged(
        ['HybridModel.py',
         '--test-data', test_h5_path,
         '--eval',
         '--load-model-path', tuned_model_path],
        log_dir, 'eval',
    )


@pytest.fixture(scope='session')
def eval_overlap_result(test_h5_path: str, tuning_result: tuple[str, subprocess.CompletedProcess[str]],
                        log_dir: pathlib.Path | None) -> subprocess.CompletedProcess[str]:
    """Run eval with overlap with the tuned model; return CompletedProcess."""
    tuned_model_path, _ = tuning_result
    return run_logged(
        ['HybridModel.py',
         '--test-data', test_h5_path,
         '--eval',
         '--overlap',
         '--load-model-path', tuned_model_path],
        log_dir, 'eval_overlap',
    )


# Data download
# ---------------------------------------------------------------------------

class TestDataDownload:
    """Verify that all chromosome files were downloaded from FTP."""

    def test_train_files_downloaded(self, downloaded_data: dict[str, dict[str, str]]) -> None:
        assert os.path.getsize(downloaded_data['train']['fa']) > 0
        assert os.path.getsize(downloaded_data['train']['gff']) > 0

    def test_validation_files_downloaded(self, downloaded_data: dict[str, dict[str, str]]) -> None:
        assert os.path.getsize(downloaded_data['validation']['fa']) > 0
        assert os.path.getsize(downloaded_data['validation']['gff']) > 0

    def test_test_files_downloaded(self, downloaded_data: dict[str, dict[str, str]]) -> None:
        assert os.path.getsize(downloaded_data['test']['fa']) > 0
        assert os.path.getsize(downloaded_data['test']['gff']) > 0


# H5 export
# ---------------------------------------------------------------------------

class TestH5Export:
    """Verify h5 files produced from the downloaded chromosomes."""

    def test_training_h5_required_keys(self, train_h5_dir: str) -> None:
        with h5py.File(f'{train_h5_dir}/training_data.h5', 'r') as f:
            for key in REQUIRED_H5_KEYS:
                assert key in f, f'missing key: {key}'

    def test_validation_h5_required_keys(self, train_h5_dir: str) -> None:
        with h5py.File(f'{train_h5_dir}/validation_data.h5', 'r') as f:
            for key in REQUIRED_H5_KEYS:
                assert key in f, f'missing key: {key}'

    def test_test_h5_required_keys(self, test_h5_path: str) -> None:
        with h5py.File(test_h5_path, 'r') as f:
            for key in REQUIRED_H5_KEYS:
                assert key in f, f'missing key: {key}'

    def test_h5_chunk_size(self, test_h5_path: str) -> None:
        with h5py.File(test_h5_path, 'r') as f:
            assert f['data/X'].shape[1] == 21384


# Training
# ---------------------------------------------------------------------------

class TestTraining:
    """Verify that HybridModel.py training completes and saves a model."""

    def test_exit_code_zero(self, training_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = training_result
        assert result.returncode == 0, f'training failed:\n{result.stderr}'

    def test_model_file_created(self, training_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        model_path, _ = training_result
        assert os.path.exists(model_path), ('no model saved, the validation genic F1 never rose above 0; rerun with '
                                            '--training-logs to see the training output')

    def test_model_is_valid_h5(self, training_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        model_path, _ = training_result
        with h5py.File(model_path, 'r') as f:
            assert len(f.keys()) > 0


# Tuning
# ---------------------------------------------------------------------------

class TestTuning:
    """Verify that resume training (tuning) completes and saves a model."""

    def test_exit_code_zero(self, tuning_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = tuning_result
        assert result.returncode == 0, f'tuning failed:\n{result.stderr}'

    def test_model_file_created(self, tuning_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        model_path, _ = tuning_result
        assert os.path.exists(model_path), 'no model saved; rerun with --training-logs to see the tuning output'

    def test_model_is_valid_h5(self, tuning_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        model_path, _ = tuning_result
        with h5py.File(model_path, 'r') as f:
            assert len(f.keys()) > 0


# Eval
# ---------------------------------------------------------------------------

class TestEval:
    """Verify that evaluation mode exits cleanly with and without overlap."""

    def test_eval_exit_code_zero(self, eval_result: subprocess.CompletedProcess[str]) -> None:
        assert eval_result.returncode == 0, f'eval failed:\n{eval_result.stderr}'

    def test_eval_overlap_exit_code_zero(self, eval_overlap_result: subprocess.CompletedProcess[str]) -> None:
        assert eval_overlap_result.returncode == 0, (
            f'eval with overlap failed:\n{eval_overlap_result.stderr}'
        )
