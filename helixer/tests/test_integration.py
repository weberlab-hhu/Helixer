"""
Integration tests for the Helixer prediction pipeline.

These tests verify correct output structure and content at each major pipeline
stage by running actual CLI scripts and checking outputs programmatically.

Run from the Helixer/helixer directory with:
    pytest --verbose tests/test_integration.py

GFF3 generation tests require helixer_post_bin to be in PATH; they fail
when the binary is absent, as a missing binary indicates an incomplete installation.

Training and tuning pipeline stages are not covered here, the separate test_training.py handles that.
"""
import os
import pathlib
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from geenuff.applications.importer import ImportController
from helixer.export.exporter import HelixerExportController, HelixerFastaToH5Controller

# Paths and constants
# ---------------------------------------------------------------------------

HELIXER_TESTDATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testdata'))

CHUNK_SIZE = 400
SPECIES = 'dummy'
INFERENCE_SUBSEQ_LEN = 21384

REQUIRED_H5_KEYS = (
    'data/X',
    'data/y',
    'data/seqids',
    'data/species',
    'data/start_ends',
    'data/sample_weights',
)
FASTA2H5_REQUIRED_KEYS = ('data/X', 'data/seqids', 'data/species', 'data/start_ends')


# Session-scoped fixtures (pipeline stages build on each other)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def check_cwd() -> None:
    """Abort if tests are not run from the expected working directory."""
    if not os.getcwd().endswith('Helixer/helixer'):
        pytest.exit('Integration tests must be run from the Helixer/helixer directory')


@pytest.fixture(scope='session')
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp('helixer_integration')


@pytest.fixture(scope='session')
def geenuff_db(tmp_dir: pathlib.Path) -> str:
    """Import mini_test_data FASTA + GFF3 into a fresh geenuff database."""
    db_path = f'{tmp_dir}/mini_test_data.sqlite3'
    fa = os.path.join(HELIXER_TESTDATA, 'mini_test_data.fa')
    gff = os.path.join(HELIXER_TESTDATA, 'mini_test_data.gff')
    controller = ImportController(database_path=f'sqlite:///{db_path}', config={})
    controller.add_genome(fa, gff, genome_args={'species': SPECIES})
    return db_path


@pytest.fixture(scope='session')
def geenuff_h5(tmp_dir: pathlib.Path, geenuff_db: str) -> str:
    """Export the geenuff database to h5 via the Python API."""
    h5_path = f'{tmp_dir}/geenuff.h5'
    controller = HelixerExportController(geenuff_db, h5_path)
    controller.export(chunk_size=CHUNK_SIZE, write_by=5_346_000, one_hot=True, longest_only=False)
    return h5_path


@pytest.fixture(scope='session')
def fasta_h5(tmp_dir: pathlib.Path) -> str:
    """Export the same FASTA directly to h5 without going through geenuff."""
    h5_path = f'{tmp_dir}/fasta.h5'
    fa = os.path.join(HELIXER_TESTDATA, 'mini_test_data.fa')
    controller = HelixerFastaToH5Controller(fa, h5_path)
    controller.export_fasta_to_h5(
        chunk_size=CHUNK_SIZE, compression='gzip', species=SPECIES,
        write_by=CHUNK_SIZE * 10,
    )
    return h5_path


@pytest.fixture(scope='session')
def cli_result(tmp_dir: pathlib.Path, geenuff_db: str) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run geenuff2h5.py as a subprocess and return (output_h5_path, CompletedProcess)."""
    h5_path = f'{tmp_dir}/cli.h5'
    result = subprocess.run(
        ['geenuff2h5.py',
         '--input-db-path', geenuff_db,
         '--h5-output-path', h5_path,
         '--subsequence-length', str(CHUNK_SIZE)],
        capture_output=True,
        text=True,
    )
    return h5_path, result


@pytest.fixture(scope='session')
def gff3_path(tmp_dir: pathlib.Path) -> str:
    """Run helixer_post_bin using the pipeline-generated h5; fail if binary is not in PATH.
    """
    bin_path = shutil.which('helixer_post_bin')
    if bin_path is None:
        pytest.fail('helixer_post_bin not found in PATH - is it installed and on PATH or in the bin '
                    'folder of your virtual environment?')
    out = f'{tmp_dir}/output.gff3'
    result = subprocess.run(
        [bin_path,
         '--window-size', '100', '--edge-threshold', '0.1', '--peak-threshold', '0.8',
         '--min-coding-length', '60',
         'testdata/mini_test_data.h5',
         'testdata/mini_test_preds.h5',
         out],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f'helixer_post_bin failed:\n{result.stderr}'
    return out


@pytest.fixture(scope='session')
def model_path(request: pytest.FixtureRequest) -> str:
    """Return the path to the land_plant Helixer model file.

    The land_plant model is the default because the test data originates from a plant genome.
    Pass --helixer-model-path <path/to/model.h5> to pytest to override.
    """
    custom = request.config.getoption('--helixer-model-path')
    if custom is not None:
        if not os.path.isfile(custom):
            pytest.fail(
                f'model file not found: {custom}. '
                f'Verify the path passed to --helixer-model-path.'
            )
        return custom
    from helixer.core.data import MODEL_PATH
    lineage_dir = os.path.join(MODEL_PATH, 'land_plant')
    if os.path.isdir(lineage_dir):
        candidates = sorted(f for f in os.listdir(lineage_dir) if f.endswith('.h5'))
        if candidates:
            return os.path.join(lineage_dir, candidates[0])
    pytest.fail(
        f'no land_plant model found under {os.path.join(MODEL_PATH, "land_plant")}. '
        f'Download it with fetch_helixer_models.py --lineage land_plant, or pass '
        f'--helixer-model-path <path/to/model.h5> to pytest.'
    )


@pytest.fixture(scope='session')
def fasta2h5_result(tmp_dir: pathlib.Path) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run fasta2h5.py as a subprocess; return (output_h5_path, CompletedProcess)."""
    h5_path = f'{tmp_dir}/fasta2h5_cli.h5'
    fa = os.path.join(HELIXER_TESTDATA, 'mini_test_data.fa')
    result = subprocess.run(
        ['fasta2h5.py',
         '--fasta-path', fa,
         '--species', SPECIES,
         '--h5-output-path', h5_path,
         '--subsequence-length', str(INFERENCE_SUBSEQ_LEN)],
        capture_output=True,
        text=True,
    )
    return h5_path, result


@pytest.fixture(scope='session')
def hybrid_preds_result(tmp_dir: pathlib.Path, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]],
                        model_path: str) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run HybridModel.py on the fasta2h5 output; return (preds_h5_path, CompletedProcess)."""
    fasta_h5_path, _ = fasta2h5_result
    preds_path = f'{tmp_dir}/hybrid_preds.h5'
    result = subprocess.run(
        ['HybridModel.py',
         '--load-model-path', model_path,
         '--test-data', fasta_h5_path,
         '--prediction-output-path', preds_path,
         '--val-test-batch-size', '8'],
        capture_output=True,
        text=True,
    )
    return preds_path, result


@pytest.fixture(scope='session')
def helixer_gff3(tmp_dir: pathlib.Path, model_path: str) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run Helixer.py end-to-end (1-step inference); return (gff3_path, CompletedProcess)."""
    gff3 = f'{tmp_dir}/helixer.gff3'
    fa = os.path.join(HELIXER_TESTDATA, 'mini_test_data.fa')
    result = subprocess.run(
        ['Helixer.py',
         '--fasta-path', fa,
         '--species', SPECIES,
         '--gff-output-path', gff3,
         '--model-filepath', model_path,
         '--subsequence-length', str(INFERENCE_SUBSEQ_LEN),
         '--no-overlap',
         '--batch-size', '8'],
        capture_output=True,
        text=True,
    )
    return gff3, result


@pytest.fixture(scope='session')
def three_step_gff3(
    tmp_dir: pathlib.Path,
    fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]],
    hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]],
) -> str:
    """Run helixer_post_bin on 3-step pipeline output to produce a GFF3; return the path.

    Uses the same helixer_post_bin parameters as the HelixerPost README recommends/as Helixer.py defaults
    (window_size=100, edge_threshold=0.1, peak_threshold=0.8, min_coding_length=60,
    min_intron_length=50, min_intron_length_u12=30).
    """
    bin_path = shutil.which('helixer_post_bin')
    if bin_path is None:
        pytest.fail(
            'helixer_post_bin not found in PATH - is it installed and on PATH or in the '
            'bin folder of your virtual environment?'
        )
    fasta_h5_path, _ = fasta2h5_result
    preds_path, preds_result = hybrid_preds_result
    assert preds_result.returncode == 0, f'HybridModel.py failed:\n{preds_result.stderr}'
    out = f'{tmp_dir}/three_step.gff3'
    result = subprocess.run(
        [bin_path,
         '--window-size', '100', '--edge-threshold', '0.1', '--peak-threshold', '0.8',
         '--min-coding-length', '60', '--min-intron-length', '50', '--min-intron-length-u12', '30',
         fasta_h5_path, preds_path, out],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f'helixer_post_bin failed:\n{result.stderr}'
    return out


# geenuff2h5.py CLI
# ---------------------------------------------------------------------------

class TestGeenuff2h5CLI:
    def test_exit_code_zero(self, cli_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = cli_result
        assert result.returncode == 0, f'geenuff2h5.py failed:\n{result.stderr}'

    def test_output_file_created(self, cli_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = cli_result
        assert os.path.exists(h5_path)

    def test_output_is_readable_h5(self, cli_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = cli_result
        with h5py.File(h5_path, 'r') as f:
            assert len(f.keys()) > 0

    def test_required_keys_present(self, cli_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = cli_result
        with h5py.File(h5_path, 'r') as f:
            for key in REQUIRED_H5_KEYS:
                assert key in f, f'missing key: {key}'


# H5 structural consistency (mini_test_data, generated via Python API)
# ---------------------------------------------------------------------------

class TestH5Structure:
    def test_required_keys_present(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            for key in REQUIRED_H5_KEYS:
                assert key in f, f'missing key: {key}'

    def test_first_dimension_consistent_across_datasets(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            n_chunks = f['data/X'].shape[0]
            for key in REQUIRED_H5_KEYS:
                assert f[key].shape[0] == n_chunks, (
                    f'{key} first dimension {f[key].shape[0]} != {n_chunks}'
                )

    def test_chunk_length_consistent_across_sequence_datasets(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            chunk_len = f['data/X'].shape[1]
            assert f['data/y'].shape[1] == chunk_len
            assert f['data/sample_weights'].shape[1] == chunk_len

    def test_start_ends_has_two_columns(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            assert f['data/start_ends'].shape[1] == 2

    def test_x_has_four_nucleotide_channels(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            assert f['data/X'].shape[2] == 4

    def test_y_has_four_class_channels(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            assert f['data/y'].shape[2] == 4


# H5 content validity - mini_test_data (generated via Python API in this test run)
# ---------------------------------------------------------------------------

class TestH5ContentMiniTestDataGenerated:
    """Verify that the exported h5 content obeys the encoding contracts."""

    def test_x_rows_sum_to_one(self, geenuff_h5: str) -> None:
        """Every non-padding base position must encode a valid nucleotide (values sum to 1)."""
        with h5py.File(geenuff_h5, 'r') as f:
            x = f['data/X'][:]
        non_padded = x[np.any(x != 0, axis=-1)]
        assert np.allclose(non_padded.sum(axis=-1), 1.0, atol=1e-4)

    def test_x_values_in_unit_interval(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            x = f['data/X'][:]
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_y_is_one_hot_or_padding(self, geenuff_h5: str) -> None:
        """Each label position is either a one-hot class vector or an all-zero padding row."""
        with h5py.File(geenuff_h5, 'r') as f:
            y = f['data/y'][:]
        row_sums = y.sum(axis=-1)
        assert np.isin(row_sums, [0, 1]).all()

    def test_y_values_are_zero_or_one(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            y = f['data/y'][:]
        assert np.isin(y, [0, 1]).all()

    def test_sample_weights_are_binary(self, geenuff_h5: str) -> None:
        with h5py.File(geenuff_h5, 'r') as f:
            sw = f['data/sample_weights'][:]
        assert np.isin(sw, [0, 1]).all()


# H5 content validity - mini_test_data.h5 (real genome data, committed to repo)
# ---------------------------------------------------------------------------

class TestH5ContentMiniTestDataRepo:
    """Same encoding contracts, applied to the real-genome file committed to testdata/.

    These tests catch format regressions that would not be visible from the
    mini_test_data synthetic data alone.
    """

    def test_x_rows_sum_to_one(self) -> None:
        with h5py.File('testdata/mini_test_data.h5', 'r') as f:
            x = f['data/X'][:]
        non_padded = x[np.any(x != 0, axis=-1)]
        assert np.allclose(non_padded.sum(axis=-1), 1.0, atol=1e-4)

    def test_x_values_in_unit_interval(self) -> None:
        with h5py.File('testdata/mini_test_data.h5', 'r') as f:
            x = f['data/X'][:]
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_y_is_one_hot_or_padding(self) -> None:
        with h5py.File('testdata/mini_test_data.h5', 'r') as f:
            y = f['data/y'][:]
        row_sums = y.sum(axis=-1)
        assert np.isin(row_sums, [0, 1]).all()

    def test_sample_weights_are_binary(self) -> None:
        with h5py.File('testdata/mini_test_data.h5', 'r') as f:
            sw = f['data/sample_weights'][:]
        assert np.isin(sw, [0, 1]).all()


# FASTA -> H5 sequence integrity
# ---------------------------------------------------------------------------

class TestFastaSequenceIntegrity:
    def test_x_matches_geenuff_export_for_every_shared_chunk(self, geenuff_h5: str, fasta_h5: str) -> None:
        """X encoding must agree between the geenuff and FASTA-only export pipelines.

        Chunks may appear in different orders, so matching is done by
        (seqid, start, end) key.
        """
        with h5py.File(geenuff_h5, 'r') as f:
            x_db = f['data/X'][:]
            seqids_db = f['data/seqids'][:]
            se_db = f['data/start_ends'][:]
        with h5py.File(fasta_h5, 'r') as f:
            x_fa = f['data/X'][:]
            seqids_fa = f['data/seqids'][:]
            se_fa = f['data/start_ends'][:]

        fa_index = {
            (seqids_fa[i].tobytes(), int(se_fa[i, 0]), int(se_fa[i, 1])): i
            for i in range(len(x_fa))
        }
        mismatches = 0
        for i in range(len(x_db)):
            key = (seqids_db[i].tobytes(), int(se_db[i, 0]), int(se_db[i, 1]))
            if key in fa_index:
                if not np.allclose(x_db[i], x_fa[fa_index[key]]):
                    mismatches += 1
        assert mismatches == 0, f'{mismatches} X chunks differ between geenuff and FASTA exports'


# Prediction probability validity
# ---------------------------------------------------------------------------

class TestPredictionValidity:
    """Checks on mini_test_preds.h5 - committed real predictions from a trained model."""

    def test_predictions_sum_to_one(self) -> None:
        """Softmax outputs must sum to 1 per non-padding base position.

        A small fraction of positions at overlap boundaries may deviate from 1.0
        due to float16 accumulation during overlap processing; those are excluded.
        """
        with h5py.File('testdata/mini_test_preds.h5', 'r') as fp:
            preds = fp['predictions'][:].astype(np.float32)
        with h5py.File('testdata/mini_test_data.h5', 'r') as fd:
            x = fd['data/X'][:]
        non_padded_mask = np.any(x != 0, axis=-1)
        non_pad_sums = preds[non_padded_mask].sum(axis=-1)
        outlier_fraction = np.mean(np.abs(non_pad_sums - 1.0) > 0.01)
        assert outlier_fraction < 1e-4, (
            f'{outlier_fraction:.2e} of non-padding predictions deviate from sum=1 by >0.01'
        )

    def test_predictions_in_unit_interval(self) -> None:
        with h5py.File('testdata/mini_test_preds.h5', 'r') as f:
            preds = f['predictions'][:]
        assert preds.min() >= 0.0 and preds.max() <= 1.0

    def test_predictions_chunk_count_matches_data(self) -> None:
        with h5py.File('testdata/mini_test_preds.h5', 'r') as fp:
            n_pred = fp['predictions'].shape[0]
        with h5py.File('testdata/mini_test_data.h5', 'r') as fd:
            n_data = fd['data/X'].shape[0]
        assert n_pred == n_data

    def test_predictions_chunk_length_matches_data(self) -> None:
        with h5py.File('testdata/mini_test_preds.h5', 'r') as fp:
            pred_len = fp['predictions'].shape[1]
        with h5py.File('testdata/mini_test_data.h5', 'r') as fd:
            data_len = fd['data/X'].shape[1]
        assert pred_len == data_len

    def test_predictions_have_four_output_classes(self) -> None:
        with h5py.File('testdata/mini_test_preds.h5', 'r') as f:
            assert f['predictions'].shape[2] == 4


# GFF3 generation and format validity
# ---------------------------------------------------------------------------

class TestGFF3Validity:
    """End-to-end check from h5 data + predictions to a valid GFF3 output file.

    All tests in this class fail when helixer_post_bin is not in PATH, as that
    indicates an incomplete installation.
    """

    def test_gff3_file_created(self, gff3_path: str) -> None:
        assert os.path.exists(gff3_path)

    def test_gff3_is_non_empty(self, gff3_path: str) -> None:
        assert os.path.getsize(gff3_path) > 0

    def test_gff3_has_at_least_one_feature_line(self, gff3_path: str) -> None:
        with open(gff3_path) as fh:
            data_lines = [ln for ln in fh if not ln.startswith('#') and ln.strip()]
        assert len(data_lines) > 0, 'GFF3 output contains no feature lines'

    def test_gff3_data_lines_have_nine_columns(self, gff3_path: str) -> None:
        with open(gff3_path) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                assert len(columns) == 9, (
                    f'line {line_number} has {len(columns)} tab-separated columns, expected 9'
                )

    def test_gff3_coordinates_are_valid(self, gff3_path: str) -> None:
        """Start and end must be positive 1-based integers with start <= end."""
        with open(gff3_path) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                start, end = int(columns[3]), int(columns[4])
                assert start >= 1, f'line {line_number}: start={start} is below 1'
                assert start <= end, f'line {line_number}: start={start} > end={end}'

    def test_gff3_strand_is_valid(self, gff3_path: str) -> None:
        with open(gff3_path) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                assert columns[6] in ('+', '-', '.', '?'), (
                    f'line {line_number}: invalid strand value "{columns[6]}"'
                )


# fasta2h5.py CLI (step 1 of 3-step inference)
# ---------------------------------------------------------------------------

class TestFasta2h5CLI:
    """Test the fasta2h5.py command-line script."""

    def test_exit_code_zero(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = fasta2h5_result
        assert result.returncode == 0, f'fasta2h5.py failed:\n{result.stderr}'

    def test_output_file_created(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = fasta2h5_result
        assert os.path.exists(h5_path)

    def test_output_is_readable_h5(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = fasta2h5_result
        with h5py.File(h5_path, 'r') as f:
            assert len(f.keys()) > 0

    def test_required_keys_present(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = fasta2h5_result
        with h5py.File(h5_path, 'r') as f:
            for key in FASTA2H5_REQUIRED_KEYS:
                assert key in f, f'missing key: {key}'

    def test_x_has_four_nucleotide_channels(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]
                                            ) -> None:
        h5_path, _ = fasta2h5_result
        with h5py.File(h5_path, 'r') as f:
            assert f['data/X'].shape[2] == 4

    def test_x_values_in_unit_interval(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        h5_path, _ = fasta2h5_result
        with h5py.File(h5_path, 'r') as f:
            x = f['data/X'][:]
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_x_rows_sum_to_one(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        """Non-padding nucleotide encodings must sum to 1."""
        h5_path, _ = fasta2h5_result
        with h5py.File(h5_path, 'r') as f:
            x = f['data/X'][:]
        non_padded = x[np.any(x != 0, axis=-1)]
        assert np.allclose(non_padded.sum(axis=-1), 1.0, atol=1e-4)


# HybridModel.py prediction (step 2 of 3-step inference)
# ---------------------------------------------------------------------------

class TestHybridModelPrediction:
    """Test the HybridModel.py prediction step.

    Requires a land_plant model (default) or a model passed via --helixer-model-path.
    """

    def test_exit_code_zero(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = hybrid_preds_result
        assert result.returncode == 0, f'HybridModel.py failed:\n{result.stderr}'

    def test_output_file_created(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        preds_path, _ = hybrid_preds_result
        assert os.path.exists(preds_path)

    def test_predictions_key_present(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        preds_path, _ = hybrid_preds_result
        with h5py.File(preds_path, 'r') as f:
            assert 'predictions' in f

    def test_predictions_have_four_classes(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]
                                           ) -> None:
        preds_path, _ = hybrid_preds_result
        with h5py.File(preds_path, 'r') as f:
            assert f['predictions'].shape[2] == 4

    def test_predictions_chunk_count_matches_input(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]],
                                                   hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]
                                                   ) -> None:
        fasta_h5_path, _ = fasta2h5_result
        preds_path, _ = hybrid_preds_result
        with h5py.File(fasta_h5_path, 'r') as f_in:
            n_input = f_in['data/X'].shape[0]
        with h5py.File(preds_path, 'r') as f_preds:
            n_preds = f_preds['predictions'].shape[0]
        assert n_preds == n_input

    def test_predictions_chunk_length_matches_input(self, fasta2h5_result: tuple[str, subprocess.CompletedProcess[str]],
                                                    hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]
                                                    ) -> None:
        fasta_h5_path, _ = fasta2h5_result
        preds_path, _ = hybrid_preds_result
        with h5py.File(fasta_h5_path, 'r') as f_in:
            input_len = f_in['data/X'].shape[1]
        with h5py.File(preds_path, 'r') as f_preds:
            preds_len = f_preds['predictions'].shape[1]
        assert preds_len == input_len

    def test_predictions_phase_key_present(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]
                                           ) -> None:
        preds_path, _ = hybrid_preds_result
        with h5py.File(preds_path, 'r') as f:
            assert 'predictions_phase' in f

    def test_predictions_in_unit_interval(self, hybrid_preds_result: tuple[str, subprocess.CompletedProcess[str]]
                                          ) -> None:
        preds_path, _ = hybrid_preds_result
        with h5py.File(preds_path, 'r') as f:
            preds = f['predictions'][:]
            preds_phase = f['predictions_phase'][:]
        assert preds.min() >= 0.0 and preds.max() <= 1.0
        assert preds_phase.min() >= 0.0 and preds_phase.max() <= 1.0


# Helixer.py end-to-end (1-step inference)
# ---------------------------------------------------------------------------

class TestHelixerPyCLI:
    """Test the Helixer.py end-to-end pipeline (1-step inference).

    Requires a land_plant model (default) or a model passed via --helixer-model-path.
    Also requires helixer_post_bin to be in PATH.
    """

    def test_exit_code_zero(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        _, result = helixer_gff3
        assert result.returncode == 0, f'Helixer.py failed:\n{result.stderr}'

    def test_gff3_file_created(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        gff3, _ = helixer_gff3
        assert os.path.exists(gff3)

    def test_gff3_is_non_empty(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        gff3, _ = helixer_gff3
        assert os.path.getsize(gff3) > 0

    def test_gff3_data_lines_have_nine_columns(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]
                                               ) -> None:
        gff3, _ = helixer_gff3
        with open(gff3) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                assert len(columns) == 9, (
                    f'line {line_number} has {len(columns)} tab-separated columns, expected 9'
                )

    def test_gff3_coordinates_are_valid(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        """Start and end must be positive 1-based integers with start <= end."""
        gff3, _ = helixer_gff3
        with open(gff3) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                start, end = int(columns[3]), int(columns[4])
                assert start >= 1, f'line {line_number}: start={start} is below 1'
                assert start <= end, f'line {line_number}: start={start} > end={end}'

    def test_gff3_strand_is_valid(self, helixer_gff3: tuple[str, subprocess.CompletedProcess[str]]) -> None:
        gff3, _ = helixer_gff3
        with open(gff3) as fh:
            for line_number, line in enumerate(fh, 1):
                if line.startswith('#') or not line.strip():
                    continue
                columns = line.rstrip('\n').split('\t')
                assert columns[6] in ('+', '-', '.', '?'), (
                    f'line {line_number}: invalid strand value "{columns[6]}"'
                )


# 1-step vs 3-step GFF3 equivalence
# ---------------------------------------------------------------------------

class TestOneVsThreeStepEquivalence:
    """Verify that 1-step (Helixer.py) and 3-step inference produce identical GFF3 output.

    Both paths use the same genome, model, batch size, no-overlap mode, and
    helixer_post_bin parameters (window=100, edge=0.1, peak=0.8, min_coding=60, min_intron=50,
    min_intron_u12=30).
    Phase predictions are produced by the model output head in both cases.
    """

    def test_gff3_data_lines_identical(
        self,
        helixer_gff3: tuple[str, subprocess.CompletedProcess[str]],
        three_step_gff3: str,
    ) -> None:
        """Non-comment GFF3 data lines must be identical between 1-step and 3-step inference."""
        one_step_path, _ = helixer_gff3
        with open(one_step_path) as f1, open(three_step_gff3) as f2:
            one_step_lines = [ln for ln in f1 if not ln.startswith('#')]
            three_step_lines = [ln for ln in f2 if not ln.startswith('#')]
        assert one_step_lines == three_step_lines, (
            f'GFF3 data lines differ: 1-step has {len(one_step_lines)} lines, '
            f'3-step has {len(three_step_lines)} lines'
        )
