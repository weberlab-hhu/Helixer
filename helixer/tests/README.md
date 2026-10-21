# Helixer tests
Run all tests from the `Helixer/helixer` directory.

## Dependency checks — `test_imports.py`
Verifies that all required Python packages (`helixer`, `geenuff`) are importable
and that the `helixer_post_bin` binary is on PATH. Run this first after a fresh install to
get actionable error messages for missing dependencies.

```bash
pytest --verbose tests/test_imports.py
```

## Unit tests — `test_helixer.py`
Covers numerification, annotation encoding, sequence slicing, overlap processing,
confusion matrix computation, and coverage utilities. No model files or external
binaries required.

```bash
pytest --verbose tests/test_helixer.py
```

## Integration tests — `test_integration.py`
End-to-end pipeline tests that run the actual CLI scripts and verify their outputs.
All tests must be run from `Helixer/helixer`. Tiny delays during testing (~10-30 sec.) are expected;
even though the test data is small some processes like `Helixer.py` still require a few seconds to run.

```bash
pytest --verbose tests/test_integration.py
```

### Test classes

| Class                           | What it tests                                                                                                                                                           |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TestGeenuff2h5CLI`             | `geenuff2h5.py` subprocess: exit code, output file, required h5 keys                                                                                                    |
| `TestH5Structure`               | Structural consistency of a geenuff-exported h5 (key presence, dimension alignment)                                                                                     |
| `TestH5ContentDummyloci`        | Expected encoding for the geenuff-exported h5 file: nucleotide sums, one-hot labels, binary sample weights                                                              |
| `TestH5ContentMiniTestData`     | Same expected encoding for the h5 file `testdata/mini_test_data.h5` contained in this repository                                                                        |
| `TestFastaSequenceIntegrity`    | X encoding agreement between the geenuff and direct-FASTA export paths                                                                                                  |
| `TestPredictionValidity`        | Format and range the h5 file `testdata/mini_test_preds.h5` contained in this repository                                                                                 |
| `TestGFF3Validity`              | `helixer_post_bin` output: file created, 9-column format, valid coordinates and strand                                                                                  |
| `TestFasta2h5CLI`               | `fasta2h5.py` subprocess: exit code, output file, required h5 keys, nucleotide encoding                                                                                 |
| `TestHybridModelPrediction`     | `HybridModel.py` prediction: exit code, predictions and predictions_phase keys, shape consistency, unit-interval values (predictions and phase: values between 0 and 1) |
| `TestHelixerPyCLI`              | `Helixer.py` end-to-end (1-step): exit code, GFF3 file created, 9-column format, valid coordinates and strand                                                           |
| `TestOneVsThreeStepEquivalence` | GFF3 data lines from 1-step (`Helixer.py`) and 3-step (`fasta2h5.py` + `HybridModel.py` + `helixer_post_bin`) inference must be identical                               |

### Requirements
- **`helixer_post_bin`** must be in `PATH` or your virtual Python environments bin folder for
  `TestGFF3Validity`, `TestHelixerPyCLI`, and `TestOneVsThreeStepEquivalence`.
  Missing it is treated as a test failure (incomplete installation).
- **A land_plant model** is required for `TestHybridModelPrediction` and `TestHelixerPyCLI`.
  The default location is `~/.local/share/Helixer/models/land_plant/<model>.h5`.
  Download it with:
  ```bash
  fetch_helixer_models.py --lineage land_plant
  # or
  fetch_helixer_models.py  # fetches all models
  ```
  To use a model stored elsewhere, pass `--helixer-model-path` to pytest:
  ```bash
  pytest --verbose tests/test_integration.py --helixer-model-path /path/to/model.h5
  ```

## Training tests — `test_training.py`
End-to-end tests for the training workflow (mostly for developers).
Downloads _Arabidopsis lyrata_ chromosomes from Ensembl Genomes FTP, so network access
and significant compute time are required (GPU recommended). It can be slow as downloading 
(depending on Network speed) and training can take a few minutes.
On an NVIDIA GeForce RTX 4080 SUPER this script takes ~11 minutes to run.

```bash
pytest --verbose tests/test_training.py
```

### Test classes

| Class              | What it tests                                                                        |
|--------------------|--------------------------------------------------------------------------------------|
| `TestDataDownload` | FTP downloads for all three chromosomes (train/val/test)                             |
| `TestH5Export`     | Required h5 keys and chunk size for exported training, validation, and test h5 files |
| `TestTraining`     | HybridModel.py training (5 epochs): exit code, model file created, valid h5          |
| `TestTuning`       | Resume training (2 epochs): exit code, model file created, valid h5                  |
| `TestEval`         | Eval mode exit code with and without overlap                                         |

The train chromosome uses an uncompressed FASTA and a zip-archived GFF3 to exercise
geenuff's multi-format input support.
