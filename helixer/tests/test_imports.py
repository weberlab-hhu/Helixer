"""Dependency availability tests for Helixer.

Run before other tests to verify all required packages and binaries are installed:
    pytest --verbose tests/test_imports.py
"""
import importlib
import shutil

import pytest


def _assert_importable(module: str, hint: str) -> None:
    """Import module and fail with an installation hint if it is missing."""
    try:
        importlib.import_module(module)
    except ImportError:
        pytest.fail(f"Required package '{module}' is not installed. Install with:\n  {hint}")


def test_helixer_importable() -> None:
    _assert_importable('helixer', 'pip install .')


def test_geenuff_importable() -> None:
    _assert_importable('geenuff', 'pip install git+https://github.com/usadellab/GeenuFF')


def test_dustdas_importable() -> None:
    _assert_importable('dustdas', 'pip install git+https://github.com/weberlab-hhu/dustdas')


def test_helixer_post_bin_available() -> None:
    """helixer_post_bin must be on PATH or in the virtual environments bin folder for GFF3 generation."""
    if shutil.which('helixer_post_bin') is None:
        pytest.fail(
            'helixer_post_bin not found in PATH or in the virtual environments bin folder.\n'
            'Install from: https://github.com/usadellab/HelixerPost'
        )
