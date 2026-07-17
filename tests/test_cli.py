"""Tests for CLI module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from automl_pipeline.cli import main


class TestCli:
    def test_version(self):
        with patch("sys.argv", ["prog", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2

    def test_missing_data(self):
        with patch("sys.argv", ["prog", "--data", "nonexistent.csv", "--target", "y"]):
            assert main() == 1
