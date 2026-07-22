"""Tests for binning module."""

from __future__ import annotations

import pandas as pd
import pytest

from automl_pipeline.binning import equal_width_bins, quantile_bins, auto_bin


class TestBinning:
    def test_equal_width_output_type(self):
        s = pd.Series(range(100))
        result = equal_width_bins(s, n_bins=5)
        assert result.nunique() == 5

    def test_quantile_output_type(self):
        s = pd.Series(range(100))
        result = quantile_bins(s, n_bins=5)
        assert result.nunique() <= 5

    def test_auto_bin_default_strategy(self):
        s = pd.Series(range(100))
        result = auto_bin(s, strategy="quantile", n_bins=10)
        assert result.nunique() <= 10
