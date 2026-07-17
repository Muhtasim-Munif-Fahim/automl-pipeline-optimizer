"""Tests for profiling module."""

from __future__ import annotations

import pandas as pd
import pytest

from automl_pipeline.feature_selection import variance_threshold_selection


class TestFeatureSelection:
    def test_variance_threshold(self):
        X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
        result = variance_threshold_selection(X, threshold=0.5)
        assert "a" not in result.columns
