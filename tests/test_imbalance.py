"""Tests for imbalanced data handling."""

from __future__ import annotations

import pandas as pd
import pytest

from automl_pipeline.imbalance import compute_class_weights, downsample_majority


class TestClassWeights:
    def test_balanced_weights(self):
        y = pd.Series([0, 0, 1, 1])
        weights = compute_class_weights(y)
        assert weights[0] == weights[1]


class TestDownsample:
    def test_balanced_output(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
        y = pd.Series([0, 0, 0, 0, 1, 1])
        X_res, y_res = downsample_majority(X, y)
        assert y_res.value_counts().min() == y_res.value_counts().max()
