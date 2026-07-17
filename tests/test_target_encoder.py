"""Tests for target encoding module."""

from __future__ import annotations

import pandas as pd

from automl_pipeline.target_encoder import TargetEncoder


class TestTargetEncoder:
    def test_fit_transform(self):
        X = pd.DataFrame({"cat": ["a", "a", "b", "b", "c"], "num": [1, 2, 3, 4, 5]})
        y = pd.Series([1, 2, 1, 2, 1])
        encoder = TargetEncoder(smoothing=1.0)
        result = encoder.fit_transform(X, y)
        assert "cat" in result.columns
        assert result["cat"].dtype.kind in ("i", "f")

    def test_transform_without_y(self):
        X = pd.DataFrame({"cat": ["a", "b"]})
        encoder = TargetEncoder()
        encoder.fit(X[["cat"]], pd.Series([1.0, 2.0]))
        result = encoder.transform(X[["cat"]])
        assert result is not None
