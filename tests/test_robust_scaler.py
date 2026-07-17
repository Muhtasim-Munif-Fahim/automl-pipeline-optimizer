"""Tests for robust scaler module."""

from __future__ import annotations

import pandas as pd

from automl_pipeline.robust_scaler import RobustScalerDF


class TestRobustScalerDF:
    def test_fit_transform(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaler = RobustScalerDF()
        result = scaler.fit_transform(X)
        assert result.shape == X.shape
        assert isinstance(result, pd.DataFrame)

    def test_custom_quantile_range(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = RobustScalerDF(quantile_range=(10.0, 90.0))
        result = scaler.fit_transform(X)
        assert not result.isna().any().any()
