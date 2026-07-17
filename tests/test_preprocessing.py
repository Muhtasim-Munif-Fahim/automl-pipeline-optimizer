"""Tests for preprocessing module."""

from __future__ import annotations

import pandas as pd

from automl_pipeline.preprocessing import (
    encode_categorical,
    handle_outliers,
    impute_missing,
    scale_numeric,
)


class TestImputeMissing:
    def test_numeric_median(self):
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
        result = impute_missing(df)
        assert result["x"].isna().sum() == 0
        assert result["x"].iloc[2] == 2.0

    def test_categorical_mode(self):
        df = pd.DataFrame({"x": ["a", "b", None, "a"]})
        result = impute_missing(df)
        assert result["x"].isna().sum() == 0


class TestEncodeCategorical:
    def test_onehot(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        result = encode_categorical(df)
        assert "color" not in result.columns
        assert "color_blue" in result.columns or "color_green" in result.columns


class TestScaleNumeric:
    def test_standard(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = scale_numeric(df)
        assert abs(result["x"].mean()) < 0.001


class TestHandleOutliers:
    def test_clip(self):
        df = pd.DataFrame({"x": [1, 2, 3, 100, 4, 5]})
        result = handle_outliers(df, method="clip", threshold=2.0)
        assert result["x"].max() < df["x"].max()
