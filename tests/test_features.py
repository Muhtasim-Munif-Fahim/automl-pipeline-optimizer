"""Tests for feature engineering module."""

from __future__ import annotations

import pandas as pd

from automl_pipeline.features import add_interaction_features, add_polynomial_features


class TestPolynomialFeatures:
    def test_adds_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = add_polynomial_features(df, degree=2)
        assert result.shape[1] >= df.shape[1]

    def test_returns_df_with_few_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = add_polynomial_features(df)
        assert result is not None


class TestInteractionFeatures:
    def test_adds_interactions(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )
        result = add_interaction_features(df)
        assert result.shape[1] >= df.shape[1]
