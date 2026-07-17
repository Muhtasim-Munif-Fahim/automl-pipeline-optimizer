"""Tests for feature selection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from automl_pipeline.feature_selection import (
    get_feature_importance,
    mutual_information_selection,
    variance_threshold_selection,
)


class TestVarianceThresholdSelection:
    def test_removes_low_variance(self):
        X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4], "c": [0, 0, 0, 0]})
        result = variance_threshold_selection(X, threshold=0.01)
        assert "b" in result.columns
        assert "a" not in result.columns or "c" not in result.columns

    def test_keeps_all_when_high_variance(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        result = variance_threshold_selection(X, threshold=0.01)
        assert list(result.columns) == ["a", "b"]


class TestMutualInformationSelection:
    def test_selects_top_k(self):
        X = pd.DataFrame({"a": range(20), "b": range(20, 40), "c": np.random.randn(20)})
        y = pd.Series(range(20))
        result = mutual_information_selection(X, y, k=2, problem_type="regression")
        assert len(result.columns) == 2

    def test_handles_large_k(self):
        X = pd.DataFrame({"a": range(10), "b": range(10, 20), "c": range(20, 30)})
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = mutual_information_selection(X, y, k=10, problem_type="classification")
        assert len(result.columns) <= 3


class TestGetFeatureImportance:
    def test_tree_importance(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y = [0, 0, 1, 1]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        scores = get_feature_importance(model, ["a", "b"], top_n=1)
        assert len(scores) == 1

    def test_unknown_model(self):
        scores = get_feature_importance(object(), ["a"], top_n=5)
        assert scores == {}

    def test_empty_feature_importance(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = [0, 1]
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        from sklearn.linear_model import LogisticRegression
        linear = LogisticRegression(max_iter=100, random_state=42)
        linear.fit(X, y)
        scores = get_feature_importance(linear, ["a", "b"], top_n=5)
        assert len(scores) == 2
