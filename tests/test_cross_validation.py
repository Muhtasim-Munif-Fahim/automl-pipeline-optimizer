"""Tests for cross-validation module."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from automl_pipeline.cross_validation import get_cv_splits, cross_validate


class TestGetCVSplits:
    def test_stratified(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cv = get_cv_splits(X, y, n_folds=3, strategy="stratified")
        assert cv is not None


class TestCrossValidate:
    def test_returns_dict(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        model = LogisticRegression(max_iter=1000)
        result = cross_validate(model, X, y, cv=3, problem_type="classification")
        assert "scores" in result
        assert "mean" in result
