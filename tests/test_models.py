"""Tests for model training and evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from automl_pipeline.models import get_models, train_model, evaluate_model


class TestGetModels:
    def test_classification_models(self):
        models = get_models("binary_classification")
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_regression_models(self):
        models = get_models("regression")
        assert "linear_regression" in models
        assert "ridge" in models


class TestTrainAndEvaluate:
    def test_classification(self):
        X_train = pd.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
        y_train = pd.Series([0, 0, 1, 1])
        X_test = pd.DataFrame({"x": [1.5, 3.5], "y": [5.5, 7.5]})
        y_test = pd.Series([0, 1])
        model = train_model("logistic_regression", X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, "binary_classification")
        assert "accuracy" in metrics

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            train_model("unknown_model", pd.DataFrame(), pd.Series())
