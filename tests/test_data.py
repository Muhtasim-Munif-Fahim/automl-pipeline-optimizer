"""Tests for data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl_pipeline.data import load_data, validate_data, detect_problem_type, split_data


class TestLoadData:
    def test_csv(self, tmp_path: Path):
        p = tmp_path / "test.csv"
        p.write_text("a,b\n1,2\n3,4\n")
        df = load_data(p)
        assert len(df) == 2

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent.csv")

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_data(p)

    def test_unsupported_format(self, tmp_path: Path):
        p = tmp_path / "data.xyz"
        p.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_data(p)


class TestValidateData:
    def test_valid(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        errors = validate_data(df, "y")
        assert errors == []

    def test_missing_target(self):
        df = pd.DataFrame({"x": [1, 2]})
        errors = validate_data(df, "y")
        assert any("target" in e for e in errors)

    def test_too_small(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        errors = validate_data(df, "y")
        assert any("small" in e for e in errors)


class TestDetectProblemType:
    def test_binary(self):
        s = pd.Series([0, 1, 0, 1, 0])
        assert detect_problem_type(s) == "binary_classification"

    def test_multiclass(self):
        s = pd.Series([0, 1, 2, 3, 4, 5])
        assert detect_problem_type(s) == "multiclass_classification"

    def test_regression(self):
        s = pd.Series([1.5, 2.3, 3.7, 4.1])
        assert detect_problem_type(s) == "regression"


class TestSplitData:
    def test_split(self):
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        X_train, X_test, y_train, y_test = split_data(df, "y", test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
