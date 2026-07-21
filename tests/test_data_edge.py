"""Tests for data module edge cases (uncovered lines)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from automl_pipeline.data import load_data, validate_data, detect_problem_type


class TestLoadDataEdgeCases:
    def test_parquet_format(self, tmp_path: Path):
        p = tmp_path / "test.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(p)
        df = load_data(p)
        assert len(df) == 2

    def test_json_format(self, tmp_path: Path):
        p = tmp_path / "test.json"
        pd.DataFrame({"a": [1, 2]}).to_json(p)
        df = load_data(p)
        assert len(df) == 2

    def test_empty_file_raises(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        import pandas as pd
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            load_data(p)


class TestDetectProblemTypeEdge:
    def test_float_target_regression(self):
        s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert detect_problem_type(s) == "regression"

    def test_int_binary(self):
        s = pd.Series([0, 1, 0, 1, 0, 1])
        assert detect_problem_type(s) == "binary_classification"

    def test_int_multiclass(self):
        s = pd.Series([0, 1, 2, 3, 4])
        assert detect_problem_type(s) == "multiclass_classification"
