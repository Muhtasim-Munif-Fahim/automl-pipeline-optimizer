"""Tests for profiling module."""

from __future__ import annotations

import pandas as pd
import pytest

from automl_pipeline.profiling import profile_dataframe, detect_issues


class TestProfiling:
    def test_profile_returns_keys(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        profile = profile_dataframe(df)
        assert "rows" in profile
        assert "columns" in profile
        assert "dtypes" in profile

    def test_detect_missing(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        issues = detect_issues(df)
        assert any(i["type"] == "missing_values" for i in issues)
