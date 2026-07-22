"""Tests for config validator module."""

from __future__ import annotations

from automl_pipeline.config_validator import validate_pipeline_config


class TestConfigValidator:
    def test_missing_required(self):
        errors = validate_pipeline_config({})
        assert any("Missing" in e for e in errors)

    def test_valid_config(self):
        config = {"data": "data.csv", "target": "y", "test_size": 0.2, "cv_folds": 5}
        errors = validate_pipeline_config(config)
        assert errors == []

    def test_invalid_test_size(self):
        config = {"data": "d.csv", "target": "y", "test_size": 1.5}
        errors = validate_pipeline_config(config)
        assert any("test_size" in e for e in errors)

    def test_invalid_cv_folds(self):
        config = {"data": "d.csv", "target": "y", "cv_folds": 0}
        errors = validate_pipeline_config(config)
        assert any("cv_folds" in e for e in errors)

    def test_wrong_type(self):
        config = {"data": "d.csv", "target": "y", "test_size": "0.2"}
        errors = validate_pipeline_config(config)
        assert any("test_size" in e for e in errors)
