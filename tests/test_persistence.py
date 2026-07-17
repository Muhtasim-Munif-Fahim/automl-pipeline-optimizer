"""Tests for model persistence module."""

from __future__ import annotations

from pathlib import Path

import pytest

from automl_pipeline.persistence import (
    load_model,
    load_pipeline_artifacts,
    save_model,
    save_pipeline_artifacts,
)


class TestSaveLoadModel:
    def test_save_and_load_model(self, tmp_path: Path):
        model = {"dummy": "classifier"}
        path = tmp_path / "model.pkl"
        save_model(model, path)
        assert path.exists()
        loaded = load_model(path)
        assert loaded == model

    def test_load_missing_model(self):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent.pkl")


class TestPipelineArtifacts:
    def test_save_and_load_artifacts(self, tmp_path: Path):
        artifacts: dict = {
            "metrics": {"accuracy": 0.95},
        }
        save_pipeline_artifacts(artifacts, tmp_path)
        assert (tmp_path / "metrics.json").exists()

        loaded = load_pipeline_artifacts(tmp_path)
        assert "metrics" in loaded
        assert loaded["metrics"] == {"accuracy": 0.95}
