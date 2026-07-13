"""Integration tests for the full pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from automl_pipeline.pipeline import run_pipeline, save_results
from automl_pipeline.data import load_data


class TestPipeline:
    def _make_data(self, tmp_path: Path, rows: int = 20) -> Path:
        data = tmp_path / "data.csv"
        lines = ["x,y,label"]
        for i in range(rows):
            label = "A" if i < rows // 2 else "B"
            lines.append(f"{i},{i+1},{label}")
        data.write_text("\n".join(lines))
        return data

    def test_end_to_end_classification(self, tmp_path: Path):
        data = self._make_data(tmp_path)
        result = run_pipeline(str(data), "label", test_size=0.3, optimize=False, models=["logistic_regression"])
        assert result.best_model_name == "logistic_regression"
        assert len(result.all_results) >= 1

    def test_pipeline_with_optimization(self, tmp_path: Path):
        data = self._make_data(tmp_path)
        result = run_pipeline(str(data), "label", test_size=0.3, optimize=True, models=["logistic_regression"])
        assert result.best_model is not None
        assert result.training_time > 0

    def test_save_results(self, tmp_path: Path):
        data = tmp_path / "data.csv"
        data.write_text("x,y,label\n" + "\n".join(f"{i},{i+1},{'A' if i<7 else 'B'}" for i in range(15)))
        result = run_pipeline(str(data), "label", test_size=0.3, optimize=False, models=["logistic_regression"])
        out = tmp_path / "results.json"
        save_results(result, out)
        assert out.exists()
        import json
        loaded = json.loads(out.read_text())
        assert "best_model" in loaded
