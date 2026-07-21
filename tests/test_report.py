"""Tests for report generation module (0% coverage gap)."""

from __future__ import annotations

from automl_pipeline.report import generate_summary, compare_results


class TestReport:
    def test_generate_summary_contains_keys(self):
        result = {"best_model": "rf", "best_score": 0.95, "problem_type": "classification", "training_time_seconds": 10.5, "all_results": {"rf": {"accuracy": 0.95}}, "output_path": "out.json"}
        text = generate_summary(result)
        assert "rf" in text
        assert "0.95" in text

    def test_compare_results_format(self):
        r1 = {"best_model": "lr", "best_score": 0.9, "training_time_seconds": 5.0}
        r2 = {"best_model": "rf", "best_score": 0.95, "training_time_seconds": 10.0}
        text = compare_results([r1, r2])
        assert "lr" in text
        assert "rf" in text
