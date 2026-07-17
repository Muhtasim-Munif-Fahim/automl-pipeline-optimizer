"""HTML/JSON report generation for pipeline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(result_dict: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")


def generate_summary(result_dict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("AutoML Pipeline Results Summary")
    lines.append("=" * 50)
    lines.append(f"Problem type: {result_dict.get('problem_type', 'unknown')}")
    lines.append(f"Best model: {result_dict.get('best_model', 'N/A')}")
    lines.append(f"Best score: {result_dict.get('best_score', 0):.4f}")
    lines.append(f"Training time: {result_dict.get('training_time_seconds', 0):.1f}s")
    lines.append("")
    lines.append("All model results:")
    for model, metrics in result_dict.get("all_results", {}).items():
        scores = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        lines.append(f"  {model}: {scores}")
    lines.append("")
    lines.append(f"Results saved to: {result_dict.get('output_path', 'N/A')}")
    return "\n".join(lines)


def compare_results(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Model Comparison:")
    lines.append("-" * 60)
    lines.append(f"{'Model':<25} {'Score':<10} {'Time(s)':<10}")
    lines.append("-" * 60)
    for res in results:
        model_name = res.get('best_model', 'N/A')
        score = res.get('best_score', 0)
        elapsed = res.get('training_time_seconds', 0)
        lines.append(f"{model_name:<25} {score:<10.4f} {elapsed:<10.1f}")
    return "\n".join(lines)
