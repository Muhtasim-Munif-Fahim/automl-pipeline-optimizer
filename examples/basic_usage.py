#!/usr/bin/env python3
"""Example: AutoML pipeline on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
from automl_pipeline.pipeline import run_pipeline, save_results


def create_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 2 + 1,
        "feature_c": np.random.randn(n) * 0.5 - 1,
        "category": np.random.choice(["X", "Y", "Z"], n),
    })
    df["target"] = (df["feature_a"] * 2 + df["feature_b"] * 0.5 + np.random.randn(n) * 0.1 > 0).astype(int)
    return df


def main() -> int:
    data_path = Path("sample_data.csv")
    create_sample_data().to_csv(data_path, index=False)

    print("Running AutoML pipeline on sample data...")
    result = run_pipeline(
        data_path=str(data_path),
        target="target",
        test_size=0.2,
        optimize=True,
        models=["logistic_regression", "random_forest"],
    )

    output = Path("pipeline_results.json")
    save_results(result, output)

    print(f"Best model: {result.best_model_name}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"All results: {result.all_results}")
    print(f"Results saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
