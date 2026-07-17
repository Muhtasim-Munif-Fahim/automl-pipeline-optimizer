"""Command-line interface for the AutoML pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import run_pipeline, save_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline Optimizer - automated ML pipeline optimization",
        epilog="Example: automl-pipeline --data data.csv --target price --optimize",
    )
    parser.add_argument(
        "--data", required=True, help="Path to dataset (CSV, Parquet, JSON, etc.)"
    )
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Enable hyperparameter optimization"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to try (default: all)"
    )
    parser.add_argument(
        "--output", default="pipeline_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--impute",
        default="auto",
        choices=["auto", "mean", "median"],
        help="Missing value strategy",
    )
    parser.add_argument(
        "--scaling",
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Scaling method",
    )
    parser.add_argument(
        "--encoding", default="auto", choices=["auto", "onehot"], help="Encoding method"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args, _ = parser.parse_known_args()

    if args.version:
        print(f"automl-pipeline-optimizer v{__version__}")
        return 0

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        return 1

    preprocess_config = {
        "impute_strategy": args.impute,
        "scaling_method": args.scaling,
        "encoding_method": args.encoding,
    }

    print("Running AutoML pipeline...")
    print(f"  Data: {args.data}")
    print(f"  Target: {args.target}")
    print(f"  Models: {args.models or 'all'}")

    try:
        result = run_pipeline(
            data_path=args.data,
            target=args.target,
            test_size=args.test_size,
            cv_folds=args.cv,
            optimize=args.optimize,
            models=args.models,
            preprocess_config=preprocess_config,
        )
    except Exception as exc:
        print(f"ERROR: Pipeline failed: {exc}")
        return 1

    save_results(result, args.output)

    print("\nResults:")
    print(f"  Best model: {result.best_model_name}")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Training time: {result.training_time:.1f}s")
    print(f"  Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
