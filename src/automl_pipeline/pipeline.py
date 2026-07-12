"""Main pipeline orchestrator that ties all components together."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CV_FOLDS, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from .data import load_data, validate_data, detect_problem_type, split_data
from .models import get_models, train_model, evaluate_model
from .optimization import get_param_grid, optimize_grid, get_search_results
from .preprocessing import preprocess


class PipelineResult:
    def __init__(self) -> None:
        self.problem_type: str = ""
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.best_score: float = 0.0
        self.all_results: dict[str, dict[str, float]] = {}
        self.feature_importance: dict[str, float] = {}
        self.preprocessing_config: dict[str, Any] = {}
        self.training_time: float = 0.0
        self.optimization_results: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "best_model": self.best_model_name,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "feature_importance": self.feature_importance,
            "preprocessing": self.preprocessing_config,
            "training_time_seconds": round(self.training_time, 2),
        }


def run_pipeline(
    data_path: str | Path,
    target: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    optimize: bool = True,
    models: list[str] | None = None,
    preprocess_config: dict[str, Any] | None = None,
) -> PipelineResult:
    result = PipelineResult()
    start_time = time.time()

    df = load_data(data_path)
    errors = validate_data(df, target)
    if errors:
        raise ValueError(f"Data validation failed: {'; '.join(errors)}")

    result.problem_type = detect_problem_type(df[target])
    result.preprocessing_config = preprocess_config or {}

    X, y = preprocess(df, target, result.preprocessing_config)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    available_models = get_models(result.problem_type)
    if models:
        available_models = {k: v for k, v in available_models.items() if k in models}

    for model_name in available_models:
        try:
            model = train_model(model_name, X_train, y_train)
            eval_results = evaluate_model(model, X_test, y_test, result.problem_type)
            result.all_results[model_name] = eval_results

            primary_metric = list(eval_results.keys())[0]
            score = eval_results[primary_metric]

            if optimize:
                param_grid = get_param_grid(model_name)
                if param_grid:
                    try:
                        base_cls, defaults = available_models[model_name]
                        base = base_cls(**defaults)
                        search = optimize_grid(base, param_grid, X_train, y_train, cv=cv_folds, scoring=primary_metric)
                        opt_score = float(search.best_score_)
                        if opt_score > score:
                            model = search.best_estimator_
                            score = opt_score
                            eval_results.update(evaluate_model(model, X_test, y_test, result.problem_type))
                            result.optimization_results[model_name] = get_search_results(search)
                    except Exception:
                        pass

            if score > result.best_score:
                result.best_score = score
                result.best_model_name = model_name
                result.best_model = model
        except Exception as exc:
            result.all_results[model_name] = {"error": str(exc)}

    result.training_time = time.time() - start_time
    return result


def save_results(result: PipelineResult, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
