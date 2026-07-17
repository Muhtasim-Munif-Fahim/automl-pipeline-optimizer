"""Configuration constants for the AutoML pipeline."""

from __future__ import annotations

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_VALIDATION_SIZE: float = 0.1
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_CV_FOLDS: int = 5
DEFAULT_N_ITER_OPTUNA: int = 50
DEFAULT_TIMEOUT_SECONDS: int = 300

RANDOM_STATE: int = 42
N_JOBS: int = -1

MODEL_TYPES: dict[str, list[str]] = {
    "classification": [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "svm",
    ],
    "regression": [
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
    ],
}

SCORING_METRICS: dict[str, list[str]] = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "regression": [
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "explained_variance",
    ],
}

PREPROCESSING_STEPS: list[str] = [
    "impute_missing",
    "encode_categorical",
    "scale_numeric",
    "handle_outliers",
    "feature_selection",
]
