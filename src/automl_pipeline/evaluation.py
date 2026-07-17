"""Model evaluation: calibration curves, confusion matrix, learning curves."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix as cm

    matrix = cm(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "matrix": matrix.tolist(),
        "labels": labels,
        "accuracy": float((y_true == y_pred).mean()),
    }


def classification_report_data(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report


def calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}


def learning_curve_data(
    model, X_train, y_train, cv: int = 5, train_sizes: np.ndarray | None = None
) -> dict[str, Any]:
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes, n_jobs=-1
    )
    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "test_scores_mean": test_scores.mean(axis=1).tolist(),
    }
