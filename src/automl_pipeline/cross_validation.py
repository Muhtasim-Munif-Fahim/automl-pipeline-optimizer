"""Cross-validation utilities for the AutoML pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit


def get_cv_splits(X, y=None, n_folds: int = 5, strategy: str = "stratified", shuffle: bool = True, random_state: int = 42):
    if strategy == "stratified" and y is not None and y.dtype.kind in ("i", "b"):
        return StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    return KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)


def cross_validate(model, X, y, cv=5, scoring=None, problem_type: str = "regression"):
    from sklearn.model_selection import cross_val_score, cross_val_predict
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()), "predictions": preds.tolist()}
