"""Imbalanced data handling: SMOTE, class weights, resampling."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def compute_class_weights(y: pd.Series) -> dict:
    classes = y.unique()
    n_samples = len(y)
    weights = {c: n_samples / (len(classes) * (y == c).sum()) for c in classes}
    return weights


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)
    except ImportError:
        return X, y


def downsample_majority(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    df = X.copy()
    df["_target"] = y
    classes = df["_target"].value_counts()
    min_count = classes.min()
    sampled = df.groupby("_target").sample(n=min_count, random_state=random_state)
    return sampled.drop(columns=["_target"]), sampled["_target"]
