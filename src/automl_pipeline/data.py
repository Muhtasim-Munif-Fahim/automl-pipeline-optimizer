"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_FORMATS: dict[str, Any] = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".xls": pd.read_excel,
    ".xlsx": pd.read_excel,
    ".feather": pd.read_feather,
}


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {path}")

    suffix = path.suffix.lower()
    reader = SUPPORTED_FORMATS.get(suffix)
    if reader is None:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    try:
        return reader(path)
    except Exception as exc:
        raise ValueError(f"Error reading {path}: {exc}") from exc


def validate_data(df: pd.DataFrame, target: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        errors.append("DataFrame is empty")
    if target not in df.columns:
        errors.append(f"Target column '{target}' not found in data")
    if len(df) < 10:
        errors.append(f"Dataset too small: {len(df)} rows (minimum 10)")
    if df.isna().all().all():
        errors.append("All values are missing")
    return errors


def detect_problem_type(y: pd.Series) -> str:
    unique = y.nunique()
    if y.dtype.kind in ("i", "b") or y.dtype.kind == "O" and unique <= 20:
        if unique == 2:
            return "binary_classification"
        if unique < 20:
            return "multiclass_classification"
    return "regression"


def split_data(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
