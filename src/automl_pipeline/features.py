"""Feature engineering: polynomial, interaction, datetime, text features."""

from __future__ import annotations

import pandas as pd


def add_polynomial_features(
    df: pd.DataFrame, degree: int = 2, columns: list[str] | None = None
) -> pd.DataFrame:
    from sklearn.preprocessing import PolynomialFeatures

    cols = columns or df.select_dtypes(include=["int64", "float64"]).columns[:5]
    if len(cols) < 2:
        return df
    pf = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly = pf.fit_transform(df[cols])
    poly_df = pd.DataFrame(
        poly[:, len(cols) :],
        columns=[f"poly_{i}" for i in range(poly.shape[1] - len(cols))],
        index=df.index,
    )
    return pd.concat([df, poly_df], axis=1)


def add_interaction_features(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    from sklearn.preprocessing import PolynomialFeatures

    cols = columns or df.select_dtypes(include=["int64", "float64"]).columns[:5]
    if len(cols) < 2:
        return df
    pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    interactions = pf.fit_transform(df[cols])
    n_orig = len(cols)
    n_inter = interactions.shape[1] - n_orig
    if n_inter == 0:
        return df
    inter_df = pd.DataFrame(
        interactions[:, n_orig:],
        columns=[f"inter_{i}" for i in range(n_inter)],
        index=df.index,
    )
    return pd.concat([df, inter_df], axis=1)
