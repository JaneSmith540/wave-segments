"""Conditional feature-noise robustness checks for frozen cluster classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def perturb_features_from_train_scale(
    scored: pd.DataFrame,
    training: pd.DataFrame,
    feature_columns: list[str],
    *,
    noise_fraction: float,
    random_state: int,
    fixed_columns: tuple[str, ...] = ("duration_bars", "boundary_uncertainty"),
) -> pd.DataFrame:
    """Add seeded Gaussian noise using robust scales estimated on train only.

    This isolates classifier sensitivity conditional on fixed boundaries and a
    frozen fitted model. It does not simulate market paths or refit uncertainty.
    """
    if noise_fraction < 0:
        raise ValueError("noise_fraction must be non-negative")
    missing = set(feature_columns) - set(training.columns) | (set(feature_columns) - set(scored.columns))
    if missing:
        raise ValueError(f"missing perturbation features: {sorted(missing)}")
    result = scored.copy()
    rng = np.random.default_rng(random_state)
    for column in feature_columns:
        if column in fixed_columns:
            continue
        train_values = pd.to_numeric(training[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        center = train_values.median()
        robust_scale = float((train_values - center).abs().median() * 1.4826) if pd.notna(center) else 0.0
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            robust_scale = float(train_values.std())
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            continue
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        values[valid] += rng.normal(0.0, robust_scale * noise_fraction, valid.sum())
        result[column] = values
    return result


def probability_total_variation(reference: pd.DataFrame, observed: pd.DataFrame,
                                probability_columns: list[str]) -> np.ndarray:
    """Per-row total variation distance between two categorical distributions."""
    left = reference[probability_columns].to_numpy(float)
    right = observed[probability_columns].to_numpy(float)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    result = np.full(len(reference), np.nan)
    if valid.any():
        left_valid = left[valid] / np.clip(left[valid].sum(axis=1, keepdims=True), 1e-12, None)
        right_valid = right[valid] / np.clip(right[valid].sum(axis=1, keepdims=True), 1e-12, None)
        result[valid] = 0.5 * np.abs(left_valid - right_valid).sum(axis=1)
    return result
