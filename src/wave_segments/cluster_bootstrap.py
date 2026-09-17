"""Cluster-level training resampling and fold-local cluster alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .discovery import MultiModelDiscoverer


def resample_symbols_with_replacement(training: pd.DataFrame, *, random_state: int) -> tuple[pd.DataFrame, int]:
    """Sample symbol clusters with replacement, retaining every row per draw.

    Rows from one security are not treated as independent observations. The
    repeated rows affect fitting weight, while unique-symbol support gates still
    count actual distinct symbols rather than bootstrap copies.
    """
    if "symbol" not in training:
        raise ValueError("training data must contain symbol")
    symbols = training.symbol.dropna().drop_duplicates().to_numpy()
    if len(symbols) < 2:
        raise ValueError("at least two unique symbols are required for cluster bootstrap")
    rng = np.random.default_rng(random_state)
    draws = rng.choice(symbols, size=len(symbols), replace=True)
    parts = [training.loc[training.symbol.eq(symbol)] for symbol in draws]
    sample = pd.concat(parts, ignore_index=True)
    return sample, int(pd.Series(draws).nunique())


def align_bootstrap_to_reference(
    reference: MultiModelDiscoverer,
    bootstrap: MultiModelDiscoverer,
) -> tuple[dict[str, str], float]:
    """Align arbitrary bootstrap component IDs to a fold's reference prototypes.

    Centers are inverse-transformed to raw units, then distances are scaled by
    the *reference fold's* RobustScaler scale. Returns bootstrap-label ->
    reference-label mapping and normalized matched-center RMS distance.
    """
    if reference.reference_means_ is None or bootstrap.reference_means_ is None:
        raise ValueError("both discoverers must be fitted")
    if len(reference.cluster_labels_) != len(bootstrap.cluster_labels_):
        raise ValueError("reference/bootstrap cluster counts differ; alignment is not identifiable")
    ref_centers = reference.scaler.inverse_transform(reference.reference_means_)
    boot_centers = bootstrap.scaler.inverse_transform(bootstrap.reference_means_)
    scale = np.asarray(reference.scaler.scale_, float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    cost = (((boot_centers[:, None, :] - ref_centers[None, :, :]) / scale[None, None, :]) ** 2).mean(axis=2)
    boot_index, ref_index = linear_sum_assignment(cost)
    mapping = {
        bootstrap.cluster_labels_[int(b)]: reference.cluster_labels_[int(r)]
        for b, r in zip(boot_index, ref_index)
    }
    matched = cost[boot_index, ref_index]
    rms = float(np.sqrt(np.mean(matched))) if len(matched) else float("nan")
    return mapping, rms


def align_by_probability_overlap(
    reference_probabilities: np.ndarray,
    bootstrap_probabilities: np.ndarray,
    reference_labels: list[str],
    bootstrap_labels: list[str],
) -> tuple[dict[str, str], float]:
    """Align fold-local clusters by soft-membership overlap on common rows.

    Cosine-normalized cross-membership makes the assignment less sensitive to
    cluster prevalence than raw counts. The same fixed training sample must be
    used for both matrices; evaluation rows must not determine the permutation.
    """
    ref = np.asarray(reference_probabilities, float)
    boot = np.asarray(bootstrap_probabilities, float)
    if ref.ndim != 2 or boot.ndim != 2 or ref.shape[0] != boot.shape[0]:
        raise ValueError("probability matrices must be 2-D with the same number of rows")
    if ref.shape[1] != len(reference_labels) or boot.shape[1] != len(bootstrap_labels):
        raise ValueError("probability matrix dimensions do not match label lists")
    if ref.shape[1] != boot.shape[1]:
        raise ValueError("cluster counts differ; probability-overlap alignment is not identifiable")
    valid = np.isfinite(ref).all(axis=1) & np.isfinite(boot).all(axis=1)
    if not valid.any():
        raise ValueError("no finite common probability rows for cluster alignment")
    ref, boot = ref[valid], boot[valid]
    cross = ref.T @ boot
    ref_norm = np.sqrt((ref * ref).sum(axis=0))
    boot_norm = np.sqrt((boot * boot).sum(axis=0))
    similarity = cross / np.clip(np.outer(ref_norm, boot_norm), 1e-12, None)
    ref_index, boot_index = linear_sum_assignment(-similarity)
    mapping = {
        bootstrap_labels[int(b)]: reference_labels[int(r)]
        for r, b in zip(ref_index, boot_index)
    }
    matched = similarity[ref_index, boot_index]
    return mapping, float(np.mean(matched)) if len(matched) else float("nan")


def remap_bootstrap_predictions(
    predictions: pd.DataFrame,
    mapping: dict[str, str],
    reference_labels: list[str],
) -> pd.DataFrame:
    """Rename mapped labels and reorder soft probabilities to the reference IDs."""
    result = predictions.copy()
    remapped_candidate = result.candidate_label.astype(str).map(mapping)
    result["candidate_label"] = remapped_candidate.fillna(result.candidate_label)
    known = result.label.astype(str).ne("UNKNOWN")
    remapped_label = result.label.astype(str).map(mapping)
    result.loc[known, "label"] = remapped_label.loc[known].fillna(result.loc[known, "label"])
    local_probability = {
        column.removeprefix("prob_"): column
        for column in predictions.columns if column.startswith("prob_")
    }
    aligned = np.zeros((len(result), len(reference_labels)), dtype=float)
    for local_label, column in local_probability.items():
        reference_label = mapping.get(local_label)
        if reference_label in reference_labels:
            aligned[:, reference_labels.index(reference_label)] = predictions[column].to_numpy(float)
    for index, label in enumerate(reference_labels):
        result[f"prob_{label}"] = aligned[:, index]
    return result
