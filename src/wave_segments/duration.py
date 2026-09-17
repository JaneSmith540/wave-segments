"""Duration-aware state smoothing for completed segments.

Offline HSMM decoding uses the complete sequence and is therefore for historical
description/audit only.  Use :func:`causal_duration_filter` when producing a
point-in-time state stream for the isolated selection layer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd

from .model import UNKNOWN
from .stability import merge_consecutive_same_label


def _probability_columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    selected = list(columns) if columns is not None else [c for c in frame if c.startswith("prob_")]
    if not selected:
        raise ValueError("No canonical prob_* emission columns found")
    if missing := set(selected) - set(frame.columns):
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    return selected


def geometric_duration_pmf(
    labels: Sequence[str], max_duration: int = 8, min_duration: int | Mapping[str, int] = 2,
    continuation: float = .65,
) -> dict[str, np.ndarray]:
    """Create explicit-duration priors indexed from 0..max_duration."""
    if max_duration < 1 or not 0 < continuation < 1:
        raise ValueError("Invalid duration parameters")
    output = {}
    for label in labels:
        minimum = int(min_duration.get(label, 1)) if isinstance(min_duration, Mapping) else int(min_duration)
        values = np.zeros(max_duration + 1)
        for duration in range(max(1, minimum), max_duration + 1):
            values[duration] = (1 - continuation) * continuation ** (duration - max(1, minimum))
        if values.sum() == 0:
            values[max_duration] = 1
        output[str(label)] = values / values.sum()
    return output


def _transition_matrix(labels: list[str], transition: pd.DataFrame | np.ndarray | None) -> np.ndarray:
    k = len(labels)
    if transition is None:
        if k == 1:
            return np.ones((1, 1))
        matrix = np.ones((k, k)) - np.eye(k)
        return matrix / matrix.sum(axis=1, keepdims=True)
    if isinstance(transition, pd.DataFrame):
        matrix = transition.reindex(index=labels, columns=labels, fill_value=0).to_numpy(float)
    else:
        matrix = np.asarray(transition, float)
    if matrix.shape != (k, k):
        raise ValueError(f"transition matrix must have shape {(k, k)}")
    matrix = np.clip(matrix, 0, None)
    return matrix / np.where(matrix.sum(axis=1, keepdims=True) > 0, matrix.sum(axis=1, keepdims=True), 1)


def _decode_block(emission: np.ndarray, labels: list[str], transition: np.ndarray,
                  duration_pmf: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
    n, k = emission.shape
    log_emission = np.log(np.clip(emission, 1e-12, 1))
    cumulative = np.vstack([np.zeros(k), np.cumsum(log_emission, axis=0)])
    score = np.full((n, k), -np.inf)
    previous = np.full((n, k), -1, dtype=int)
    chosen_duration = np.ones((n, k), dtype=int)
    log_transition = np.log(np.clip(transition, 1e-12, 1))
    for end in range(1, n + 1):
        for state, label in enumerate(labels):
            prior = np.asarray(duration_pmf[label], float)
            for duration in range(1, min(end, len(prior) - 1) + 1):
                if prior[duration] <= 0:
                    continue
                start = end - duration
                emission_score = cumulative[end, state] - cumulative[start, state]
                if start == 0:
                    candidate, prior_state = -np.log(k), -1
                else:
                    candidates = score[start - 1] + log_transition[:, state]
                    candidates[state] = -np.inf  # explicit durations, no self-transition between runs
                    prior_state = int(np.argmax(candidates))
                    candidate = candidates[prior_state]
                candidate += emission_score + np.log(prior[duration])
                if candidate > score[end - 1, state]:
                    score[end - 1, state] = candidate
                    previous[end - 1, state] = prior_state
                    chosen_duration[end - 1, state] = duration
    state = int(np.argmax(score[-1])); total_score = float(score[-1, state])
    if not np.isfinite(total_score):
        path = emission.argmax(axis=1)
        return path, float(np.log(np.clip(emission.max(axis=1), 1e-12, 1)).sum())
    path = np.empty(n, dtype=int); end = n
    while end > 0:
        duration = int(chosen_duration[end - 1, state])
        path[end - duration:end] = state
        state = int(previous[end - 1, state])
        end -= duration
        if end > 0 and state < 0:
            raise RuntimeError("HSMM backtracking failed")
    return path, total_score


def decode_hsmm(
    segments: pd.DataFrame, *, probability_columns: Sequence[str] | None = None,
    transition: pd.DataFrame | np.ndarray | None = None,
    duration_pmf: Mapping[str, np.ndarray] | None = None,
    max_duration: int = 8, min_duration: int | Mapping[str, int] = 2,
    label_col: str = "label",
) -> pd.DataFrame:
    """Offline explicit-duration Viterbi, independently for every symbol.

    Existing UNKNOWN rows split the sequence and remain UNKNOWN.  Because the
    decoder observes later completed segments, ``hsmm_label`` must not be used as
    a historical point-in-time feature.
    """
    required = {"symbol", "start_idx", "end_idx"}
    if missing := required - set(segments):
        raise ValueError(f"segments missing columns: {sorted(missing)}")
    probs = _probability_columns(segments, probability_columns)
    labels = [c.removeprefix("prob_") for c in probs]
    trans = _transition_matrix(labels, transition)
    durations = dict(duration_pmf or geometric_duration_pmf(labels, max_duration, min_duration))
    if set(durations) != set(labels):
        raise ValueError("duration_pmf must provide every emission label")
    out = segments.copy()
    out["hsmm_label"] = UNKNOWN
    out["hsmm_path_score"] = np.nan
    for _, group in out.groupby("symbol", sort=False):
        ordered = group.sort_values(["start_idx", "end_idx"])
        is_unknown = ordered[label_col].astype(str).eq(UNKNOWN) if label_col in ordered else pd.Series(False, index=ordered.index)
        block_id = is_unknown.cumsum() + is_unknown.shift(fill_value=False).cumsum()
        for _, block in ordered.loc[~is_unknown].groupby(block_id[~is_unknown]):
            emission = block[probs].to_numpy(float)
            emission /= np.clip(emission.sum(axis=1, keepdims=True), 1e-12, None)
            path, score = _decode_block(emission, labels, trans, durations)
            out.loc[block.index, "hsmm_label"] = np.asarray(labels, object)[path]
            out.loc[block.index, "hsmm_path_score"] = score / len(block)
    original = out[label_col].astype(str) if label_col in out else out[probs].idxmax(axis=1).str.removeprefix("prob_")
    out["hsmm_changed"] = out["hsmm_label"].ne(original)
    return out


def causal_duration_filter(
    segments: pd.DataFrame, *, probability_columns: Sequence[str] | None = None,
    min_dwell_bars: int | Mapping[str, int] = 5,
) -> pd.DataFrame:
    """Past-only dwell filter suitable for constructing point-in-time states."""
    probs = _probability_columns(segments, probability_columns)
    labels = np.asarray([c.removeprefix("prob_") for c in probs], object)
    out = segments.copy(); out["causal_duration_label"] = UNKNOWN
    for _, group in out.groupby("symbol", sort=False):
        current: str | None = None; dwell = 0
        for index, row in group.sort_values(["start_idx", "end_idx"]).iterrows():
            if str(row.get("label", "")) == UNKNOWN:
                out.at[index, "causal_duration_label"] = UNKNOWN; current = None; dwell = 0; continue
            candidate = str(labels[int(np.argmax(row[probs].to_numpy(float)))])
            minimum = int(min_dwell_bars.get(current, 1)) if isinstance(min_dwell_bars, Mapping) and current else int(min_dwell_bars)
            if current is None or candidate == current or dwell >= minimum:
                if candidate != current: current, dwell = candidate, 0
            out.at[index, "causal_duration_label"] = current
            dwell += int(row.get("n_bars", int(row.end_idx) - int(row.start_idx) + 1))
    return out


def hsmm_smooth_and_merge(segments: pd.DataFrame, **kwargs) -> pd.DataFrame:
    decoded = decode_hsmm(segments, **kwargs).copy()
    decoded["label"] = decoded["hsmm_label"]
    return merge_consecutive_same_label(decoded)
