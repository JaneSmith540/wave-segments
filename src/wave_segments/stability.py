"""Resampling-based boundary stability and conservative segment merging.

These helpers evaluate the *segmentation* layer only.  They deliberately do
not consume future-return labels or classifier outputs when estimating a
boundary's stability.  A caller should run them inside each training/test time
window separately: resampling a full history cannot make a boundary causal.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import blake2b

import numpy as np
import pandas as pd

from .config import SegmentationConfig
from .schema import normalize_ohlcv
from .segmentation import segment_ohlcv, segment_ohlcv_causal


Segmenter = Callable[[pd.DataFrame], pd.DataFrame]


def perturb_ohlcv(
    ohlcv: pd.DataFrame,
    *,
    price_noise: float = 0.002,
    volume_noise: float = 0.08,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return an OHLCV-preserving multiplicative noise draw.

    The draw is independent per observed bar and keeps timestamps, symbols and
    each bar's high/low ordering intact.  It uses no labels or forward returns.
    ``price_noise`` and ``volume_noise`` are log standard deviations; zero
    gives an exact copy.  This is a local robustness test, not a generative
    market simulator.
    """
    if price_noise < 0 or volume_noise < 0:
        raise ValueError("noise scales must be non-negative")
    out = normalize_ohlcv(ohlcv).copy()
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    price_multiplier = np.exp(rng.normal(0.0, price_noise, len(out)))
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].to_numpy(float) * price_multiplier
    # Scaling every price in one bar preserves high >= open/close >= low.  The
    # source may have malformed values, so enforce the invariant defensively.
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = out["volume"].to_numpy(float) * np.exp(rng.normal(0.0, volume_noise, len(out)))
    return out


def boundary_table(segments: pd.DataFrame, *, include_edges: bool = False) -> pd.DataFrame:
    """Extract unique positional boundaries from a segment table.

    The output is keyed by ``symbol`` and the per-symbol ``boundary_idx`` used
    by :func:`segment_ohlcv`; this avoids accidental matching across symbols.
    """
    needed = {"symbol", "start_idx", "end_idx"}
    if missing := needed - set(segments.columns):
        raise ValueError(f"segments missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for symbol, group in segments.groupby("symbol", sort=False):
        starts = group[["start_idx"]].rename(columns={"start_idx": "boundary_idx"})
        ends = group[["end_idx"]].rename(columns={"end_idx": "boundary_idx"})
        values = sorted(set(starts.boundary_idx.astype(int)) | set(ends.boundary_idx.astype(int)))
        edge = {min(values), max(values)} if values else set()
        rows.extend({"symbol": symbol, "boundary_idx": value, "is_edge": value in edge} for value in values)
    out = pd.DataFrame(rows, columns=["symbol", "boundary_idx", "is_edge"])
    return out if include_edges else out.loc[~out.is_edge].reset_index(drop=True)


def _match_boundaries(reference: Sequence[int], observed: Sequence[int], tolerance: int) -> int:
    """Maximum cardinality monotone one-to-one tolerant matching."""
    i = j = matches = 0
    ref, obs = sorted(reference), sorted(observed)
    while i < len(ref) and j < len(obs):
        if obs[j] < ref[i] - tolerance:
            j += 1
        elif obs[j] > ref[i] + tolerance:
            i += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def boundary_precision_recall_f1(
    predicted: pd.DataFrame | Iterable[tuple[str, int]],
    truth: pd.DataFrame | Iterable[tuple[str, int]],
    *,
    tolerance: int = 3,
    include_edges: bool = False,
) -> dict[str, float | int]:
    """Compute one-to-one boundary precision/recall/F1 within bar tolerance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    def coerce(value: pd.DataFrame | Iterable[tuple[str, int]]) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            if {"start_idx", "end_idx"}.issubset(value.columns):
                return boundary_table(value, include_edges=include_edges)[["symbol", "boundary_idx"]]
            if not {"symbol", "boundary_idx"}.issubset(value.columns):
                raise ValueError("boundary data must have symbol/boundary_idx or segment bounds")
            return value[["symbol", "boundary_idx"]].copy()
        return pd.DataFrame(list(value), columns=["symbol", "boundary_idx"])

    pred, actual = coerce(predicted), coerce(truth)
    matches = 0
    symbols = set(pred.symbol) | set(actual.symbol)
    for symbol in symbols:
        matches += _match_boundaries(
            actual.loc[actual.symbol.eq(symbol), "boundary_idx"].astype(int).tolist(),
            pred.loc[pred.symbol.eq(symbol), "boundary_idx"].astype(int).tolist(), tolerance,
        )
    n_pred, n_truth = len(pred), len(actual)
    precision = matches / n_pred if n_pred else 0.0
    recall = matches / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"matches": matches, "predicted": n_pred, "truth": n_truth, "precision": precision, "recall": recall, "f1": f1, "tolerance": tolerance}


def bootstrap_boundary_stability(
    ohlcv: pd.DataFrame,
    config: SegmentationConfig | None = None,
    *,
    iterations: int = 100,
    tolerance: int = 3,
    price_noise: float = 0.002,
    volume_noise: float = 0.08,
    random_state: int | None = 42,
    use_changepoints: bool = False,
    segmenter: Segmenter | None = None,
    include_edges: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate recurrence frequency of each baseline boundary under perturbation.

    Returns ``(baseline_segments, stability_table)``.  A baseline boundary is
    counted once per run when any resampled boundary of the same symbol lies
    within ``tolerance``; repeated nearby candidates cannot inflate frequency.
    The optional ``segmenter`` is primarily useful for alternate algorithms and
    tests; it must return the standard segment bounds.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    cfg = config or SegmentationConfig()
    run = segmenter or (lambda frame: segment_ohlcv(frame, cfg, use_changepoints=use_changepoints))
    source = normalize_ohlcv(ohlcv)
    baseline = run(source)
    reference = boundary_table(baseline, include_edges=include_edges)
    hits = np.zeros(len(reference), dtype=int)
    rng = np.random.default_rng(random_state)
    for _ in range(iterations):
        sampled = run(perturb_ohlcv(source, price_noise=price_noise, volume_noise=volume_noise, random_state=rng))
        candidate = boundary_table(sampled, include_edges=include_edges)
        for pos, row in enumerate(reference.itertuples(index=False)):
            values = candidate.loc[candidate.symbol.eq(row.symbol), "boundary_idx"].to_numpy(int)
            hits[pos] += bool(np.any(np.abs(values - row.boundary_idx) <= tolerance))
    result = reference.copy()
    result["bootstrap_hits"] = hits
    result["bootstrap_iterations"] = iterations
    result["stability_frequency"] = hits / iterations
    result["tolerance"] = tolerance
    return baseline, result


def merge_consecutive_same_label(
    segments: pd.DataFrame,
    *,
    label_col: str = "label",
    probability_columns: Sequence[str] | None = None,
    merge_unknown: bool = False,
) -> pd.DataFrame:
    """Merge adjacent/overlapping same-label segments without losing provenance.

    Probability columns (``prob_*`` by default) are duration-weighted and then
    renormalised.  ``segment_ids`` stores every original id; callers can always
    recover the pre-merge rows.  UNKNOWN remains separate by default because it
    is an audit queue, not evidence of one homogeneous market state.
    """
    required = {"segment_id", "symbol", "start_idx", "end_idx", label_col}
    if missing := required - set(segments.columns):
        raise ValueError(f"segments missing columns: {sorted(missing)}")
    probs = list(probability_columns) if probability_columns is not None else [c for c in segments if c.startswith("prob_")]
    out: list[pd.Series] = []
    for symbol, group in segments.groupby("symbol", sort=False):
        ordered = group.sort_values(["start_idx", "end_idx", "segment_id"], kind="stable")
        runs: list[list[pd.Series]] = []
        for _, row in ordered.iterrows():
            label = row[label_col]
            can_merge = bool(runs) and label == runs[-1][-1][label_col] and int(row.start_idx) <= int(runs[-1][-1].end_idx) + 1
            if str(label).upper() == "UNKNOWN" and not merge_unknown:
                can_merge = False
            if can_merge:
                runs[-1].append(row)
            else:
                runs.append([row])
        for run_rows in runs:
            first, last = run_rows[0].copy(), run_rows[-1]
            ids = [str(x.segment_id) for x in run_rows]
            first["segment_ids"] = ids
            first["segment_id"] = ids[0] if len(ids) == 1 else f"{symbol}:merged:{ids[0]}:{ids[-1]}"
            first["end_idx"] = int(last.end_idx)
            for col in ("end", "end_timestamp", "end_boundary_probability", "end_boundary_sources",
                        "end_boundary_stability", "detector_end_boundary_probability"):
                if col in first.index:
                    first[col] = last[col]
            first["n_bars"] = int(first.end_idx) - int(first.start_idx) + 1
            weights = np.asarray([max(1, int(x.get("n_bars", x.end_idx - x.start_idx + 1))) for x in run_rows], dtype=float)
            if probs:
                values = np.asarray([[float(x.get(c, 0.0)) for c in probs] for x in run_rows])
                aggregate = np.average(values, axis=0, weights=weights)
                total = aggregate.sum()
                aggregate = aggregate / total if total > 0 else aggregate
                for col, value in zip(probs, aggregate): first[col] = value
                first["soft_label"] = {c.removeprefix("prob_"): float(v) for c, v in zip(probs, aggregate)}
            if ("detector_start_boundary_probability" in first.index
                    and "detector_end_boundary_probability" in first.index):
                first["detector_boundary_probability"] = min(
                    float(first.detector_start_boundary_probability),
                    float(first.detector_end_boundary_probability),
                )
            if "start_boundary_stability" in first.index and "end_boundary_stability" in first.index:
                first["bootstrap_boundary_stability"] = min(
                    float(first.start_boundary_stability), float(first.end_boundary_stability)
                )
            if "start_boundary_probability" in first.index and "end_boundary_probability" in first.index:
                boundary = min(float(first.start_boundary_probability), float(first.end_boundary_probability))
                first["boundary_probability"] = boundary
                first["boundary_uncertainty"] = 1.0 - boundary
            out.append(first)
    result = pd.DataFrame(out).reset_index(drop=True)
    if "segment_no" in result:
        result["segment_no"] = result.groupby("symbol", sort=False).cumcount()
    return result


def bootstrap_causal_boundary_stability(
    ohlcv: pd.DataFrame,
    config: SegmentationConfig | None = None,
    *,
    iterations: int = 100,
    tolerance: int = 3,
    price_noise: float = 0.002,
    volume_noise: float = 0.08,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recheck each confirmed causal pivot using perturbations ending at its confirmation bar.

    A perturbed full path is segmented once per symbol/iteration, then candidate
    pivots are filtered by their own confirmation indices for each baseline
    boundary. This is equivalent to prefix-only evaluation because the causal
    segmenter is prefix-invariant, and later bars cannot influence an earlier
    score. Frequency measures perturbation recurrence, not human correctness.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    cfg = config or SegmentationConfig()
    source = normalize_ohlcv(ohlcv)
    baseline = segment_ohlcv_causal(source, cfg)
    groups = {symbol: group.reset_index(drop=True) for symbol, group in source.groupby("symbol", sort=False)}
    boundaries: dict[tuple[object, int], tuple[int, pd.Timestamp]] = {}
    for row in baseline.itertuples(index=False):
        group = groups[row.symbol]
        end_idx, confirm_idx = int(row.end_idx), int(row.confirmation_idx)
        boundaries[(row.symbol, end_idx)] = (confirm_idx, pd.Timestamp(row.available_at))
        start_idx = int(row.start_idx)
        if start_idx == 0:
            boundaries[(row.symbol, start_idx)] = (0, pd.Timestamp(group.loc[0, "timestamp"]))
        else:
            start_time = pd.Timestamp(row.start_available_at)
            matches = np.flatnonzero(group["timestamp"].eq(start_time).to_numpy())
            if len(matches):
                boundaries[(row.symbol, start_idx)] = (int(matches[0]), start_time)

    hits_by_boundary = {key: 0 for key in boundaries}
    by_symbol: dict[object, list[tuple[tuple[object, int], int]]] = {}
    for key, (confirmation_idx, _) in boundaries.items():
        by_symbol.setdefault(key[0], []).append((key, confirmation_idx))
        if key[1] == 0:
            hits_by_boundary[key] = iterations
    for symbol, symbol_boundaries in by_symbol.items():
        if all(key[1] == 0 for key, _ in symbol_boundaries):
            continue
        group = groups[symbol]
        for iteration in range(iterations):
            # Symbol/iteration seed makes prefix scores invariant to unrelated
            # symbols and to appended future bars in the same symbol.
            seed_text = f"{random_state}:{symbol}:{iteration}"
            seed = int.from_bytes(blake2b(seed_text.encode("utf-8"), digest_size=4).digest(), "little")
            perturbed = perturb_ohlcv(
                group, price_noise=price_noise, volume_noise=volume_noise,
                random_state=np.random.default_rng(seed),
            )
            candidate = segment_ohlcv_causal(perturbed, cfg)
            for key, confirmation_idx in symbol_boundaries:
                if key[1] == 0:
                    continue
                available_candidates = candidate.loc[candidate["confirmation_idx"] <= confirmation_idx]
                indices = available_candidates["end_idx"].to_numpy(int)
                hits_by_boundary[key] += int(np.any(np.abs(indices - key[1]) <= tolerance))

    records = []
    for (symbol, boundary_idx), (confirmation_idx, available_at) in sorted(
        boundaries.items(), key=lambda item: (str(item[0][0]), item[1][0], item[0][1]),
    ):
        hits = hits_by_boundary[(symbol, boundary_idx)]
        records.append({
            "symbol": symbol, "boundary_idx": boundary_idx,
            "confirmation_idx": confirmation_idx, "available_at": available_at,
            "bootstrap_hits": hits, "bootstrap_iterations": iterations,
            "stability_frequency": hits / iterations, "tolerance": tolerance,
        })
    return baseline, pd.DataFrame(records)
