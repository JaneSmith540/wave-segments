"""Parameter-sensitivity diagnostics for causal variable-length segmentation."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def causal_boundaries(segments: pd.DataFrame) -> pd.DataFrame:
    """Return confirmed pivots, excluding only the first observed data boundary.

    The latest candidate endpoint is a real confirmed pivot in the causal
    segmenter: it is not the last input bar (the open tail is omitted).
    """
    required = {"symbol", "start_idx", "end_idx", "start", "end"}
    if missing := required - set(segments):
        raise ValueError(f"segments missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for symbol, group in segments.groupby("symbol", sort=False):
        endpoints = sorted(set(group.start_idx.astype(int)) | set(group.end_idx.astype(int)))
        if len(endpoints) <= 1:
            continue
        pivots = set(endpoints[1:])
        # Each endpoint is stored at both sides of a shared turning bar. Use
        # its calendar timestamp from the right-hand boundary of a segment.
        end_dates = group[["end_idx", "end"]].rename(columns={"end_idx": "idx", "end": "date"})
        start_dates = group[["start_idx", "start"]].rename(columns={"start_idx": "idx", "start": "date"})
        dates = pd.concat([start_dates, end_dates], ignore_index=True).drop_duplicates("idx").set_index("idx")["date"]
        for idx in sorted(pivots):
            if idx in dates.index:
                timestamp = pd.Timestamp(dates.loc[idx])
                records.append({"symbol": symbol, "boundary_idx": idx,
                                "boundary_date": timestamp, "year": timestamp.year})
    return pd.DataFrame(records, columns=["symbol", "boundary_idx", "boundary_date", "year"])


def _matched_pairs(reference: Iterable[int], observed: Iterable[int], tolerance: int) -> list[tuple[int, int]]:
    """Maximum-cardinality monotone matching under absolute index tolerance."""
    ref, obs = sorted(map(int, reference)), sorted(map(int, observed))
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(ref) and j < len(obs):
        if obs[j] < ref[i] - tolerance:
            j += 1
        elif obs[j] > ref[i] + tolerance:
            i += 1
        else:
            pairs.append((ref[i], obs[j]))
            i += 1
            j += 1
    return pairs


def compare_causal_boundaries(
    reference_segments: pd.DataFrame,
    observed_segments: pd.DataFrame,
    *,
    tolerance: int = 3,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Compare segmenter outputs as agreement, never as true boundary accuracy.

    Returns an overall agreement metric, per-symbol metrics, and per-year
    metrics. The reference is a chosen parameterization, not ground truth.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ref, obs = causal_boundaries(reference_segments), causal_boundaries(observed_segments)
    symbol_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    overall_matches = 0
    symbols = sorted(set(ref.symbol) | set(obs.symbol))
    for symbol in symbols:
        left = ref.loc[ref.symbol.eq(symbol)]
        right = obs.loc[obs.symbol.eq(symbol)]
        pairs = _matched_pairs(left.boundary_idx, right.boundary_idx, tolerance)
        overall_matches += len(pairs)
        p = len(pairs) / len(right) if len(right) else 0.0
        r = len(pairs) / len(left) if len(left) else 0.0
        symbol_rows.append({"symbol": symbol, "reference_boundaries": len(left),
                            "observed_boundaries": len(right), "matches": len(pairs),
                            "precision_vs_reference": p, "recall_vs_reference": r,
                            "f1_agreement": 2 * p * r / (p + r) if p + r else 0.0})
        # Year slices use boundary year on each side. A cross-year match is
        # counted only in its respective year's denominator, not as a match.
        years = sorted(set(left.year) | set(right.year))
        for year in years:
            ly, ry = left.loc[left.year.eq(year)], right.loc[right.year.eq(year)]
            year_pairs = _matched_pairs(ly.boundary_idx, ry.boundary_idx, tolerance)
            yp = len(year_pairs) / len(ry) if len(ry) else 0.0
            yr = len(year_pairs) / len(ly) if len(ly) else 0.0
            year_rows.append({"symbol": symbol, "year": int(year),
                              "reference_boundaries": len(ly), "observed_boundaries": len(ry),
                              "matches": len(year_pairs), "precision_vs_reference": yp,
                              "recall_vs_reference": yr,
                              "f1_agreement": 2 * yp * yr / (yp + yr) if yp + yr else 0.0})
    n_ref, n_obs = len(ref), len(obs)
    precision = overall_matches / n_obs if n_obs else 0.0
    recall = overall_matches / n_ref if n_ref else 0.0
    overall = {"matches": overall_matches, "reference_boundaries": n_ref,
               "observed_boundaries": n_obs, "precision_vs_reference": precision,
               "recall_vs_reference": recall,
               "f1_agreement": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
               "tolerance_bars": tolerance}
    return overall, pd.DataFrame(symbol_rows), pd.DataFrame(year_rows)


def segment_distribution(segments: pd.DataFrame, *, short_threshold: int = 5) -> dict[str, float | int]:
    """Summarize lengths without treating any configuration as correct."""
    if "n_bars" not in segments:
        raise ValueError("segments missing n_bars")
    lengths = pd.to_numeric(segments.n_bars, errors="coerce").dropna()
    return {
        "segments": len(lengths),
        "median_bars": float(lengths.median()) if len(lengths) else float("nan"),
        "p10_bars": float(lengths.quantile(.1)) if len(lengths) else float("nan"),
        "p90_bars": float(lengths.quantile(.9)) if len(lengths) else float("nan"),
        "short_segment_fraction": float((lengths < short_threshold).mean()) if len(lengths) else float("nan"),
    }
