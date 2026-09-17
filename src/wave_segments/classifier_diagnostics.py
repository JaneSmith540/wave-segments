"""Descriptive, label-free structural diagnostics for wave classifications.

These diagnostics deliberately do not consume forward returns or claim semantic
accuracy.  Temporary cluster IDs are only comparable inside one fitted model
version, so every sequence is keyed by both symbol and model version.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .duration import decode_hsmm


def _sequence_key(frame: pd.DataFrame) -> pd.Series:
    version = frame.get("oos_model_trained_at", pd.Series("NO_MODEL", index=frame.index))
    version = pd.to_datetime(version, errors="coerce").astype("string").fillna("NO_MODEL")
    return frame["symbol"].astype("string") + "|" + version


def classification_structure_summary(
    states: pd.DataFrame,
    *,
    probability_columns: Sequence[str] | None = None,
    min_duration_segments: int = 2,
    max_duration_segments: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-version metrics, raw transitions, and HSMM sensitivity metrics.

    The HSMM is an offline descriptive smoother; it is not point-in-time and
    must not be used as a feature in the independent selection layer.
    """
    required = {"symbol", "start_idx", "end_idx", "n_bars", "label"}
    if missing := required - set(states.columns):
        raise ValueError(f"states missing columns: {sorted(missing)}")
    probs = list(probability_columns or [c for c in states if c.startswith("prob_")])
    if not probs:
        raise ValueError("No prob_* columns found")
    if missing := set(probs) - set(states.columns):
        raise ValueError(f"missing probability columns: {sorted(missing)}")

    frame = states.copy()
    frame["_model_version"] = pd.to_datetime(
        frame.get("oos_model_trained_at", pd.Series(pd.NaT, index=frame.index)), errors="coerce"
    ).astype("string").fillna("NO_MODEL")
    frame["_sequence"] = frame["symbol"].astype("string") + "|" + frame["_model_version"]
    frame["_version_order"] = frame["_model_version"].where(frame["_model_version"].ne("NO_MODEL"), "")
    frame = frame.sort_values(["_version_order", "symbol", "start_idx", "end_idx"], kind="stable")
    frame["_identified"] = frame["label"].astype(str).ne("UNKNOWN")
    frame["_duration"] = pd.to_numeric(frame["n_bars"], errors="coerce")

    versions = []
    transitions = []
    for version, group in frame.groupby("_model_version", sort=False):
        known = group.loc[group["_identified"]]
        raw_pairs = []
        runs = 0
        run_bars = []
        for _, seq in group.groupby("_sequence", sort=False):
            seq = seq.sort_values(["start_idx", "end_idx"], kind="stable")
            previous = None
            current_label = None
            current_bars = 0
            for row in seq.itertuples(index=False):
                label = str(row.label)
                if label == "UNKNOWN":
                    if current_label is not None:
                        runs += 1
                        run_bars.append(current_bars)
                    current_label, current_bars, previous = None, 0, None
                    continue
                if previous is not None and int(row.start_idx) <= int(previous.end_idx) + 1:
                    raw_pairs.append((str(previous.label), label))
                if current_label == label and previous is not None and int(row.start_idx) <= int(previous.end_idx) + 1:
                    # Segment boundaries may share their pivot bar; count the
                    # union of bar intervals instead of double-counting it.
                    current_bars += max(0, int(row.end_idx) - int(previous.end_idx))
                else:
                    if current_label is not None:
                        runs += 1
                        run_bars.append(current_bars)
                    current_label, current_bars = label, int(row.n_bars)
                previous = row
            if current_label is not None:
                runs += 1
                run_bars.append(current_bars)

        labels = sorted({str(x).removeprefix("prob_") for x in probs})
        counts = pd.crosstab(
            pd.Series([a for a, _ in raw_pairs], name="from"),
            pd.Series([b for _, b in raw_pairs], name="to"),
        ).reindex(index=labels, columns=labels, fill_value=0)
        denominator = counts.sum(axis=1).replace(0, np.nan)
        normalized = counts.div(denominator, axis=0).fillna(0)
        for from_label in labels:
            for to_label in labels:
                transitions.append({
                    "oos_model_trained_at": version,
                    "from_label": from_label,
                    "to_label": to_label,
                    "transition_count": int(counts.loc[from_label, to_label]),
                    "transition_probability": float(normalized.loc[from_label, to_label]),
                })

        # The metrics below are descriptive only.  Self-transition uses adjacent
        # candidate segments, while run length collapses consecutive same labels.
        versions.append({
            "oos_model_trained_at": version,
            "segments": len(group),
            "identified_segments": len(known),
            "unknown_fraction": float(1 - len(known) / len(group)) if len(group) else np.nan,
            "symbols": int(group.symbol.nunique()),
            "median_segment_bars_identified": float(known._duration.median()) if len(known) else np.nan,
            "p90_segment_bars_identified": float(known._duration.quantile(.9)) if len(known) else np.nan,
            "adjacent_identified_transition_pairs": len(raw_pairs),
            "adjacent_self_transition_rate": float(np.mean([a == b for a, b in raw_pairs])) if raw_pairs else np.nan,
            "same_label_runs": runs,
            "median_same_label_run_bars": float(np.median(run_bars)) if run_bars else np.nan,
            "identified_class_count": int(known.label.nunique()),
        })

    # Compare two explicit-duration settings on each symbol × fitted-version
    # block. This is a sensitivity diagnostic, not a claim that smoothing is
    # more correct than the unsmoothed output.
    hsmm_rows = []
    for minimum in sorted({max(1, int(min_duration_segments)), max(1, int(min_duration_segments) + 1)}):
        hsmm_input = frame.copy()
        hsmm_input["symbol"] = hsmm_input["_sequence"]
        decoded = decode_hsmm(
            hsmm_input,
            probability_columns=probs,
            max_duration=max_duration_segments,
            min_duration=minimum,
            label_col="label",
        )
        decoded = decoded.copy()
        decoded["_sequence"] = hsmm_input["_sequence"].to_numpy()
        for version, group in decoded.groupby("_model_version", sort=False):
            identified = group.loc[group.label.astype(str).ne("UNKNOWN")]
            comp = identified.loc[identified.hsmm_label.astype(str).ne("UNKNOWN")]
            pair_count = same = 0
            for _, seq in group.groupby("_sequence", sort=False):
                seq = seq.sort_values(["start_idx", "end_idx"], kind="stable")
                pairs = list(zip(seq.hsmm_label.astype(str).iloc[:-1], seq.hsmm_label.astype(str).iloc[1:]))
                valid = [(a, b) for a, b in pairs if a != "UNKNOWN" and b != "UNKNOWN"]
                pair_count += len(valid)
                same += sum(a == b for a, b in valid)
            hsmm_rows.append({
                "oos_model_trained_at": version,
                "min_duration_segments": minimum,
                "max_duration_segments": max_duration_segments,
                "identified_segments_before": len(identified),
                "hsmm_changed_fraction": float(comp.hsmm_changed.mean()) if len(comp) else np.nan,
                "identified_segments_after": len(comp),
                "adjacent_transition_pairs_after": pair_count,
                "self_transition_rate_after": float(same / pair_count) if pair_count else np.nan,
                "median_duration_bars_after": float(comp.n_bars.median()) if len(comp) else np.nan,
            })

    return pd.DataFrame(versions), pd.DataFrame(transitions), pd.DataFrame(hsmm_rows)
