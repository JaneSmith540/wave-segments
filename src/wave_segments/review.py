"""Human-review queue and label feedback helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REVIEW_COLUMNS = ["segment_id", "symbol", "start_timestamp", "end_timestamp", "candidate_status", "predicted_label", "suggested_semantic_label", "max_probability", "normalized_entropy", "recognizability", "unknown_reason", "review_priority", "sampling_stratum", "sampling_probability", "annotator", "review_status", "accepted", "true_label", "boundary_ok", "boundary_start_correction", "boundary_end_correction", "notes", "reviewed_at"]
REVIEWER_COLUMNS = ["annotator", "review_status", "accepted", "true_label", "boundary_ok", "boundary_start_correction", "boundary_end_correction", "notes", "reviewed_at"]


def _id_column(frame: pd.DataFrame) -> pd.Series:
    if "segment_id" in frame:
        return frame["segment_id"].astype(str)
    fields = [c for c in ("symbol", "start_timestamp", "end_timestamp", "start_idx", "end_idx") if c in frame]
    return frame[fields].astype(str).agg("|".join, axis=1) if fields else pd.Series([f"segment-{i}" for i in range(len(frame))], index=frame.index)


def create_review_queue(segments: pd.DataFrame, path: str | Path | None = None, *, limit: int | None = None) -> pd.DataFrame:
    """Create a CSV-friendly queue, strictly prioritising UNKNOWN segments."""
    queue = pd.DataFrame(index=segments.index)
    queue["segment_id"] = _id_column(segments)
    queue["symbol"] = segments.get("symbol", pd.NA)
    queue["start_timestamp"] = segments[next((c for c in ("start_timestamp", "start") if c in segments), "segment_id")] if any(c in segments for c in ("start_timestamp", "start")) else pd.NA
    queue["end_timestamp"] = segments[next((c for c in ("end_timestamp", "end") if c in segments), "segment_id")] if any(c in segments for c in ("end_timestamp", "end")) else pd.NA
    source = next((x for x in ("label", "display_label", "predicted_label", "hard_label") if x in segments), None)
    queue["predicted_label"] = segments[source].astype("string") if source else "UNKNOWN"
    queue["predicted_label"] = queue["predicted_label"].fillna("UNKNOWN")
    reason = segments.get("unknown_reason", pd.Series("", index=segments.index)).fillna("").astype(str)
    if "candidate_status" in segments:
        queue["candidate_status"] = segments["candidate_status"].fillna("unreviewed_candidate").astype(str)
    else:
        unknown = queue["predicted_label"].str.upper().eq("UNKNOWN")
        queue["candidate_status"] = np.where(
            reason.eq("unreviewed_candidate"), "unreviewed_candidate",
            np.where(unknown, "model_abstention", "classified"),
        )
    queue["suggested_semantic_label"] = segments.get("suggested_semantic_label", pd.NA)
    sources = {"max_probability": ("max_probability", "confidence", "max_posterior"), "normalized_entropy": ("normalized_entropy", "entropy"), "recognizability": ("recognizability", "recognizability_score"), "unknown_reason": ("unknown_reason", "abstention_reason")}
    for target, options in sources.items():
        column = next((x for x in options if x in segments), None)
        queue[target] = segments[column] if column else ("" if target == "unknown_reason" else np.nan)
    unknown = queue.predicted_label.str.upper().eq("UNKNOWN")
    recog = pd.to_numeric(queue.recognizability, errors="coerce").fillna(0).clip(0, 1)
    entropy = pd.to_numeric(queue.normalized_entropy, errors="coerce").fillna(1).clip(0, 1)
    queue["review_priority"] = unknown.astype(int) * 10 + (1 - recog) + entropy * .25
    queue["sampling_stratum"] = segments.get("sampling_stratum", pd.Series(pd.NA, index=segments.index))
    queue["sampling_probability"] = segments.get("sampling_probability", pd.Series(np.nan, index=segments.index))
    for col in REVIEWER_COLUMNS:
        queue[col] = pd.NA
    if path is not None and Path(path).exists():
        queue = merge_reviews(queue, pd.read_csv(path, dtype={"segment_id": str}))
    queue = queue.sort_values(["review_priority", "segment_id"], ascending=[False, True], kind="stable")
    if limit is not None:
        queue = queue.head(limit)
    queue = queue.reindex(columns=REVIEW_COLUMNS).reset_index(drop=True)
    if path is not None:
        output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); queue.to_csv(output, index=False, encoding="utf-8-sig")
    return queue


def build_stratified_review_sample(
    segments: pd.DataFrame,
    target_size: int,
    *,
    unknown_share: float = 0.35,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a reproducible UNKNOWN-enriched but representative review sample.

    Sampling is balanced in round-robin order across available year, industry,
    market regime, size bucket and current temporary label strata.  Symbols are
    sampled randomly *within* these coarse strata: including symbol in the
    stratum key creates thousands of near-singleton strata and can make a
    bounded sample depend on lexical symbol/year order.

    ``sampling_probability`` is the actual stratum sampling fraction, not just
    the overall UNKNOWN/known pool fraction, so inverse-probability estimates
    remain valid under the deliberately non-proportional allocation.
    """
    if target_size < 1:
        raise ValueError("target_size must be positive")
    if not 0 <= unknown_share <= 1:
        raise ValueError("unknown_share must be between 0 and 1")
    if segments.empty:
        return create_review_queue(segments)
    data = segments.copy()
    start_col = next((c for c in ("start", "start_timestamp") if c in data), None)
    data["_year"] = pd.to_datetime(data[start_col], errors="coerce").dt.year.astype("Int64").astype(str) if start_col else "NA"
    label_col = next((c for c in ("label", "predicted_label", "candidate_label") if c in data), None)
    data["_label"] = data[label_col].fillna("UNKNOWN").astype(str) if label_col else "UNKNOWN"
    stratum_cols = ["_year", "_label"] + [
        c for c in ("industry", "market_regime", "size_bucket") if c in data
    ]
    data["sampling_stratum"] = data[stratum_cols].fillna("NA").astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(random_state)

    def balanced(pool: pd.DataFrame, count: int) -> pd.DataFrame:
        if count <= 0 or pool.empty:
            return pool.iloc[0:0]
        groups = [(str(key), group) for key, group in pool.groupby("sampling_stratum", sort=True)]
        allocation = {key: 0 for key, _ in groups}
        capacity = {key: len(group) for key, group in groups}
        # Randomize each round so a target smaller than the number of strata is
        # not biased toward early years or lexical category names.
        remaining = min(count, len(pool))
        while remaining:
            eligible = [key for key, _ in groups if allocation[key] < capacity[key]]
            if not eligible:
                break
            for key in rng.permutation(eligible):
                allocation[str(key)] += 1
                remaining -= 1
                if not remaining:
                    break
        pieces = []
        for key, group in groups:
            take = allocation[key]
            if take:
                chosen = group.iloc[rng.choice(len(group), size=take, replace=False)].copy()
                chosen["sampling_probability"] = take / len(group)
                pieces.append(chosen)
        return pd.concat(pieces) if pieces else pool.iloc[0:0]

    unknown = data[data["_label"].str.upper().eq("UNKNOWN")]
    known = data[~data.index.isin(unknown.index)]
    total = min(target_size, len(data))
    unknown_n = min(len(unknown), round(total * unknown_share))
    known_n = min(len(known), total - unknown_n)
    unknown_n = min(len(unknown), unknown_n + max(0, total - unknown_n - known_n))
    selected = pd.concat([balanced(unknown, unknown_n), balanced(known, known_n)])
    if len(selected) < total:
        selected = pd.concat([selected, balanced(data.drop(index=selected.index), total - len(selected))])
    return create_review_queue(selected.drop(columns=["_random"], errors="ignore"))


def create_review_table(segments: pd.DataFrame, path: str | Path | None = None, *, limit: int | None = None) -> pd.DataFrame:
    """Pipeline-compatible alias for :func:`create_review_queue`."""
    return create_review_queue(segments, path, limit=limit)


def merge_reviews(queue: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Merge reviewer-owned fields onto a freshly generated queue by segment_id."""
    out = queue.copy(); out["segment_id"] = _id_column(out)
    if "segment_id" not in reviews:
        raise ValueError("review CSV must contain a segment_id column")
    old = reviews.copy(); old.segment_id = old.segment_id.astype(str)
    if "reviewed_at" in old: old = old.sort_values("reviewed_at", kind="stable")
    old = old.drop_duplicates("segment_id", keep="last").set_index("segment_id")
    for col in REVIEWER_COLUMNS:
        if col in old:
            values = out.segment_id.map(old[col])
            out[col] = values.combine_first(out[col]) if col in out else values
    return out


def apply_review_labels(segments: pd.DataFrame, reviews: pd.DataFrame | str | Path, *, label_col: str = "label") -> pd.DataFrame:
    """Set ``supervised_label`` from corrected labels or accepted predictions."""
    review = pd.read_csv(reviews, dtype={"segment_id": str}) if isinstance(reviews, (str, Path)) else reviews.copy()
    if "segment_id" not in review: raise ValueError("reviews must contain segment_id")
    out = segments.copy(); out["segment_id"] = _id_column(out)
    review = review.drop_duplicates("segment_id", keep="last").set_index(review.segment_id.astype(str))
    true = out.segment_id.map(review.get("true_label", pd.Series(dtype="object"))).astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    accepted = out.segment_id.map(review.get("accepted", pd.Series(dtype="object"))).astype(str).str.lower().isin({"true", "1", "yes", "y", "accepted"})
    predicted = out[label_col] if label_col in out else out.get("predicted_label", pd.Series(pd.NA, index=out.index))
    out["supervised_label"] = true.where(true.notna(), predicted.astype("string").where(accepted, pd.NA))
    return out
