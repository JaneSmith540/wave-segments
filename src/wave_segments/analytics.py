"""Descriptive analysis and explainable summaries for labelled segments."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from .model import UNKNOWN, infer_feature_columns
from .validation import annotation_agreement_report


def label_descriptions(segments: pd.DataFrame, label_column: str = "label",
                       feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-label coverage, duration and numeric median/dispersion statistics."""
    if label_column not in segments:
        raise ValueError(f"Missing label column {label_column!r}")
    numeric = infer_feature_columns(segments, feature_columns)
    work = segments.copy()
    if not {"duration", "duration_bars", "n_bars"}.intersection(work.columns) and {"start", "end"}.issubset(work.columns):
        work["duration_days"] = (pd.to_datetime(work["end"]) - pd.to_datetime(work["start"])).dt.total_seconds() / 86400
        numeric = list(dict.fromkeys([*numeric, "duration_days"]))
    grouped = work.groupby(label_column, dropna=False)
    summary = grouped[numeric].agg(["count", "median", "mean", "std", "min", "max"])
    summary.columns = [f"{field}_{stat}" for field, stat in summary.columns]
    summary.insert(0, "segments", grouped.size())
    summary.insert(1, "share", grouped.size() / len(work) if len(work) else 0.0)
    summary.insert(2, "is_unknown", summary.index.astype(str) == UNKNOWN)
    return summary.reset_index().sort_values("segments", ascending=False).reset_index(drop=True)


def transition_matrix(segments: pd.DataFrame, label_column: str = "label", symbol_column: str = "symbol",
                      order_column: str | None = None, include_unknown: bool = True) -> pd.DataFrame:
    """Row-normalized current-label → next-label matrix, never crossing a symbol."""
    if label_column not in segments:
        raise ValueError(f"Missing label column {label_column!r}")
    work = segments.copy()
    if order_column is None:
        order_column = next((c for c in ("start_timestamp", "start", "segment_id") if c in work), None)
    by = [symbol_column] if symbol_column in work else []
    if order_column:
        work = work.sort_values([*by, order_column])
    work["_next"] = work.groupby(symbol_column if symbol_column in work else lambda _: 0)[label_column].shift(-1)
    pairs = work.dropna(subset=["_next"])
    if not include_unknown:
        pairs = pairs[(pairs[label_column] != UNKNOWN) & (pairs._next != UNKNOWN)]
    counts = pd.crosstab(pairs[label_column], pairs._next)
    labels = sorted(set(work[label_column].dropna()) | set(work._next.dropna()))
    if not include_unknown:
        labels = [x for x in labels if x != UNKNOWN]
    return counts.reindex(index=labels, columns=labels, fill_value=0).div(counts.reindex(index=labels, columns=labels, fill_value=0).sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def decision_tree_rules(segments: pd.DataFrame, label_column: str = "label", feature_columns: Sequence[str] | None = None,
                        max_depth: int = 3, min_samples_leaf: int = 5, include_unknown: bool = False) -> tuple[DecisionTreeClassifier, str]:
    """Fit a shallow descriptive tree and return it with human-readable rules."""
    work = segments.copy()
    if not include_unknown:
        work = work[work[label_column] != UNKNOWN]
    if work[label_column].nunique() < 2:
        raise ValueError("Decision tree needs at least two recognised labels")
    features = infer_feature_columns(work, feature_columns)
    X = work[features].replace([np.inf, -np.inf], np.nan).fillna(work[features].median()).fillna(0)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight="balanced", random_state=42)
    model.fit(X, work[label_column])
    return model, export_text(model, feature_names=features, decimals=3)


def analytics_bundle(segments: pd.DataFrame, **kwargs) -> dict[str, object]:
    """Convenient serialisable analysis package for a pipeline/report."""
    descriptions = label_descriptions(segments, **{k: v for k, v in kwargs.items() if k in {"label_column", "feature_columns"}})
    transitions = transition_matrix(segments, label_column=kwargs.get("label_column", "label"))
    try:
        tree, rules = decision_tree_rules(segments, **{k: v for k, v in kwargs.items() if k in {"label_column", "feature_columns", "max_depth", "min_samples_leaf"}})
    except ValueError:
        tree, rules = None, "Insufficient recognised label diversity for decision-tree rules."
    return {"descriptions": descriptions, "transition_matrix": transitions, "tree": tree, "rules": rules}


def human_annotation_agreement(annotations: pd.DataFrame, *, tolerance: int = 3) -> dict[str, object]:
    """Expose conservative multi-rater agreement beside descriptive analytics."""
    return annotation_agreement_report(annotations, tolerance=tolerance)


# Notebook-friendly aliases.
describe_labels = label_descriptions
build_descriptions = label_descriptions
build_transition_matrix = transition_matrix


def build_tree_rules(segments: pd.DataFrame, **kwargs) -> str:
    try:
        return decision_tree_rules(segments, **kwargs)[1]
    except ValueError:
        return "Insufficient recognised label diversity for decision-tree rules."
