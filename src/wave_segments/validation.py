"""Leakage-aware validation for the *descriptive* segment classifier.

This module deliberately has no forward-return, IC, or trading code.  Its inputs
are features known when a segment closes and human shape labels; selection-model
validation belongs in a separate layer.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

UNKNOWN = "UNKNOWN"
_FORWARD_TOKENS = ("future", "forward", "fwd", "next_", "return_5", "return_20", "return_60")
_ID_COLUMNS = {"segment_id", "symbol", "start", "end", "start_timestamp", "end_timestamp",
               "start_idx", "end_idx", "label", "consensus_label", "annotator",
               "segment_no", "max_probability", "posterior_entropy", "model_disagreement",
               "density_score", "density_is_low", "recognizability", "is_unknown",
               "cluster_support_valid"}


def assert_descriptive_features(columns: Sequence[str]) -> None:
    """Reject obvious outcome fields; callers should use only close-time features."""
    bad = [c for c in columns if any(t in c.lower() for t in _FORWARD_TOKENS)]
    if bad:
        raise ValueError(f"Future/outcome columns are forbidden in classification: {bad}")


def infer_validation_features(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> list[str]:
    cols = list(feature_columns) if feature_columns is not None else [
        c for c in frame.select_dtypes(include=np.number).columns
        if c not in _ID_COLUMNS and not c.startswith(("prob_", "future_", "forward_"))
    ]
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature columns not found: {sorted(missing)}")
    assert_descriptive_features(cols)
    if not cols:
        raise ValueError("No descriptive numeric features supplied")
    return cols


def _intervals_overlap(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Whether each left interval overlaps any right interval, within a symbol."""
    out = np.zeros(len(left), dtype=bool)
    for symbol, lgrp in left.groupby("symbol", sort=False):
        rgrp = right[right.symbol == symbol]
        if rgrp.empty:
            continue
        rs = rgrp["start"].to_numpy()
        re = rgrp["end"].to_numpy()
        for pos, (_, row) in enumerate(lgrp.iterrows()):
            out[left.index.get_loc(row.name)] = bool(((rs <= row.end) & (re >= row.start)).any())
    return out


@dataclass(frozen=True)
class WalkForwardFold:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


class PurgedWalkForwardSplit:
    """Global-calendar expanding folds with a separate calibration window.

    Any earlier segment whose full K-line interval touches a later partition is
    removed.  Thus overlapping bars can never appear in train/calibration/test.
    """
    def __init__(self, n_splits: int = 3, min_train_segments: int = 8):
        if n_splits < 1:
            raise ValueError("n_splits must be positive")
        self.n_splits, self.min_train_segments = n_splits, min_train_segments

    def split(self, frame: pd.DataFrame) -> Iterator[WalkForwardFold]:
        required = {"symbol", "start", "end"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Missing interval columns: {sorted(missing)}")
        data = frame.copy()
        data["start"] = pd.to_datetime(data.start)
        data["end"] = pd.to_datetime(data.end)
        if (data.end < data.start).any():
            raise ValueError("A segment end precedes its start")
        dates = pd.Index(data["end"].drop_duplicates().sort_values())
        blocks = [pd.Index(block) for block in np.array_split(dates.to_numpy(), self.n_splits + 2) if len(block)]
        if len(blocks) < self.n_splits + 2:
            return
        for fold in range(self.n_splits):
            train_dates = pd.Index(np.concatenate([block.to_numpy() for block in blocks[:fold + 1]]))
            calibration_dates, test_dates = blocks[fold + 1], blocks[fold + 2]
            train = data[data.end.isin(train_dates)]
            calibration = data[data.end.isin(calibration_dates)]
            test = data[data.end.isin(test_dates)]
            if calibration.empty or test.empty:
                continue
            # Global purging is intentionally stricter than per-symbol purging:
            # shared market/context bars cannot cross partition boundaries either.
            train = train[train.end < min(calibration.start.min(), test.start.min())]
            calibration = calibration[calibration.end < test.start.min()]
            if len(train) >= self.min_train_segments and not calibration.empty:
                yield WalkForwardFold(train.index.to_numpy(), calibration.index.to_numpy(), test.index.to_numpy())


def resolve_human_annotations(annotations: pd.DataFrame, segment_col: str = "segment_id",
                              annotator_col: str = "annotator", label_col: str = "label",
                              inclusion_probability_col: str | None = "sampling_probability") -> tuple[pd.DataFrame, dict]:
    """Create a conservative multi-rater consensus and agreement statistics.

    Each annotator has one effective vote per segment (the latest append-only
    event when ``reviewed_at`` is present).  A label is a ground truth only when
    at least two independent raters unanimously agree.  Every other case,
    including a single rater, is explicitly ``UNKNOWN``.  When a review sample
    records inclusion probabilities, all population-level agreement estimates
    use inverse-probability weights.
    """
    required = {segment_col, annotator_col, label_col}
    if missing := required - set(annotations.columns):
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    work = _latest_annotation_events(annotations, segment_col, annotator_col)
    if inclusion_probability_col and inclusion_probability_col in work:
        inclusion = pd.to_numeric(work[inclusion_probability_col], errors="coerce")
        if (inclusion.notna() & ((inclusion <= 0) | ~np.isfinite(inclusion))).any():
            raise ValueError("Inclusion probabilities must be finite and positive")
    rows = []
    for seg, grp in work.dropna(subset=[label_col]).groupby(segment_col, sort=False):
        labels = grp[label_col].astype(str)
        counts = labels.value_counts()
        top = counts.index[0]
        rate = float(counts.iloc[0] / len(labels))
        unanimous = counts.size == 1 and len(labels) >= 2
        probability = _segment_inclusion_probability(grp, inclusion_probability_col)
        reason = "unanimous" if unanimous else ("insufficient_independent_raters" if len(labels) < 2 else "label_disagreement")
        # Any disagreement (or only one rater) is UNKNOWN: avoids manufacturing
        # an unreliable truth from a single subjective interpretation.
        rows.append({segment_col: seg, "consensus_label": top if unanimous else UNKNOWN,
                     "annotator_count": len(labels), "agreement_rate": rate,
                     "is_disputed": not unanimous, "vote_counts": json.dumps(counts.to_dict(), ensure_ascii=False, sort_keys=True),
                     "consensus_reason": reason, "sampling_probability": probability})
    consensus = pd.DataFrame(rows)
    kappas, agreements, pair_details = [], [], []
    pivot = work.pivot(index=segment_col, columns=annotator_col, values=label_col)
    probability_by_segment = (consensus.set_index(segment_col)["sampling_probability"]
                              if not consensus.empty else pd.Series(dtype=float))
    for a, b in combinations(pivot.columns, 2):
        pair = pivot[[a, b]].dropna()
        if len(pair):
            weights = _inverse_probability_weights(pair.index, probability_by_segment)
            raw_agreement = float(np.average(pair[a].eq(pair[b]), weights=weights))
            kappa = _weighted_cohen_kappa(pair[a].astype(str), pair[b].astype(str), weights)
            agreements.append(raw_agreement); kappas.append(kappa)
            pair_details.append({"annotator_a": str(a), "annotator_b": str(b), "shared_segments": len(pair),
                                 "weighted_segments": float(weights.sum()), "raw_agreement": raw_agreement,
                                 "cohen_kappa": kappa})
    multi = pivot.dropna(thresh=2)
    fleiss = _weighted_fleiss_kappa(multi, _inverse_probability_weights(multi.index, probability_by_segment)) if len(pivot.columns) >= 3 else np.nan
    weights = _inverse_probability_weights(consensus.index, consensus["sampling_probability"] if not consensus.empty else pd.Series(dtype=float))
    finite_agreements = [value for value in agreements if np.isfinite(value)]
    finite_kappas = [value for value in kappas if np.isfinite(value)]
    summary = {"pairwise_agreement": float(np.mean(finite_agreements)) if finite_agreements else np.nan,
               "cohen_kappa": float(np.mean(finite_kappas)) if finite_kappas else np.nan,
               "fleiss_kappa": fleiss,
               "annotated_segments": len(rows), "weighted_annotated_segments": float(weights.sum()),
               "disputed_segments": int(sum(r["is_disputed"] for r in rows)),
               "weighted_dispute_rate": float(np.average(consensus["is_disputed"], weights=weights)) if len(consensus) else np.nan,
               "pairwise_details": pair_details}
    return consensus, summary


def _latest_annotation_events(annotations: pd.DataFrame, segment_col: str, annotator_col: str) -> pd.DataFrame:
    """Select the latest immutable event for every (segment, annotator) vote."""
    work = annotations.copy()
    work["_event_order"] = np.arange(len(work))
    if "reviewed_at" in work:
        work["_reviewed_at"] = pd.to_datetime(work["reviewed_at"], errors="coerce", utc=True)
        # Undated legacy events are older than any auditable timestamp. If all
        # events in a pair are undated, append order remains the fallback.
        work = work.sort_values([segment_col, annotator_col, "_reviewed_at", "_event_order"],
                                kind="stable", na_position="first")
    return work.drop_duplicates([segment_col, annotator_col], keep="last").drop(
        columns=["_event_order", "_reviewed_at"], errors="ignore"
    )


def _segment_inclusion_probability(group: pd.DataFrame, column: str | None) -> float:
    if not column or column not in group:
        return 1.0
    values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
    if len(values) > 1:
        raise ValueError("A segment has conflicting sampling probabilities")
    return float(values[0]) if len(values) else 1.0


def _inverse_probability_weights(index: pd.Index, probabilities: pd.Series) -> np.ndarray:
    values = pd.to_numeric(probabilities.reindex(index), errors="coerce").fillna(1.0).to_numpy(float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Inclusion probabilities must be finite and positive")
    return 1.0 / values


def _weighted_cohen_kappa(left: pd.Series, right: pd.Series, weights: np.ndarray) -> float:
    """Nominal Cohen kappa computed from weighted observed/marginal rates."""
    labels = pd.Index(sorted(set(left) | set(right)))
    total = float(weights.sum())
    if not total:
        return np.nan
    observed = float(np.sum(weights * left.eq(right).to_numpy()) / total)
    p_left = np.array([weights[left.eq(label).to_numpy()].sum() / total for label in labels])
    p_right = np.array([weights[right.eq(label).to_numpy()].sum() / total for label in labels])
    expected = float(np.dot(p_left, p_right))
    return float((observed - expected) / (1 - expected)) if not np.isclose(expected, 1.0) else np.nan


def _weighted_fleiss_kappa(votes: pd.DataFrame, weights: np.ndarray) -> float:
    """Fleiss kappa for incomplete multi-rater panels, weighted by sample design."""
    if votes.empty or len(votes.columns) < 3:
        return np.nan
    rows, row_weights = [], []
    labels = sorted({str(value) for value in votes.to_numpy().ravel() if pd.notna(value)})
    for pos, (_, row) in enumerate(votes.iterrows()):
        values = row.dropna().astype(str)
        n = len(values)
        if n < 2:
            continue
        counts = values.value_counts().reindex(labels, fill_value=0).to_numpy(float)
        rows.append((counts, n)); row_weights.append(weights[pos])
    if not rows:
        return np.nan
    row_weights = np.asarray(row_weights, float)
    weighted_n = sum(weight * n for weight, (_, n) in zip(row_weights, rows))
    category_share = sum(weight * counts for weight, (counts, _) in zip(row_weights, rows)) / weighted_n
    p_bar = np.average([(np.square(counts).sum() - n) / (n * (n - 1)) for counts, n in rows], weights=row_weights)
    p_expected = float(np.square(category_share).sum())
    return float((p_bar - p_expected) / (1 - p_expected)) if not np.isclose(p_expected, 1.0) else np.nan


def boundary_annotation_agreement(annotations: pd.DataFrame, *, tolerance: int = 3,
                                  segment_col: str = "segment_id", annotator_col: str = "annotator",
                                  start_col: str = "start_idx", end_col: str = "end_idx",
                                  start_correction_col: str = "boundary_start_correction",
                                  end_correction_col: str = "boundary_end_correction",
                                  inclusion_probability_col: str | None = "sampling_probability") -> dict:
    """Compare independently proposed boundaries within ``±tolerance`` bars.

    Correction fields take precedence over the sampled segment's original
    integer indices.  The UI records corrected timestamps as well; those are
    intentionally *not* coerced to a bar number here because that requires the
    source bar calendar.  Join corrected timestamps to bars first, then pass
    their resulting ``start_idx``/``end_idx`` here.  This keeps the metric from
    silently treating calendar days as trading bars.

    The returned pairwise values view each annotator in both directions, so
    precision means "my proposed boundaries matched the other rater" and
    recall means the converse.  Inclusion probabilities are inverse weighted.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    required = {segment_col, annotator_col}
    if missing := required - set(annotations.columns):
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    work = _latest_annotation_events(annotations, segment_col, annotator_col)
    bounds = pd.DataFrame({segment_col: work[segment_col], annotator_col: work[annotator_col]})
    for output, base, correction in (("start_boundary_idx", start_col, start_correction_col),
                                     ("end_boundary_idx", end_col, end_correction_col)):
        base_values = pd.to_numeric(work[base], errors="coerce") if base in work else pd.Series(np.nan, index=work.index)
        correction_values = pd.to_numeric(work[correction], errors="coerce") if correction in work else pd.Series(np.nan, index=work.index)
        bounds[output] = correction_values.where(correction_values.notna(), base_values)
    if inclusion_probability_col and inclusion_probability_col in work:
        bounds["sampling_probability"] = pd.to_numeric(work[inclusion_probability_col], errors="coerce")
        conflicts = (bounds.groupby(segment_col)["sampling_probability"].nunique(dropna=True) > 1)
        if conflicts.any():
            raise ValueError("A segment has conflicting sampling probabilities")
    else:
        bounds["sampling_probability"] = 1.0
    bounds["_reviewed"] = True
    details: list[dict] = []
    all_scores: list[dict] = []
    for a, b in combinations(bounds[annotator_col].dropna().unique(), 2):
        pair = bounds[bounds[annotator_col].isin([a, b])].pivot(index=segment_col, columns=annotator_col,
                                                                  values=["start_boundary_idx", "end_boundary_idx", "sampling_probability", "_reviewed"])
        # A rater must have proposed both values for a boundary to enter that
        # boundary-type denominator; unknown / no-opinion stays out of it.
        for side in ("start", "end"):
            try:
                left, right = pair[(f"{side}_boundary_idx", a)], pair[(f"{side}_boundary_idx", b)]
            except KeyError:
                continue
            # An absent rater is not a false boundary proposal.  Only segments
            # that both reviewers independently reviewed enter this pair's
            # denominators; otherwise prolific reviewers are unfairly penalised.
            both_reviewed = pair[("_reviewed", a)].notna() & pair[("_reviewed", b)].notna()
            valid_left, valid_right = both_reviewed & left.notna(), both_reviewed & right.notna()
            shared = valid_left & valid_right
            if not shared.any():
                continue
            probability = pair[("sampling_probability", a)].combine_first(pair[("sampling_probability", b)])
            weights = _inverse_probability_weights(pair.index, probability)
            matched = shared & left.sub(right).abs().le(tolerance)
            tp = float(weights[matched].sum())
            predicted = float(weights[valid_left].sum())
            reference = float(weights[valid_right].sum())
            precision = tp / predicted if predicted else np.nan
            recall = tp / reference if reference else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            item = {"annotator_a": str(a), "annotator_b": str(b), "boundary": side,
                    "tolerance_bars": tolerance, "shared_segments": int(shared.sum()),
                    "weighted_true_positive": tp, "weighted_predicted": predicted,
                    "weighted_reference": reference, "precision": precision, "recall": recall, "f1": f1}
            details.append(item); all_scores.append(item)
    if all_scores:
        # Macro-average prevents a prolific pair or side from dominating the
        # headline agreement measure; raw denominators remain in details.
        aggregate = {metric: float(np.nanmean([row[metric] for row in all_scores]))
                     for metric in ("precision", "recall", "f1")}
    else:
        aggregate = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
    return {"tolerance_bars": tolerance, "pairwise": pd.DataFrame(details),
            "pairwise_macro": aggregate, "scored_pairs": len(details)}


def annotation_agreement_report(annotations: pd.DataFrame, *, tolerance: int = 3,
                                **kwargs) -> dict[str, object]:
    """One report for consensus, label agreement and boundary agreement."""
    consensus, labels = resolve_human_annotations(annotations, **kwargs)
    boundary_kwargs = {key: value for key, value in kwargs.items()
                       if key in {"segment_col", "annotator_col", "inclusion_probability_col"}}
    return {"consensus": consensus, "label_agreement": labels,
            "boundary_agreement": boundary_annotation_agreement(annotations, tolerance=tolerance, **boundary_kwargs)}


def expected_calibration_error(y_true: Sequence[str], probabilities: np.ndarray,
                               classes: Sequence[str], bins: int = 10,
                               sample_weight: Sequence[float] | None = None) -> float:
    y = np.asarray(y_true, dtype=object)
    classes = np.asarray(classes, dtype=object)
    confidence = probabilities.max(axis=1)
    correct = classes[probabilities.argmax(axis=1)] == y
    weights = np.ones(len(y), float) if sample_weight is None else np.asarray(sample_weight, float)
    total = weights.sum()
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            mass = weights[mask].sum() / total
            accuracy = np.average(correct[mask], weights=weights[mask])
            mean_confidence = np.average(confidence[mask], weights=weights[mask])
            ece += mass * abs(float(accuracy) - float(mean_confidence))
    return float(ece)


def reliability_table(y_true: Sequence[str], probabilities: np.ndarray,
                      classes: Sequence[str], bins: int = 10,
                      sample_weight: Sequence[float] | None = None) -> pd.DataFrame:
    """Return reliability-diagram bins without coupling validation to plotting."""
    y, classes = np.asarray(y_true, object), np.asarray(classes, object)
    confidence = probabilities.max(axis=1)
    correct = classes[probabilities.argmax(axis=1)] == y
    weights = np.ones(len(y), float) if sample_weight is None else np.asarray(sample_weight, float)
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for number, (lo, hi) in enumerate(pairwise(edges)):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else confidence <= hi)
        rows.append({"bin": number, "lower": float(lo), "upper": float(hi),
                     "count": int(mask.sum()),
                     "weighted_count": float(weights[mask].sum()),
                     "mean_confidence": float(np.average(confidence[mask], weights=weights[mask])) if mask.any() else np.nan,
                     "accuracy": float(np.average(correct[mask], weights=weights[mask])) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def coverage_risk_curve(y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str],
                        thresholds: Sequence[float] | None = None,
                        sample_weight: Sequence[float] | None = None) -> pd.DataFrame:
    thresholds = np.linspace(0, 1, 101) if thresholds is None else np.asarray(thresholds, float)
    y, classes = np.asarray(y_true, object), np.asarray(classes, object)
    conf, pred = probabilities.max(1), classes[probabilities.argmax(1)]
    weights = np.ones(len(y), float) if sample_weight is None else np.asarray(sample_weight, float)
    rows = []
    for t in thresholds:
        accepted = conf >= t
        error = float(np.average(pred[accepted] != y[accepted], weights=weights[accepted])) if accepted.any() else np.nan
        coverage = float(weights[accepted].sum() / weights.sum()) if weights.sum() else np.nan
        rows.append({"threshold": float(t), "coverage": coverage, "risk": error,
                     "accepted": int(accepted.sum())})
    return pd.DataFrame(rows)


class CalibratedSegmentClassifier:
    """Simple supervised baseline calibrated only on its held-out calibration set."""
    def __init__(self, feature_columns: Sequence[str] | None = None, method: str = "sigmoid",
                 estimator: str = "logistic", random_state: int = 42):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be sigmoid or isotonic")
        self.feature_columns, self.method, self.estimator, self.random_state = feature_columns, method, estimator, random_state

    def _base(self):
        model = (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state)
                 if self.estimator == "logistic" else DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=self.random_state))
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame, label_col: str = "consensus_label") -> CalibratedSegmentClassifier:
        self.feature_columns = infer_validation_features(train, self.feature_columns)
        assert_descriptive_features(self.feature_columns)
        train = train[train[label_col].notna() & train[label_col].ne(UNKNOWN)]
        calibration = calibration[calibration[label_col].notna() & calibration[label_col].ne(UNKNOWN)]
        if train[label_col].nunique() < 2 or calibration.empty:
            raise ValueError("Training needs >=2 labels and calibration needs labelled rows")
        self.base_ = self._base().fit(train[self.feature_columns], train[label_col])
        try:
            # sklearn >=1.6 replaced cv="prefit" with an explicit frozen wrapper.
            from sklearn.frozen import FrozenEstimator
            self.calibrator_ = CalibratedClassifierCV(FrozenEstimator(self.base_), method=self.method)
        except ImportError:  # pragma: no cover - compatibility with older sklearn
            self.calibrator_ = CalibratedClassifierCV(self.base_, cv="prefit", method=self.method)
        self.calibrator_.fit(calibration[self.feature_columns], calibration[label_col])
        self.classes_ = self.calibrator_.classes_
        self.unknown_threshold_ = 1.0
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator_.predict_proba(frame[self.feature_columns])

    def choose_unknown_threshold(self, calibration: pd.DataFrame, label_col: str = "consensus_label",
                                 target_error: float = .10, min_coverage: float = 0.0,
                                 inclusion_probability_col: str | None = None) -> float:
        y = calibration[label_col].to_numpy(object)
        p = self.predict_proba(calibration)
        weights = None
        if inclusion_probability_col:
            probabilities = pd.to_numeric(calibration[inclusion_probability_col], errors="coerce").to_numpy(float)
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
                raise ValueError("Inclusion probabilities must be finite and positive")
            weights = 1 / probabilities
        curve = coverage_risk_curve(y, p, self.classes_, sample_weight=weights)
        eligible = curve[(curve.coverage >= min_coverage) & (curve.risk <= target_error)]
        self.unknown_threshold_ = float(eligible.threshold.min()) if not eligible.empty else 1.0
        return self.unknown_threshold_

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(frame)
        labels = self.classes_[p.argmax(1)].astype(object)
        labels[p.max(1) < self.unknown_threshold_] = UNKNOWN
        return labels

    def evaluate(self, frame: pd.DataFrame, label_col: str = "consensus_label",
                 inclusion_probability_col: str | None = None) -> dict:
        evaluated = frame[label_col].notna()
        all_y = frame.loc[evaluated, label_col].to_numpy(object)
        all_p = self.predict_proba(frame.loc[evaluated])
        all_weights = np.ones(len(all_y), float)
        if inclusion_probability_col:
            inclusion = pd.to_numeric(frame.loc[evaluated, inclusion_probability_col], errors="coerce").to_numpy(float)
            if np.any(~np.isfinite(inclusion)) or np.any(inclusion <= 0):
                raise ValueError("Inclusion probabilities must be finite and positive")
            all_weights = 1 / inclusion
        known = all_y != UNKNOWN
        y, p, weights = all_y[known], all_p[known], all_weights[known]
        if not len(y):
            raise ValueError("Evaluation needs at least one non-UNKNOWN truth label")
        raw = self.classes_[p.argmax(1)]
        precision, recall, _, support = precision_recall_fscore_support(
            y, raw, labels=self.classes_, sample_weight=weights, zero_division=0
        )
        onehot = (y[:, None] == self.classes_[None, :]).astype(float)
        all_accepted = all_p.max(1) >= self.unknown_threshold_
        accepted = all_accepted[known]
        selective_curve = coverage_risk_curve(all_y, all_p, self.classes_, sample_weight=all_weights)
        accepted_error = self.classes_[all_p.argmax(1)][all_accepted] != all_y[all_accepted]
        selective_risk = (float(np.average(accepted_error, weights=all_weights[all_accepted]))
                          if all_accepted.any() else np.nan)
        unknown = all_y == UNKNOWN
        return {"macro_f1": float(f1_score(y, raw, average="macro", sample_weight=weights, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, raw, sample_weight=weights)),
                "per_class": {str(c): {"precision": float(pr), "recall": float(re), "weighted_support": float(s)} for c, pr, re, s in zip(self.classes_, precision, recall, support)},
                "brier": float(np.average(np.sum((p - onehot) ** 2, axis=1), weights=weights)),
                "log_loss": float(log_loss(y, p, labels=self.classes_, sample_weight=weights)),
                "ece": expected_calibration_error(all_y, all_p, self.classes_, sample_weight=all_weights),
                "reliability": reliability_table(all_y, all_p, self.classes_, sample_weight=all_weights),
                "coverage_risk": selective_curve,
                "coverage": float(all_weights[all_accepted].sum() / all_weights.sum()),
                "selective_risk": selective_risk,
                "known_coverage": float(weights[accepted].sum() / weights.sum()),
                "unknown_rejection_rate": (float(all_weights[unknown & ~all_accepted].sum() / all_weights[unknown].sum())
                                           if unknown.any() else np.nan)}

    def fit_aps(self, calibration: pd.DataFrame, label_col: str = "consensus_label", alpha: float = .10) -> float:
        """Split-conformal APS set predictor; exchangeability is an assumption."""
        y, p = calibration[label_col].to_numpy(object), self.predict_proba(calibration)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        scores = []
        for label, row in zip(y, p):
            if label in lookup:
                scores.append(np.sort(row)[::-1][:np.sum(row >= row[lookup[label]])].sum())
        if not scores:
            raise ValueError("No calibration labels overlap classifier classes")
        rank = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1)
        self.aps_q_ = float(np.sort(scores)[rank])
        return self.aps_q_

    def predict_sets(self, frame: pd.DataFrame) -> list[set[str]]:
        if not hasattr(self, "aps_q_"):
            raise RuntimeError("Call fit_aps before predict_sets")
        result = []
        for row in self.predict_proba(frame):
            order = np.argsort(row)[::-1]; total, chosen = 0.0, set()
            for i in order:
                total += row[i]; chosen.add(str(self.classes_[i]))
                if total >= self.aps_q_:
                    break
            result.append(chosen)
        return result
