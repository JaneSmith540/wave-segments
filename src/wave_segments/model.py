"""Soft, abstaining descriptions of variable length market segments.

The model deliberately does not produce a trading prediction.  It groups rows of a
segment feature table and exposes *all* candidate memberships.  A label is emitted
only when there is enough evidence; otherwise ``UNKNOWN`` is a normal first-class
result.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from .config import ModelConfig

UNKNOWN = "UNKNOWN"
_NON_FEATURES = {
    "segment_id", "symbol", "start", "end", "start_timestamp", "end_timestamp",
    "segment_no", "start_idx", "end_idx", "start_boundary_probability",
    "end_boundary_probability", "boundary_probability", "boundary_uncertainty",
    "max_probability", "posterior_entropy", "model_disagreement", "density_score",
    "density_is_low", "recognizability", "is_unknown", "context_conflict",
    "label", "hard_label", "semantic_label", "review_status", "review_note",
}


def infer_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> list[str]:
    """Return finite numeric columns suitable for unsupervised fitting.

    Explicit columns are validated.  With inference, identifiers and already-created
    model output columns are excluded so calling ``fit_predict`` repeatedly is safe.
    """
    if feature_columns is not None:
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {sorted(missing)}")
        cols = list(feature_columns)
    else:
        cols = [
            c for c in frame.select_dtypes(include=np.number).columns
            if c not in _NON_FEATURES and not c.startswith(("prob_", "ensemble_", "gate_", "gmm_"))
        ]
    if not cols:
        raise ValueError("No numeric segment features available for the model")
    return cols


class SegmentSoftClassifier:
    """A seed ensemble of GMMs with component alignment and abstention gates.

    Parameters mirror :class:`ModelConfig`.  Input rows are one segment each.  The
    optional columns ``boundary_uncertainty`` / ``boundary_probability`` and
    ``context_conflict`` / ``multi_period_conflict`` are consumed by the gates when
    present; they are not required.
    """

    def __init__(self, config: ModelConfig | None = None, feature_columns: Sequence[str] | None = None):
        self.config = config or ModelConfig()
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.scaler = RobustScaler()
        self.models: list[GaussianMixture] = []
        self.component_labels: list[str] = []
        self.profile_suggestions: list[str] = []
        self.component_valid_: list[bool] = []
        self.density_floor_: float = float("nan")
        self._reference_means: np.ndarray | None = None

    def _matrix(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cols = infer_feature_columns(frame, self.feature_columns)
        if self.feature_columns is None:
            self.feature_columns = cols
        values = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        medians = values.median(numeric_only=True).fillna(0.0)
        values = values.fillna(medians).fillna(0.0)
        return self.scaler.fit_transform(values) if fit else self.scaler.transform(values)

    def fit(self, segments: pd.DataFrame) -> SegmentSoftClassifier:
        if len(segments) < 2:
            raise ValueError("At least two segments are required to fit a soft classifier")
        X = self._matrix(segments, fit=True)
        n_components = min(self.config.n_components, len(segments), max(1, len(np.unique(X, axis=0))))
        self.models = []
        for seed in range(self.config.ensemble_size):
            model = GaussianMixture(
                n_components=n_components, covariance_type="full", reg_covar=1e-5,
                n_init=2, random_state=self.config.random_state + seed,
            ).fit(X)
            self.models.append(model)
        # The first model gives a stable output component order.  Other model
        # posteriors are matched to it by scaled mean distance (Hungarian solve).
        self._reference_means = self.models[0].means_.copy()
        scores = np.mean([m.score_samples(X) for m in self.models], axis=0)
        self.density_floor_ = float(np.quantile(scores, self.config.min_density_quantile))
        reference_probability = self.models[0].predict_proba(X)
        self.component_labels = [f"CLUSTER_{chr(65 + i)}" for i in range(n_components)]
        self.profile_suggestions = self._semantic_labels(segments, reference_probability)
        hard = reference_probability.argmax(axis=1)
        years = pd.to_datetime(segments.get("start", pd.Series(pd.NaT, index=segments.index)), errors="coerce").dt.year
        self.component_valid_ = []
        for i in range(n_components):
            mask = hard == i
            sample_ok = int(mask.sum()) >= self.config.min_cluster_samples
            symbol_ok = segments.loc[mask, "symbol"].nunique() >= self.config.min_cluster_symbols if "symbol" in segments else True
            year_ok = years[mask].nunique() >= self.config.min_cluster_years if years.notna().any() else True
            self.component_valid_.append(bool(sample_ok and symbol_ok and year_ok))
        return self

    def _semantic_labels(self, frame: pd.DataFrame, probabilities: np.ndarray) -> list[str]:
        """Map arbitrary GMM ids to human-oriented descriptive names."""
        ret_col = next((c for c in ("cumulative_return", "return", "segment_return", "slope") if c in frame), None)
        vol_col = next((c for c in ("volatility", "atr", "amplitude") if c in frame), None)
        labels: list[str] = []
        returns = frame[ret_col].to_numpy(float) if ret_col else np.zeros(len(frame))
        scale = np.nanmedian(np.abs(returns)) or 1e-9
        for i in range(probabilities.shape[1]):
            weights = probabilities[:, i]
            if float(weights.sum()) <= 1e-12:
                mean_return = float(np.nanmedian(returns))
                vol = float(np.nanmedian(frame[vol_col].to_numpy(float))) if vol_col else 0.0
            else:
                mean_return = float(np.average(returns, weights=weights))
                vol = float(np.average(frame[vol_col].to_numpy(float), weights=weights)) if vol_col else 0.0
            if mean_return > 0.35 * scale:
                base = "UPTREND"
            elif mean_return < -0.35 * scale:
                base = "DOWNTREND"
            elif abs(mean_return) < 0.12 * scale and vol > 0:
                base = "RANGE"
            else:
                base = "TRANSITION"
            labels.append(f"{base}_{i + 1}")
        return labels

    def _aligned_proba(self, model: GaussianMixture, X: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X)
        assert self._reference_means is not None
        distance = ((model.means_[:, None, :] - self._reference_means[None, :, :]) ** 2).sum(axis=2)
        source, target = linear_sum_assignment(distance)
        aligned = np.zeros_like(proba)
        aligned[:, target] = proba[:, source]
        return aligned

    def predict(self, segments: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit before predict")
        X = self._matrix(segments)
        members = np.stack([self._aligned_proba(m, X) for m in self.models])
        probabilities = members.mean(axis=0)
        disagreement = members.std(axis=0).mean(axis=1)
        score = np.mean([m.score_samples(X) for m in self.models], axis=0)
        result = segments.copy()
        for i, name in enumerate(self.component_labels):
            result[f"prob_{name}"] = probabilities[:, i]
        result["soft_label"] = [dict(zip(self.component_labels, row)) for row in probabilities]
        result["max_probability"] = probabilities.max(axis=1)
        if probabilities.shape[1] == 1:
            result["posterior_entropy"] = 0.0
        else:
            result["posterior_entropy"] = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1) / np.log(probabilities.shape[1])
        result["model_disagreement"] = disagreement
        result["density_score"] = score
        result["density_is_low"] = score < self.density_floor_
        result["boundary_uncertainty"] = self._boundary_uncertainty(result)
        result["context_conflict"] = self._context_conflict(result)
        result["recognizability"] = self._recognizability(result)
        gates = pd.DataFrame({
            "low_posterior": result.max_probability < self.config.min_max_probability,
            "high_entropy": result.posterior_entropy > self.config.max_normalized_entropy,
            "model_disagreement": result.model_disagreement > self.config.max_disagreement,
            "low_density": result.density_is_low,
            "boundary_unstable": result.boundary_uncertainty > self.config.max_boundary_uncertainty,
            "context_conflict": result.context_conflict > self.config.max_context_conflict,
            "low_recognizability": result.recognizability < self.config.min_recognizability,
        })
        candidate_index = probabilities.argmax(axis=1)
        candidate = np.asarray(self.component_labels, object)[candidate_index]
        result["candidate_label"] = candidate
        result["suggested_semantic_label"] = np.asarray(self.profile_suggestions, object)[candidate_index]
        result["cluster_support_valid"] = np.asarray(self.component_valid_, bool)[candidate_index]
        gates["insufficient_cluster_support"] = ~result["cluster_support_valid"]
        result["unknown_reason"] = gates.apply(lambda r: ";".join(r.index[r].tolist()), axis=1)
        result["label"] = np.where(result.unknown_reason.eq(""), candidate, UNKNOWN)
        result["is_unknown"] = result.label.eq(UNKNOWN)
        return result

    def _boundary_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        if "boundary_uncertainty" in frame and frame["boundary_uncertainty"].notna().any():
            return frame["boundary_uncertainty"].fillna(1.0).clip(0, 1).to_numpy(float)
        for col in ("boundary_probability", "start_boundary_probability", "end_boundary_probability"):
            if col in frame:
                return (1 - frame[col].fillna(0).clip(0, 1)).to_numpy(float)
        return np.zeros(len(frame))

    def _context_conflict(self, frame: pd.DataFrame) -> np.ndarray:
        for col in ("context_conflict", "multi_period_conflict", "multiperiod_conflict"):
            if col in frame:
                return frame[col].fillna(0).clip(0, 1).to_numpy(float)
        context_columns = [
            c for c in frame.select_dtypes(include=np.number).columns
            if c.startswith("multitf_") and any(token in c.lower() for token in ("return", "slope", "trend"))
        ]
        if context_columns and "cumulative_return" in frame:
            segment_sign = np.sign(frame["cumulative_return"].fillna(0).to_numpy(float))[:, None]
            context_sign = np.sign(frame[context_columns].fillna(0).to_numpy(float))
            valid = context_sign != 0
            conflicts = (context_sign != segment_sign) & valid & (segment_sign != 0)
            return (conflicts.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)).clip(0, 1)
        return np.zeros(len(frame))

    def _recognizability(self, frame: pd.DataFrame) -> np.ndarray:
        # Smooth evidence score: higher posterior and density, lower uncertainty.
        density = (frame.density_score >= self.density_floor_).astype(float)
        return (0.40 * frame.max_probability + 0.25 * (1 - frame.posterior_entropy)
                + 0.15 * (1 - np.minimum(frame.model_disagreement / max(self.config.max_disagreement, 1e-9), 1))
                + 0.10 * density + 0.05 * (1 - frame.boundary_uncertainty)
                + 0.05 * (1 - frame.context_conflict)).clip(0, 1).to_numpy(float)

    def fit_predict(self, segments: pd.DataFrame) -> pd.DataFrame:
        return self.fit(segments).predict(segments)

    def metadata(self) -> dict:
        return {"config": asdict(self.config), "feature_columns": self.feature_columns,
                "component_labels": self.component_labels, "profile_suggestions": self.profile_suggestions,
                "component_valid": self.component_valid_, "density_floor": self.density_floor_}


# Short aliases make the public API forgiving for notebooks and pipeline code.
ProbabilisticSegmentModel = SegmentSoftClassifier
ProbabilisticWaveModel = SegmentSoftClassifier
classify_segments = lambda segments, config=None, feature_columns=None: SegmentSoftClassifier(config, feature_columns).fit_predict(segments)
