"""Multi-model, *non-semantic* discovery of segment clusters.

This module deliberately stops before business labels.  It is intended to make
candidate clusters and their uncertainty available to the later human-labelling
and supervised-classification stages, not to predict returns or select stocks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import RobustScaler

from .model import UNKNOWN, infer_feature_columns


_FORWARD_TOKENS = ("future", "forward", "target", "label", "return_5d", "return_20d", "return_60d")


@dataclass
class DiscoveryConfig:
    """Parameters for unsupervised discovery, separate from classifier gates."""

    n_components: int = 5
    random_state: int = 42
    ensemble_size: int = 5
    min_cluster_samples: int = 5
    min_cluster_symbols: int = 2
    min_cluster_years: int = 1
    min_density_quantile: float = 0.04
    max_disagreement: float = 0.25
    min_consensus_probability: float = 0.50
    max_normalized_entropy: float = 0.78
    max_boundary_uncertainty: float = 0.44
    max_context_conflict: float = 0.75
    min_recognizability: float = 0.48
    min_hdbscan_strength: float = 0.30
    include_hdbscan: bool = True
    max_iter: int = 100


def infer_discovery_features(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> list[str]:
    """Select descriptive numeric fields and reject any look-ahead fields.

    The rejection also applies to explicit fields: discovery must not quietly use
    future return targets supplied by a selection-study table.
    """
    columns = infer_feature_columns(frame, feature_columns)
    forbidden = [c for c in columns if any(token in c.lower() for token in _FORWARD_TOKENS)]
    if forbidden:
        if feature_columns is not None:
            raise ValueError(f"Look-ahead fields cannot be used for discovery: {forbidden}")
        columns = [c for c in columns if c not in forbidden]
    if not columns:
        raise ValueError("No non-look-ahead numeric discovery features available")
    return columns


class MultiModelDiscoverer:
    """Align GMM and DPGMM memberships to stable ``CLUSTER_A``-style ids.

    ``fit_predict`` returns per-model probabilities, candidates, disagreement,
    density/noise diagnostics and a conservative ``UNKNOWN`` decision.  Cluster
    ids are arbitrary temporary identifiers; ``profile_suggestions`` are numeric
    summaries only and must not be treated as semantic labels.
    """

    def __init__(self, config: DiscoveryConfig | None = None, feature_columns: Sequence[str] | None = None):
        self.config = config or DiscoveryConfig()
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.scaler = RobustScaler()
        self.medians_: pd.Series | None = None
        self.gmm_: GaussianMixture | None = None
        self.gmms_: list[GaussianMixture] = []
        self.dpgmm_: BayesianGaussianMixture | None = None
        self.hdbscan_: object | None = None
        self.hdbscan_cluster_map_: dict[int, int] = {}
        self.reference_means_: np.ndarray | None = None
        self.cluster_labels_: list[str] = []
        self.cluster_valid_: list[bool] = []
        self.gmm_converged_: bool = False
        self.dpgmm_converged_: bool = False
        self.density_floor_: float = float("nan")
        self.profile_suggestions_: pd.DataFrame = pd.DataFrame()

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        columns = infer_discovery_features(frame, self.feature_columns)
        if self.feature_columns is None:
            self.feature_columns = columns
        elif list(columns) != list(self.feature_columns):
            # Explicit features have already been validated. On prediction require
            # exactly the trained schema rather than silently changing it.
            missing = set(self.feature_columns) - set(frame.columns)
            if missing:
                raise ValueError(f"Prediction frame is missing features: {sorted(missing)}")
        values = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        if fit:
            self.medians_ = values.median().fillna(0.0)
        assert self.medians_ is not None
        values = values.fillna(self.medians_).fillna(0.0)
        return self.scaler.fit_transform(values) if fit else self.scaler.transform(values)

    def fit(self, segments: pd.DataFrame) -> "MultiModelDiscoverer":
        if len(segments) < 2:
            raise ValueError("At least two segments are required for discovery")
        X = self._matrix(segments, fit=True)
        n_unique = max(1, len(np.unique(X, axis=0)))
        k = min(self.config.n_components, len(X), n_unique)
        self.gmm_ = GaussianMixture(k, covariance_type="diag", n_init=3, reg_covar=1e-4,
                                    max_iter=self.config.max_iter,
                                    random_state=self.config.random_state).fit(X)
        self.gmms_ = [self.gmm_]
        rng = np.random.default_rng(self.config.random_state)
        for member in range(1, max(1, self.config.ensemble_size)):
            sample = rng.integers(0, len(X), size=len(X))
            member_k = min(k, max(1, len(np.unique(X[sample], axis=0))))
            candidate = GaussianMixture(
                member_k, covariance_type="diag", n_init=2, reg_covar=1e-4,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state + member,
            ).fit(X[sample])
            self.gmms_.append(candidate)
        self.gmm_converged_ = all(bool(model.converged_) for model in self.gmms_)
        # Bayesian mixture is a finite truncation of a Dirichlet-process mixture.
        self.dpgmm_ = BayesianGaussianMixture(
            n_components=k, covariance_type="diag", covariance_prior=np.ones(X.shape[1]) * 1e-3,
            weight_concentration_prior_type="dirichlet_process", n_init=2,
            reg_covar=1e-5, max_iter=self.config.max_iter, random_state=self.config.random_state + 1,
        ).fit(X)
        self.dpgmm_converged_ = bool(self.dpgmm_.converged_)
        # Deterministic temporary ids within this fitted model: order reference
        # components by scaled cumulative return (then duration when available).
        # Names are not stable semantic identities across walk-forward refits.
        anchors = [c for c in ("duration_bars", "cumulative_return") if c in self.feature_columns]
        keys = tuple(self.gmm_.means_[:, self.feature_columns.index(c)] for c in anchors)
        order = np.lexsort(keys) if keys else np.arange(k)
        self.reference_means_ = self.gmm_.means_[order].copy()
        self.cluster_labels_ = [f"CLUSTER_{chr(65 + i)}" for i in range(k)]
        self.density_floor_ = float(np.quantile(self.gmm_.score_samples(X), self.config.min_density_quantile))
        self._fit_hdbscan(X)
        base_probabilities = self._align(self.gmm_, X)
        self._cluster_constraints(segments, base_probabilities.argmax(axis=1))
        self.profile_suggestions_ = self._profiles(segments, base_probabilities)
        return self

    def _fit_hdbscan(self, X: np.ndarray) -> None:
        self.hdbscan_ = None
        if not self.config.include_hdbscan:
            return
        try:
            import hdbscan  # type: ignore[import-not-found]
        except ImportError:
            return
        min_size = max(2, min(self.config.min_cluster_samples, len(X)))
        self.hdbscan_ = hdbscan.HDBSCAN(min_cluster_size=min_size, prediction_data=True).fit(X)
        self.hdbscan_cluster_map_ = {}
        labels = np.asarray(self.hdbscan_.labels_)
        for cluster in np.unique(labels[labels >= 0]):
            centroid = X[labels == cluster].mean(axis=0)
            self.hdbscan_cluster_map_[int(cluster)] = int(
                np.argmin(((self.reference_means_ - centroid) ** 2).sum(axis=1))
            )

    def _cluster_constraints(self, frame: pd.DataFrame, hard: np.ndarray) -> None:
        years = pd.to_datetime(frame.get("start", pd.Series(pd.NaT, index=frame.index)), errors="coerce").dt.year
        self.cluster_valid_ = []
        for i in range(len(self.cluster_labels_)):
            mask = hard == i
            enough_rows = int(mask.sum()) >= self.config.min_cluster_samples
            # Missing provenance is not evidence of broad support. In
            # particular, never let absent symbol/date columns silently pass
            # cross-sectional or cross-year release gates.
            symbol_count = frame.loc[mask, "symbol"].nunique() if "symbol" in frame else 0
            enough_symbols = symbol_count >= self.config.min_cluster_symbols
            enough_years = years[mask].nunique() >= self.config.min_cluster_years
            self.cluster_valid_.append(bool(enough_rows and enough_symbols and enough_years))

    def _profiles(self, frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
        numeric = [c for c in ("cumulative_return", "return_volatility", "amplitude", "duration_bars", "max_drawdown") if c in frame]
        rows = []
        for i, label in enumerate(self.cluster_labels_):
            weights = probabilities[:, i]
            row: dict[str, object] = {"cluster_label": label, "sample_support_valid": self.cluster_valid_[i],
                                      "effective_samples": round(float(weights.sum()), 2)}
            for col in numeric:
                row[f"weighted_{col}"] = float(np.average(frame[col].fillna(frame[col].median()).fillna(0), weights=weights))
            rows.append(row)
        return pd.DataFrame(rows)

    def _align(self, model: GaussianMixture | BayesianGaussianMixture, X: np.ndarray) -> np.ndarray:
        assert self.reference_means_ is not None
        raw = model.predict_proba(X)
        distances = ((model.means_[:, None, :] - self.reference_means_[None, :, :]) ** 2).sum(axis=2)
        source, target = linear_sum_assignment(distances)
        aligned = np.zeros((len(X), len(self.cluster_labels_)))
        aligned[:, target] = raw[:, source]
        # Both models have the same truncation but retain this normalization for
        # a robust public invariant should sklearn change sparse components.
        return aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)

    def _hdbscan_probabilities(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map optional HDBSCAN clusters to reference ids; noise becomes UNKNOWN."""
        n, k = len(X), len(self.cluster_labels_)
        if self.hdbscan_ is None:
            return np.empty((0, n, k)), np.zeros(n, dtype=bool), np.zeros(n)
        # labels_ applies only to the fitting rows. approximate_predict preserves
        # the HDBSCAN prediction semantics for a later pipeline batch.
        import hdbscan  # type: ignore[import-not-found]
        labels, strengths = hdbscan.approximate_predict(self.hdbscan_, X)
        labels, strengths = np.asarray(labels), np.asarray(strengths, dtype=float)
        noise = labels < 0
        proba = np.zeros((n, k))
        for cluster in np.unique(labels[~noise]):
            members = labels == cluster
            nearest = self.hdbscan_cluster_map_.get(int(cluster))
            if nearest is not None:
                # Preserve membership strength: unexplained mass remains diffuse.
                proba[members] = ((1 - strengths[members]) / k)[:, None]
                proba[members, nearest] += strengths[members]
        empty = proba.sum(axis=1) == 0
        proba[empty] = 1.0 / k
        proba /= proba.sum(axis=1, keepdims=True)
        return proba[None, :, :], noise, strengths

    def predict(self, segments: pd.DataFrame) -> pd.DataFrame:
        if self.gmm_ is None or self.dpgmm_ is None:
            raise RuntimeError("Call fit before predict")
        X = self._matrix(segments, fit=False)
        gmm_members = np.stack([self._align(model, X) for model in self.gmms_])
        gmm = gmm_members.mean(axis=0)
        dpgmm = self._align(self.dpgmm_, X)
        hdb, noise, hdb_strength = self._hdbscan_probabilities(X)
        members = np.concatenate([gmm_members, dpgmm[None, :, :], hdb], axis=0)
        ensemble = members.mean(axis=0)
        candidate_idx = ensemble.argmax(axis=1)
        result = segments.copy()
        for prefix, values in (("gmm", gmm), ("dpgmm", dpgmm), ("ensemble", ensemble)):
            for i, name in enumerate(self.cluster_labels_):
                result[f"{prefix}_prob_{name}"] = values[:, i]
            result[f"{prefix}_candidate"] = np.asarray(self.cluster_labels_, dtype=object)[values.argmax(axis=1)]
        # Backward-compatible fields make this discoverer interchangeable with
        # SegmentSoftClassifier in the existing pipeline.
        for i, name in enumerate(self.cluster_labels_):
            result[f"prob_{name}"] = ensemble[:, i]
        if self.hdbscan_ is not None:
            for i, name in enumerate(self.cluster_labels_):
                result[f"hdbscan_prob_{name}"] = hdb[0, :, i]
            result["hdbscan_candidate"] = np.where(noise, UNKNOWN, np.asarray(self.cluster_labels_, dtype=object)[hdb[0].argmax(axis=1)])
            result["hdbscan_membership_strength"] = hdb_strength
            result["hdbscan_available"] = True
            result["hdbscan_noise"] = noise
        else:
            # No installed model is not evidence that HDBSCAN found no noise.
            result["hdbscan_available"] = False
            result["hdbscan_noise"] = pd.array([pd.NA] * len(result), dtype="boolean")
        result["ensemble_soft_label"] = [dict(zip(self.cluster_labels_, row)) for row in ensemble]
        result["soft_label"] = result["ensemble_soft_label"]
        result["candidate_label"] = np.asarray(self.cluster_labels_, dtype=object)[candidate_idx]
        result["max_probability"] = ensemble.max(axis=1)
        result["posterior_entropy"] = (-(ensemble * np.log(np.clip(ensemble, 1e-12, 1))).sum(axis=1)
                                      / np.log(ensemble.shape[1]) if ensemble.shape[1] > 1 else np.zeros(len(ensemble)))
        result["model_disagreement"] = members.std(axis=0).mean(axis=1)
        result["density_score"] = self.gmm_.score_samples(X)
        result["density_is_low"] = result["density_score"] < self.density_floor_
        result["gmm_converged"] = self.gmm_converged_
        result["dpgmm_converged"] = self.dpgmm_converged_
        result["mixture_convergence_valid"] = self.gmm_converged_ and self.dpgmm_converged_
        result["boundary_uncertainty"] = self._boundary_uncertainty(result)
        result["boundary_unstable"] = result["boundary_uncertainty"] > self.config.max_boundary_uncertainty
        result["context_conflict"] = self._context_conflict(result)
        result["recognizability"] = self._recognizability(result)
        result["cluster_support_valid"] = np.asarray(self.cluster_valid_, dtype=bool)[candidate_idx]
        gates = pd.DataFrame({
            "low_consensus": result["max_probability"] < self.config.min_consensus_probability,
            "model_disagreement": result["model_disagreement"] > self.config.max_disagreement,
            "mixture_nonconverged": not (self.gmm_converged_ and self.dpgmm_converged_),
            "low_density": result["density_is_low"],
            "hdbscan_noise": noise if self.hdbscan_ is not None else False,
            "weak_hdbscan_membership": (hdb_strength < self.config.min_hdbscan_strength) & ~noise if self.hdbscan_ is not None else False,
            "high_entropy": result["posterior_entropy"] > self.config.max_normalized_entropy,
            "boundary_unstable": result["boundary_unstable"],
            "context_conflict": result["context_conflict"] > self.config.max_context_conflict,
            "low_recognizability": result["recognizability"] < self.config.min_recognizability,
            "insufficient_cluster_support": ~result["cluster_support_valid"],
        })
        result["unknown_reason"] = gates.apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
        result["label"] = np.where(result["unknown_reason"].eq(""), result["candidate_label"], UNKNOWN)
        result["is_unknown"] = result["label"].eq(UNKNOWN)
        result["candidate_status"] = np.where(result["is_unknown"], "model_abstention", "classified")
        return result

    def _boundary_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        if "boundary_uncertainty" in frame and frame["boundary_uncertainty"].notna().any():
            return frame["boundary_uncertainty"].fillna(1.0).clip(0, 1).to_numpy(float)
        for column in ("boundary_probability", "start_boundary_probability", "end_boundary_probability"):
            if column in frame:
                return (1 - frame[column].fillna(0).clip(0, 1)).to_numpy(float)
        return np.zeros(len(frame))

    def _context_conflict(self, frame: pd.DataFrame) -> np.ndarray:
        for column in ("context_conflict", "multi_period_conflict", "multiperiod_conflict"):
            if column in frame:
                return frame[column].fillna(0).clip(0, 1).to_numpy(float)
        context = [c for c in frame.select_dtypes(include=np.number) if c.startswith("multitf_")
                   and any(t in c.lower() for t in ("return", "slope", "trend"))]
        if context and "cumulative_return" in frame:
            sign = np.sign(frame["cumulative_return"].fillna(0).to_numpy(float))[:, None]
            context_sign = np.sign(frame[context].fillna(0).to_numpy(float))
            valid = context_sign != 0
            return (((context_sign != sign) & valid & (sign != 0)).sum(axis=1)
                    / np.maximum(valid.sum(axis=1), 1)).clip(0, 1)
        return np.zeros(len(frame))

    def _recognizability(self, frame: pd.DataFrame) -> np.ndarray:
        density = (frame["density_score"] >= self.density_floor_).astype(float)
        return (0.40 * frame["max_probability"] + 0.25 * (1 - frame["posterior_entropy"])
                + 0.15 * (1 - np.minimum(frame["model_disagreement"] / max(self.config.max_disagreement, 1e-9), 1))
                + 0.10 * density + 0.05 * (1 - frame["boundary_uncertainty"])
                + 0.05 * (1 - frame["context_conflict"])).clip(0, 1).to_numpy(float)

    def fit_predict(self, segments: pd.DataFrame) -> pd.DataFrame:
        return self.fit(segments).predict(segments)

    def metadata(self) -> dict[str, object]:
        return {"config": asdict(self.config), "feature_columns": self.feature_columns,
                "cluster_labels": self.cluster_labels_, "cluster_valid": self.cluster_valid_,
                "gmm_converged": self.gmm_converged_, "dpgmm_converged": self.dpgmm_converged_,
                "density_floor": self.density_floor_, "hdbscan_available": self.hdbscan_ is not None,
                "profile_suggestions": self.profile_suggestions_.to_dict("records")}


discover_segments = lambda segments, config=None, feature_columns=None: MultiModelDiscoverer(config, feature_columns).fit_predict(segments)


def walk_forward_discovery_states(
    segments: pd.DataFrame,
    *,
    config: DiscoveryConfig | None = None,
    feature_columns: Sequence[str] | None = None,
    min_train_rows: int = 100,
    retrain_every: int = 20,
    availability_col: str = "available_at",
) -> pd.DataFrame:
    """Generate past-only states from explicitly finalized segment candidates.

    ``available_at`` must be the timestamp at which the segmentation boundary
    and all segment features were actually knowable; it is never inferred from
    the pivot/end date. Fit rows must be finalized strictly earlier. Bar-range
    overlap purging is scoped to the same symbol: simultaneous bars in a
    different security are not the same observations. This function cannot
    make retrospectively segmented candidates causal by assigning ``available_at=end``.
    """
    required = {"segment_id", "symbol", "start", "end", "start_idx", "end_idx", availability_col}
    if missing := required - set(segments.columns):
        raise ValueError(f"segments missing causal walk-forward columns: {sorted(missing)}")
    if min_train_rows < 2 or retrain_every < 1:
        raise ValueError("min_train_rows must be >=2 and retrain_every positive")
    data = segments.copy()
    data["end"] = pd.to_datetime(data["end"])
    data["start"] = pd.to_datetime(data["start"])
    data[availability_col] = pd.to_datetime(data[availability_col], errors="coerce")
    if data[availability_col].isna().any():
        raise ValueError(f"{availability_col} must be present and parseable for every segment")
    if (data[availability_col] < data["end"]).any():
        raise ValueError(f"{availability_col} cannot precede the segment end")
    if data["segment_id"].astype(str).duplicated().any():
        raise ValueError("segment_id must be unique for causal walk-forward discovery")
    data = data.sort_values([availability_col, "symbol", "segment_id"]).reset_index(drop=True)
    dates = pd.Index(data[availability_col].drop_duplicates().sort_values())
    outputs: list[pd.DataFrame] = []
    fitted: MultiModelDiscoverer | None = None
    trained_at: pd.Timestamp | None = None
    trained_rows = 0
    train_max_end_by_symbol: dict[object, pd.Timestamp] = {}
    train_max_available_at: pd.Timestamp | None = None
    for date_index, decision_at in enumerate(dates):
        train = data[data[availability_col] < decision_at]
        if len(train) >= min_train_rows and (fitted is None or date_index % retrain_every == 0):
            fitted = MultiModelDiscoverer(config, feature_columns).fit(train)
            trained_at, trained_rows = decision_at, len(train)
            train_max_end_by_symbol = train.groupby("symbol")["end"].max().to_dict()
            train_max_available_at = train[availability_col].max()
        current = data[data[availability_col].eq(decision_at)].copy()
        current["_wf_row_order"] = current.index
        prior_end = pd.to_datetime(current["symbol"].map(train_max_end_by_symbol))
        scoreable = prior_end.isna() | (current["start"] > prior_end)
        if fitted is None:
            predicted = current[[c for c in current if c != "_wf_row_order"]].copy()
            predicted["label"] = UNKNOWN
            predicted["is_unknown"] = True
            predicted["unknown_reason"] = "insufficient_history"
            predicted["candidate_status"] = "model_abstention"
        else:
            # Purge intervals only within a symbol. Other securities' bars on
            # the same dates are distinct observations and remain valid train data.
            predicted_parts = []
            if scoreable.any():
                predicted_parts.append(fitted.predict(current.loc[scoreable].drop(columns="_wf_row_order")))
            if (~scoreable).any():
                withheld = current.loc[~scoreable].drop(columns="_wf_row_order").copy()
                withheld["label"] = UNKNOWN
                withheld["is_unknown"] = True
                withheld["unknown_reason"] = "purged_overlap"
                withheld["candidate_status"] = "model_abstention"
                predicted_parts.append(withheld)
            predicted = pd.concat(predicted_parts, ignore_index=False).sort_index()
        predicted["state_available_at"] = decision_at
        predicted["oos_model_trained_at"] = trained_at
        predicted["oos_train_rows"] = trained_rows
        predicted["oos_train_max_end"] = pd.to_datetime(predicted["symbol"].map(train_max_end_by_symbol))
        predicted["oos_train_max_available_at"] = train_max_available_at
        outputs.append(predicted)
    # Drop per-block all-NA columns before concat, then restore mandatory
    # provenance so an all-UNKNOWN run remains auditable without pandas dtype
    # ambiguity warnings.
    compact = [frame.dropna(axis=1, how="all") for frame in outputs]
    result = pd.concat(compact, ignore_index=True)
    for column in ("oos_model_trained_at", "oos_train_max_end", "oos_train_max_available_at"):
        if column not in result:
            result[column] = pd.NaT
    return result.sort_values(
        [availability_col, "symbol", "segment_id"]
    ).reset_index(drop=True)
