from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wave_segments.discovery import (
    DiscoveryConfig, MultiModelDiscoverer, infer_discovery_features,
    walk_forward_discovery_states,
)


def _segments(n: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n):
        group = i % 3
        rows.append({"segment_id": f"S{i}", "symbol": f"T{i % 3}",
                     "start": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i * 20),
                     "end": pd.Timestamp("2022-01-05") + pd.Timedelta(days=i * 20),
                     "cumulative_return": (group - 1) * .05 + rng.normal(0, .005),
                     "return_volatility": .01 + group * .01 + rng.random() * .002,
                     "duration_bars": 5 + group * 5, "amplitude": .03 + group * .02,
                     "future_return_20d": rng.normal(), "target_return": rng.normal()})
    return pd.DataFrame(rows)


def test_discovery_outputs_normalised_probabilities_and_temporary_ids():
    data = _segments()
    model = MultiModelDiscoverer(DiscoveryConfig(n_components=3, min_cluster_samples=2, min_cluster_symbols=1))
    output = model.fit_predict(data)
    assert len(model.gmms_) == model.config.ensemble_size
    for prefix in ("gmm_prob_", "dpgmm_prob_", "ensemble_prob_"):
        columns = [c for c in output if c.startswith(prefix)]
        assert np.allclose(output[columns].sum(axis=1), 1.0)
    assert set(output.candidate_label).issubset({"CLUSTER_A", "CLUSTER_B", "CLUSTER_C"})
    assert {"model_disagreement", "density_is_low", "unknown_reason", "is_unknown",
            "soft_label", "posterior_entropy", "recognizability", "boundary_unstable"} <= set(output)
    assert (output["candidate_status"].eq("model_abstention") == output["is_unknown"]).all()
    legacy_probabilities = [c for c in output if c.startswith("prob_")]
    assert np.allclose(output[legacy_probabilities].sum(axis=1), 1.0)
    assert "future_return_20d" not in model.feature_columns
    assert "target_return" not in model.feature_columns
    assert set(model.profile_suggestions_.columns) >= {"cluster_label", "effective_samples"}


def test_unknown_from_cluster_constraints_and_explicit_future_feature_rejected():
    data = _segments(9).assign(symbol="ONLY", start=pd.Timestamp("2024-01-01"))
    model = MultiModelDiscoverer(DiscoveryConfig(n_components=3, min_cluster_samples=20, min_cluster_symbols=2))
    output = model.fit_predict(data)
    assert output.is_unknown.all()
    with pytest.raises(ValueError, match="Look-ahead"):
        infer_discovery_features(data, ["cumulative_return", "future_return_20d"])


def test_unavailable_hdbscan_is_not_reported_as_zero_noise():
    data = _segments(36)
    config = DiscoveryConfig(
        n_components=3, ensemble_size=1, min_cluster_samples=2,
        min_cluster_symbols=1, include_hdbscan=False,
    )
    model = MultiModelDiscoverer(config).fit(data)
    output = model.predict(data)
    assert output["hdbscan_available"].eq(False).all()
    assert output["hdbscan_noise"].isna().all()
    assert model.metadata()["hdbscan_available"] is False


def test_nonconverged_gmm_ensemble_is_a_first_class_unknown_gate(monkeypatch):
    from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

    class NonConvergingGMM(SklearnGaussianMixture):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.converged_ = False
            return self

    monkeypatch.setattr("wave_segments.discovery.GaussianMixture", NonConvergingGMM)
    data = _segments(36)
    config = DiscoveryConfig(
        n_components=3, ensemble_size=2, min_cluster_samples=2,
        min_cluster_symbols=1, include_hdbscan=False,
    )
    output = MultiModelDiscoverer(config).fit_predict(data)
    assert output["gmm_converged"].eq(False).all()
    assert output["mixture_convergence_valid"].eq(False).all()
    assert output["label"].eq("UNKNOWN").all()
    assert output["unknown_reason"].str.contains("mixture_nonconverged").all()


def test_nonconverged_dpgmm_is_a_first_class_unknown_gate(monkeypatch):
    from sklearn.mixture import BayesianGaussianMixture as SklearnBayesianGaussianMixture

    class NonConvergingDPGMM(SklearnBayesianGaussianMixture):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.converged_ = False
            return self

    monkeypatch.setattr("wave_segments.discovery.BayesianGaussianMixture", NonConvergingDPGMM)
    data = _segments(36)
    config = DiscoveryConfig(
        n_components=3, ensemble_size=2, min_cluster_samples=2,
        min_cluster_symbols=1, include_hdbscan=False,
    )
    output = MultiModelDiscoverer(config).fit_predict(data)
    assert output["dpgmm_converged"].eq(False).all()
    assert output["mixture_convergence_valid"].eq(False).all()
    assert output["label"].eq("UNKNOWN").all()
    assert output["unknown_reason"].str.contains("mixture_nonconverged").all()


@pytest.mark.parametrize("missing_column", ["symbol", "start"])
def test_missing_provenance_cannot_satisfy_cluster_support_gates(missing_column):
    data = _segments(36).drop(columns=missing_column)
    config = DiscoveryConfig(
        n_components=3, ensemble_size=1, min_cluster_samples=2,
        min_cluster_symbols=2, min_cluster_years=2, include_hdbscan=False,
    )
    model = MultiModelDiscoverer(config).fit(data)
    assert not any(model.cluster_valid_)
    assert model.predict(data).is_unknown.all()


def test_walk_forward_discovery_never_fits_on_current_or_future_segments():
    data = _segments(36)
    data["start_idx"] = np.arange(len(data)) * 20
    data["end_idx"] = data["start_idx"] + 4
    # Confirmation happens after the pivot/end; the API must use this explicit
    # availability time rather than treating a retrospective end date as known.
    data["available_at"] = data["end"] + pd.Timedelta(days=3)
    config = DiscoveryConfig(
        n_components=3, ensemble_size=2, min_cluster_samples=2,
        min_cluster_symbols=1, include_hdbscan=False,
    )
    output = walk_forward_discovery_states(
        data, config=config,
        feature_columns=["cumulative_return", "return_volatility", "duration_bars", "amplitude"],
        min_train_rows=12, retrain_every=2,
    )
    scored = output[output["oos_train_rows"].gt(0)]
    assert (scored["oos_train_max_available_at"] < scored["state_available_at"]).all()
    assert (scored["oos_train_max_end"] < scored["start"]).all()
    assert (scored["available_at"] >= scored["end"]).all()
    assert output.label.eq("UNKNOWN").any()


def test_walk_forward_discovery_requires_explicit_valid_availability_time():
    data = _segments(12)
    data["start_idx"] = np.arange(len(data)) * 20
    data["end_idx"] = data["start_idx"] + 4
    config = DiscoveryConfig(n_components=3, ensemble_size=1, include_hdbscan=False)
    with pytest.raises(ValueError, match="available_at"):
        walk_forward_discovery_states(data, config=config, min_train_rows=4)

    data["available_at"] = data["end"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="cannot precede"):
        walk_forward_discovery_states(data, config=config, min_train_rows=4)

    data["available_at"] = data["end"] + pd.Timedelta(days=2)
    output = walk_forward_discovery_states(data, config=config, min_train_rows=100)
    assert output.label.eq("UNKNOWN").all()
    assert output.oos_model_trained_at.isna().all()
    assert output.oos_train_max_available_at.isna().all()


def test_walk_forward_overlap_purge_is_scoped_to_same_symbol():
    data = _segments(4)
    data["start_idx"] = np.arange(len(data)) * 10
    data["end_idx"] = data["start_idx"] + 4
    data["available_at"] = data["end"] + pd.Timedelta(days=1)
    data["symbol"] = "TRAIN"
    data.loc[3, ["symbol", "start", "end", "available_at", "start_idx", "end_idx"]] = [
        "SCORE", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"), 0, 100,
    ]
    output = walk_forward_discovery_states(
        data, config=DiscoveryConfig(
            n_components=2, ensemble_size=1, min_cluster_samples=2,
            min_cluster_symbols=1, min_cluster_years=1, include_hdbscan=False,
        ), feature_columns=["cumulative_return", "return_volatility", "duration_bars", "amplitude"],
        min_train_rows=3, retrain_every=1,
    )
    scored = output.loc[output.symbol.eq("SCORE")].iloc[0]
    assert scored.unknown_reason != "purged_overlap"
    assert scored.oos_train_rows >= 3
