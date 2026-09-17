from __future__ import annotations

import numpy as np
import pandas as pd

from wave_segments.selection import (
    assess_selection_evidence,
    build_selection_dataset,
    compare_baselines,
    evaluate_incremental_features,
    evaluate_selection,
)
import pytest


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    rows = []
    for i, symbol in enumerate(["A", "B", "C", "MKT"]):
        close = 100 + i + np.arange(len(dates)) * (1 + i * .1)
        rows.extend({"symbol": symbol, "timestamp": d, "close": c, "low": c * .98,
                     "industry": "I1" if i < 2 else "I2", "market_cap": 10 + i} for d, c in zip(dates, close))
    return pd.DataFrame(rows)


def test_states_are_asof_and_targets_are_strictly_future():
    bars = _bars()
    states = pd.DataFrame({"symbol": ["A", "A", "B", "C"], "timestamp": [bars.timestamp.iloc[2], bars.timestamp.iloc[7], bars.timestamp.iloc[2], bars.timestamp.iloc[2]],
                           "state_score": [1., 99., 2., 3.]})
    out = build_selection_dataset(bars, states, horizons=(2,), benchmark_symbol="MKT")
    a = out[out.symbol.eq("A")].sort_values("timestamp").reset_index(drop=True)
    assert pd.isna(a.loc[1, "state_score"])
    assert pd.isna(a.loc[2, "state_score"])
    assert a.loc[3, "state_score"] == 1  # close-time state becomes usable next bar
    assert a.loc[7, "state_score"] == 1
    assert a.loc[8, "state_score"] == 99
    expected = a.loc[0, "close"] and a.loc[2, "close"] / a.loc[0, "close"] - 1
    assert np.isclose(a.loc[0, "future_excess_2d"], expected - (out[out.symbol.eq("MKT")].iloc[2].close / out[out.symbol.eq("MKT")].iloc[0].close - 1))
    assert pd.isna(a.iloc[-2]["future_excess_2d"])
    with pytest.raises(ValueError, match="future/outcome"):
        build_selection_dataset(bars, states.assign(future_return_20d=1.0), horizons=(2,))
    with pytest.raises(ValueError, match="future/outcome"):
        build_selection_dataset(bars, states.assign(hsmm_label="CLUSTER_A"), horizons=(2,))


@pytest.mark.parametrize("offline_column", ["pre_hsmm_label", "hsmm_path_score", "hsmm_changed"])
def test_selection_rejects_every_offline_hsmm_artifact_but_allows_causal_duration(offline_column):
    bars = _bars()
    states = bars[["symbol", "timestamp"]].copy()
    states[offline_column] = 0.5
    with pytest.raises(ValueError, match="future/outcome"):
        build_selection_dataset(bars, states, horizons=(2,))

    causal = bars[["symbol", "timestamp"]].copy()
    causal["causal_duration_label"] = "CLUSTER_A"
    result = build_selection_dataset(bars, causal, horizons=(2,))
    assert result["causal_duration_label"].notna().any()


def test_evaluation_neutralization_costs_and_baseline_comparison():
    bars = _bars()
    states = bars[["symbol", "timestamp", "industry", "market_cap"]].copy()
    states["state_score"] = states["symbol"].map({"A": 1., "B": 2., "C": 3., "MKT": 0.})
    states["momentum_20"] = states["state_score"]
    states["volatility_20"] = -states["state_score"]
    data = build_selection_dataset(bars, states, horizons=(2,))
    report = evaluate_selection(data, "state_score", "future_excess_2d", groups=2, transaction_cost_bps=10, slippage_bps=5)
    assert {"rank_ic", "icir", "group_returns", "turnover", "net_long_short", "max_drawdown", "yearly_stability"}.issubset(report)
    assert (report["net_long_short"] <= report["gross_long_short"] + 1e-12).all()
    assert report["rebalance_every"] == 2
    assert "rank_ic_hac_t" in report
    assert len(report["gross_long_short"]) <= int(np.ceil(data.timestamp.nunique() / 2))
    comparison = compare_baselines(data, ["state_score", "momentum_20", "volatility_20"], target_col="future_excess_2d", groups=2)
    assert set(comparison.score) == {"state_score", "momentum_20", "volatility_20"}


def test_forward_horizon_uses_market_sessions_not_next_observed_stock_rows():
    dates = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame([
        {"symbol": symbol, "timestamp": date, "close": 10 + i, "low": 9 + i}
        for symbol in ("A", "B") for i, date in enumerate(dates)
        if not (symbol == "A" and date == dates[2])
    ])
    states = pd.DataFrame({"symbol": ["A", "B"], "timestamp": [dates[0], dates[0]], "score": [1., 2.]})
    out = build_selection_dataset(bars, states, horizons=(2,))
    row = out[out.symbol.eq("A") & out.timestamp.eq(dates[0])].iloc[0]
    assert row.target_available_at_2d == dates[2]
    assert pd.isna(row.future_excess_2d)


def test_conservative_evidence_gate_requires_classification_and_selection_quality():
    selection = {"rank_ic": .04, "rank_ic_hac_t": 2.4, "rank_ic_positive_rate": .60,
                 "monotonicity": .8, "mean_net_long_short": .001, "scored_rows": 2000,
                 "causal_audit_passed": True}
    classification = {"selective_risk": .08, "ece": .05, "coverage": .65,
                      "unknown_rejection_rate": .75}
    assert assess_selection_evidence(selection, classification_metrics=classification)["passed"]
    weak = dict(selection, rank_ic_hac_t=1.55, monotonicity=-.4)
    assessment = assess_selection_evidence(weak, classification_metrics=classification)
    assert not assessment["passed"]
    assert {"hac_significance", "quantile_monotonicity"}.issubset(assessment["failed_checks"])


def test_incremental_wave_feature_ablation_is_paired_and_causal():
    dates = pd.bdate_range("2023-01-02", periods=36)
    symbols = ["A", "B", "C", "D", "E", "F"]
    bars = pd.DataFrame([
        {"symbol": symbol, "timestamp": date, "close": 100 + day * (1 + rank / 20),
         "low": 99 + day * (1 + rank / 20)}
        for day, date in enumerate(dates) for rank, symbol in enumerate(symbols)
    ])
    states = bars[["symbol", "timestamp"]].copy()
    states["momentum"] = states["symbol"].map(dict(zip(symbols, range(6)))).astype(float)
    states["wave_probability"] = np.sin(np.arange(len(states)) / 9)
    data = build_selection_dataset(bars, states, horizons=(2,), allow_same_timestamp_state=True)
    report = evaluate_incremental_features(
        data, ["momentum"], ["wave_probability"], "future_excess_2d",
        min_train_dates=5, min_train_rows=20, retrain_every=2, groups=3,
    )
    assert report["overlap_scored_rows"] > 0
    assert report["baseline_causal_audit_passed"]
    assert report["augmented_causal_audit_passed"]
    assert {"baseline", "augmented", "incremental_rank_ic_hac_t"}.issubset(report)
