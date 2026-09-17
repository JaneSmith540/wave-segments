import numpy as np
import pandas as pd

from wave_segments.selection import (
    build_selection_dataset,
    walk_forward_cross_sectional_scores,
)


def _dataset():
    dates = pd.bdate_range("2024-01-02", periods=12)
    rows = []
    for i, date in enumerate(dates):
        for symbol, offset in (("AAA", 0.0), ("BBB", 1.0), ("CCC", 2.0)):
            rows.append({"symbol": symbol, "timestamp": date, "close": 10 + i + offset,
                         "low": 9 + i + offset, "state_strength": offset + i / 10})
    bars = pd.DataFrame(rows)
    states = bars[["symbol", "timestamp", "state_strength"]].copy()
    return build_selection_dataset(bars, states, horizons=(2,))


def test_build_dataset_records_actual_target_availability():
    data = _dataset()
    aaa = data[data.symbol.eq("AAA")].sort_values("timestamp").reset_index(drop=True)
    assert aaa.loc[0, "target_available_at_2d"] == aaa.loc[2, "timestamp"]
    assert pd.isna(aaa.loc[len(aaa) - 1, "target_available_at_2d"])


def test_walk_forward_scores_are_strictly_causal_and_future_invariant():
    data = _dataset()
    early_date = pd.Timestamp("2024-01-10")
    baseline = walk_forward_cross_sectional_scores(
        data, ["state_strength"], target_col="future_excess_2d", min_train_dates=2, retrain_every=1
    )
    altered = data.copy()
    altered.loc[altered.timestamp.gt(early_date), "future_excess_2d"] = 99999.0
    changed = walk_forward_cross_sectional_scores(
        altered, ["state_strength"], target_col="future_excess_2d", min_train_dates=2, retrain_every=1
    )
    left = baseline[baseline.timestamp.le(early_date)].oos_score.to_numpy()
    right = changed[changed.timestamp.le(early_date)].oos_score.to_numpy()
    np.testing.assert_allclose(left, right, equal_nan=True)
    scored = baseline.dropna(subset=["oos_score"])
    assert (scored.oos_train_max_target_available_at < scored.timestamp).all()
