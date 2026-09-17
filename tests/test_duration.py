import pandas as pd

from wave_segments.duration import (
    causal_duration_filter,
    decode_hsmm,
    geometric_duration_pmf,
)


def emissions():
    return pd.DataFrame({
        "segment_id": [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(2)],
        "symbol": ["A"] * 5 + ["B"] * 2,
        "start_idx": [0, 2, 4, 6, 8, 0, 2], "end_idx": [2, 4, 6, 8, 10, 2, 4],
        "n_bars": [3] * 7,
        "prob_CLUSTER_A": [.95, .95, .10, .95, .95, .05, .05],
        "prob_CLUSTER_B": [.05, .05, .90, .05, .05, .95, .95],
        "label": ["CLUSTER_A"] * 2 + ["CLUSTER_B"] + ["CLUSTER_A"] * 2 + ["CLUSTER_B"] * 2,
    })


def test_offline_hsmm_suppresses_single_segment_blip_and_isolates_symbols():
    frame = emissions()
    pmf = geometric_duration_pmf(["CLUSTER_A", "CLUSTER_B"], max_duration=5, min_duration=2)
    out = decode_hsmm(frame, duration_pmf=pmf)
    assert set(out[out.symbol.eq("A")].hsmm_label) == {"CLUSTER_A"}
    assert set(out[out.symbol.eq("B")].hsmm_label) == {"CLUSTER_B"}
    assert out.hsmm_changed.any()


def test_causal_filter_uses_no_later_emissions_and_preserves_unknown():
    frame = emissions()
    first = causal_duration_filter(frame.iloc[:3], min_dwell_bars=10)
    extended = causal_duration_filter(frame.iloc[:5], min_dwell_bars=10)
    assert first.causal_duration_label.tolist() == extended.iloc[:3].causal_duration_label.tolist()
    unknown = frame.iloc[:3].copy(); unknown.loc[1, "label"] = "UNKNOWN"
    assert causal_duration_filter(unknown).loc[1, "causal_duration_label"] == "UNKNOWN"
