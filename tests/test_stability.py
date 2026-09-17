from itertools import pairwise

import numpy as np
import pandas as pd

from wave_segments.stability import (
    bootstrap_boundary_stability,
    boundary_precision_recall_f1,
    merge_consecutive_same_label,
    perturb_ohlcv,
)


def bars(n=32):
    close = np.linspace(10, 12, n) + np.sin(np.arange(n) / 2)
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n), "symbol": "AAA", "open": close, "high": close + .3, "low": close - .3, "close": close, "volume": 1000})


def fixed_segmenter(frame):
    n = len(frame)
    points = [0, n // 2, n - 1]
    return pd.DataFrame([{"segment_id": f"AAA:{i}", "symbol": "AAA", "start_idx": a, "end_idx": b} for i, (a, b) in enumerate(pairwise(points))])


def test_perturbation_preserves_ohlc_invariants_and_is_reproducible():
    first = perturb_ohlcv(bars(), random_state=7)
    second = perturb_ohlcv(bars(), random_state=7)
    assert first.equals(second)
    assert (first.high >= first[["open", "close", "low"]].max(axis=1)).all()
    assert (first.low <= first[["open", "close", "high"]].min(axis=1)).all()


def test_bootstrap_frequency_and_tolerant_truth_metric():
    baseline, stability = bootstrap_boundary_stability(bars(), iterations=5, random_state=1, segmenter=fixed_segmenter)
    assert len(baseline) == 2
    assert stability.iloc[0].boundary_idx == len(bars()) // 2
    assert stability.iloc[0].stability_frequency == 1.0
    score = boundary_precision_recall_f1(
        pd.DataFrame({"symbol": ["AAA", "AAA"], "boundary_idx": [10, 20]}),
        pd.DataFrame({"symbol": ["AAA", "AAA"], "boundary_idx": [11, 25]}), tolerance=1,
    )
    assert score["matches"] == 1
    assert score["precision"] == score["recall"] == 0.5


def test_merge_consecutive_labels_keeps_provenance_and_aggregates_probabilities():
    segments = pd.DataFrame([
        {"segment_id": "AAA:0", "symbol": "AAA", "segment_no": 0, "start_idx": 0, "end_idx": 4, "label": "Cluster A", "prob_a": .8, "prob_b": .2, "start_boundary_probability": 1., "end_boundary_probability": .6},
        {"segment_id": "AAA:1", "symbol": "AAA", "segment_no": 1, "start_idx": 4, "end_idx": 9, "label": "Cluster A", "prob_a": .6, "prob_b": .4, "start_boundary_probability": .6, "end_boundary_probability": .7},
        {"segment_id": "AAA:2", "symbol": "AAA", "segment_no": 2, "start_idx": 9, "end_idx": 12, "label": "UNKNOWN", "prob_a": .5, "prob_b": .5, "start_boundary_probability": .7, "end_boundary_probability": 1.},
    ])
    actual = merge_consecutive_same_label(segments)
    assert len(actual) == 2
    merged = actual.iloc[0]
    assert merged.segment_ids == ["AAA:0", "AAA:1"]
    assert merged.start_idx == 0 and merged.end_idx == 9 and merged.n_bars == 10
    assert np.isclose(merged.prob_a + merged.prob_b, 1)
    assert np.isclose(merged.boundary_probability, .7)


def test_merge_recomputes_boundary_evidence_from_outer_endpoints_only():
    segments = pd.DataFrame([
        {"segment_id": "S0", "symbol": "AAA", "start_idx": 0, "end_idx": 4,
         "label": "A", "prob_a": .9, "prob_b": .1,
         "detector_start_boundary_probability": .9, "detector_end_boundary_probability": .4,
         "detector_boundary_probability": .4, "start_boundary_stability": .8,
         "end_boundary_stability": .2, "bootstrap_boundary_stability": .2,
         "start_boundary_probability": .8, "end_boundary_probability": .2},
        {"segment_id": "S1", "symbol": "AAA", "start_idx": 4, "end_idx": 8,
         "label": "A", "prob_a": .9, "prob_b": .1,
         "detector_start_boundary_probability": .4, "detector_end_boundary_probability": .9,
         "detector_boundary_probability": .4, "start_boundary_stability": .2,
         "end_boundary_stability": .9, "bootstrap_boundary_stability": .2,
         "start_boundary_probability": .2, "end_boundary_probability": .9},
    ])
    merged = merge_consecutive_same_label(segments).iloc[0]
    assert merged.end_boundary_stability == .9
    assert merged.detector_boundary_probability == .9
    assert merged.bootstrap_boundary_stability == .8
    assert merged.boundary_probability == .8
