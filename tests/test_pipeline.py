from __future__ import annotations

import numpy as np
import pandas as pd

from wave_segments.config import PipelineConfig, SegmentationConfig
from wave_segments.pipeline import WavePipeline
from wave_segments.review import apply_review_labels, create_review_table
from wave_segments.segmentation import segment_ohlcv


def sample_bars(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    drift = np.repeat([.003, -.003, 0, .002], [60, 60, 50, 70])[:n]
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, .006, n)))
    open_ = close * np.exp(rng.normal(0, .002, n))
    spread = np.full(n, .004)
    return pd.DataFrame({
        "symbol": "TEST", "timestamp": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread), "close": close,
        "volume": rng.lognormal(12, .3, n),
    })


def test_short_series_keeps_edge_segment():
    segments = segment_ohlcv(sample_bars(3), SegmentationConfig(min_bars=4))
    assert len(segments) == 1
    assert segments.iloc[0]["start_idx"] == 0
    assert segments.iloc[0]["end_idx"] == 2


def test_end_to_end_outputs_soft_labels_and_review():
    bars = pd.concat([sample_bars().assign(symbol=f"T{i}") for i in range(3)], ignore_index=True)
    config = PipelineConfig(segmentation=SegmentationConfig(bootstrap_iterations=2))
    result = WavePipeline(config).fit_run(bars, write_outputs=False)
    assert len(result.segments) >= 6
    probability_columns = [c for c in result.labels if c.startswith("prob_")]
    assert len(probability_columns) >= 2
    assert np.allclose(result.labels[probability_columns].sum(axis=1), 1, atol=1e-6)
    assert {"label", "recognizability", "unknown_reason"}.issubset(result.labels)
    assert result.labels["candidate_label"].str.startswith("CLUSTER_").all()
    assert not result.labels["candidate_label"].str.contains("UPTREND|DOWNTREND|RANGE").any()
    assert "bootstrap_boundary_stability" in result.segments
    assert len(result.merged_segments) <= len(result.features)
    queue = create_review_table(result.features.merge(result.labels, on="segment_id"))
    queue.loc[queue.index[0], ["review_status", "true_label"]] = ["accepted", "NEW_TYPE"]
    feedback = apply_review_labels(result.labels, queue)
    assert "NEW_TYPE" in feedback["supervised_label"].values
