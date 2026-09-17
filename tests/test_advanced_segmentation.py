import numpy as np
import pandas as pd

from wave_segments.config import SegmentationConfig
from wave_segments.segmentation import (
    _bocpd_boundaries,
    _fractal_boundaries,
    segment_ohlcv,
    segment_ohlcv_causal,
)
from wave_segments.stability import (
    bootstrap_boundary_stability,
    bootstrap_causal_boundary_stability,
)


def regime_bars():
    n = 120
    returns = np.r_[np.full(60, .003), np.full(60, -.004)]
    close = 100 * np.exp(np.cumsum(returns))
    high, low = close * 1.003, close * .997
    high[59] *= 1.05
    return pd.DataFrame({"symbol": "R", "timestamp": pd.date_range("2024-01-01", periods=n, freq="B"),
                         "open": close, "high": high, "low": low, "close": close, "volume": 1000.})


def test_fractal_and_bocpd_candidate_generators_detect_structure():
    bars = regime_bars()
    cfg = SegmentationConfig(fractal_min_prominence_atr=.2, bocpd_change_probability=.15)
    fractals = _fractal_boundaries(bars, cfg)
    changes = _bocpd_boundaries(bars, cfg)
    assert any(abs(x.index - 59) <= 2 for x in fractals)
    assert any(abs(x.index - 60) <= 5 for x in changes)


def test_sources_are_fused_without_fixed_segment_length():
    bars = regime_bars()
    cfg = SegmentationConfig(fractal_min_prominence_atr=.2, bocpd_change_probability=.15)
    segments = segment_ohlcv(bars, cfg, use_changepoints=True)
    sources = " ".join(segments.start_boundary_sources.astype(str)) + " " + " ".join(segments.end_boundary_sources.astype(str))
    assert "fractal" in sources
    assert "bocpd" in sources
    assert segments.n_bars.nunique() > 1


def test_causal_segmenter_confirms_pivots_later_and_never_rewrites_prefixes():
    close = np.r_[np.linspace(100, 130, 30), np.linspace(130, 90, 30),
                  np.linspace(90, 125, 30), np.linspace(125, 82, 30)]
    bars = pd.DataFrame({
        "symbol": "C", "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="B"),
        "open": close, "high": close * 1.004, "low": close * .996,
        "close": close, "volume": 1000.0,
    })
    cfg = SegmentationConfig(atr_period=5, atr_reversal=1.2, min_bars=3)
    full = segment_ohlcv_causal(bars, cfg)
    assert len(full) >= 2
    assert (full.available_at >= full.end).all()
    assert (full.confirmation_idx > full.end_idx).all()
    assert full.segmenter.eq("causal_atr_zigzag").all()
    for stop in (45, 65, 85, 105, len(bars)):
        prefix = segment_ohlcv_causal(bars.iloc[:stop], cfg)
        expected = full.loc[full.available_at <= bars.timestamp.iloc[stop - 1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)


def test_bootstrap_boundary_stability_runs_with_causal_segmenter():
    bars = regime_bars()
    cfg = SegmentationConfig(atr_period=5, atr_reversal=1.2, min_bars=3)
    baseline, stability = bootstrap_boundary_stability(
        bars, cfg, iterations=4, tolerance=3,
        segmenter=lambda frame: segment_ohlcv_causal(frame, cfg), include_edges=True,
    )
    assert len(baseline) > 0 and len(stability) > 0
    assert stability.stability_frequency.between(0, 1).all()
    assert stability.bootstrap_iterations.eq(4).all()


def test_causal_bootstrap_scores_each_boundary_using_only_its_confirmation_prefix():
    bars = pd.DataFrame({
        "symbol": "P", "timestamp": pd.date_range("2024-01-01", periods=120, freq="B"),
        "open": np.r_[np.linspace(100, 130, 30), np.linspace(130, 90, 30),
                      np.linspace(90, 125, 30), np.linspace(125, 82, 30)],
        "high": np.r_[np.linspace(100, 130, 30), np.linspace(130, 90, 30),
                      np.linspace(90, 125, 30), np.linspace(125, 82, 30)] * 1.004,
        "low": np.r_[np.linspace(100, 130, 30), np.linspace(130, 90, 30),
                     np.linspace(90, 125, 30), np.linspace(125, 82, 30)] * .996,
        "close": np.r_[np.linspace(100, 130, 30), np.linspace(130, 90, 30),
                       np.linspace(90, 125, 30), np.linspace(125, 82, 30)],
        "volume": 1000.0,
    })
    cfg = SegmentationConfig(atr_period=5, atr_reversal=1.2, min_bars=3)
    full_base, full_stability = bootstrap_causal_boundary_stability(
        bars, cfg, iterations=3, tolerance=3, random_state=7,
    )
    _, prefix_stability = bootstrap_causal_boundary_stability(
        bars.iloc[:85], cfg, iterations=3, tolerance=3, random_state=7,
    )
    assert not full_base.empty
    assert full_stability.stability_frequency.between(0, 1).all()
    cutoff = bars.timestamp.iloc[84]
    expected = full_stability.loc[full_stability.available_at <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix_stability.reset_index(drop=True), expected)
