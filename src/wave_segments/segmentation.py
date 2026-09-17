"""Variable-length, uncertainty-aware segmentation of OHLCV bars.

The implementation deliberately produces *candidates*, rather than pretending
that a pivot is an observed fact.  ATR ZigZag supplies robust price pivots and
an optional ``ruptures`` pass supplies structural change points.  The two are
merged into probabilistic boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SegmentationConfig
from .schema import normalize_ohlcv


@dataclass(frozen=True)
class Boundary:
    index: int
    probability: float
    sources: str
    available_index: int | None = None


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR, aligned to ``frame`` and safe for short series."""
    prev = frame["close"].shift(1)
    tr = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - prev).abs(), (frame["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / max(1, period), adjust=False, min_periods=1).mean()


def _zigzag_boundaries(frame: pd.DataFrame, cfg: SegmentationConfig) -> list[Boundary]:
    """Find ATR reversal pivots; the score is proportional to reversal size."""
    n = len(frame)
    if n < 2:
        return [Boundary(0, 1.0, "edge")] if n else []
    close = frame["close"].to_numpy(float)
    atr = average_true_range(frame, cfg.atr_period).to_numpy(float)
    # Use high/low for locating extremes, close for confirming reversal.
    high, low = frame["high"].to_numpy(float), frame["low"].to_numpy(float)
    result: list[Boundary] = [Boundary(0, 1.0, "edge")]
    direction = 0  # 1 rising leg, -1 falling leg
    extreme_idx, extreme_price = 0, close[0]
    for i in range(1, n):
        threshold = max(float(atr[i]) * cfg.atr_reversal, abs(close[i]) * 1e-6)
        if direction >= 0:
            if high[i] >= extreme_price:
                extreme_idx, extreme_price = i, high[i]
            if extreme_price - close[i] >= threshold:
                strength = (extreme_price - close[i]) / threshold
                result.append(Boundary(extreme_idx, float(np.clip(0.42 + .18 * min(strength, 3), .42, .96)), "atr_zigzag"))
                direction, extreme_idx, extreme_price = -1, i, low[i]
        if direction <= 0:
            if low[i] <= extreme_price:
                extreme_idx, extreme_price = i, low[i]
            if close[i] - extreme_price >= threshold:
                strength = (close[i] - extreme_price) / threshold
                result.append(Boundary(extreme_idx, float(np.clip(0.42 + .18 * min(strength, 3), .42, .96)), "atr_zigzag"))
                direction, extreme_idx, extreme_price = 1, i, high[i]
    result.append(Boundary(n - 1, 1.0, "edge"))
    return result


def _ruptures_boundaries(frame: pd.DataFrame, penalty: float | None) -> list[Boundary]:
    """Optional structural breaks. Missing optional dependency is harmless."""
    if len(frame) < 12:
        return []
    try:
        import ruptures as rpt  # type: ignore
    except ImportError:
        return []
    log_close = np.log(frame["close"].clip(lower=1e-12)).to_numpy()
    returns = np.diff(log_close, prepend=log_close[0])
    volume = np.log1p(frame["volume"].clip(lower=0)).to_numpy()
    x = np.column_stack((returns, volume))
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 1e-10, x.std(axis=0), 1)
    try:
        breaks = rpt.Pelt(model="rbf", min_size=max(3, len(frame) // 30)).fit(x).predict(pen=float(penalty or 5.0))
    except Exception:
        return []
    return [Boundary(int(i - 1), 0.58, "ruptures") for i in breaks[:-1] if 0 < i < len(frame) - 1]


def _structural_boundaries(frame: pd.DataFrame, cfg: SegmentationConfig) -> list[Boundary]:
    """Dependency-free adaptive structure-change candidates.

    The comparison span scales with series length; it is used only to score
    possible boundaries and never dictates the resulting segment duration.
    """
    n = len(frame)
    span = max(cfg.min_bars, int(np.sqrt(n)))
    if n < 2 * span + 1:
        return []
    log_price = np.log(frame["close"].clip(lower=1e-12))
    returns = log_price.diff().fillna(0.0)
    range_pct = ((frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)).fillna(0.0)
    log_volume = np.log1p(frame["volume"].clip(lower=0))
    signals = []
    for values in (returns, returns.abs(), range_pct, log_volume):
        before = values.rolling(span, min_periods=span).mean().shift(1)
        after = values.iloc[::-1].rolling(span, min_periods=span).mean().iloc[::-1].shift(-1)
        scale = float((values - values.median()).abs().median()) * 1.4826 + 1e-9
        signals.append(np.nan_to_num(((after - before).abs() / scale).to_numpy(float), nan=0.0))
    score = np.max(np.vstack(signals), axis=0)
    candidates: list[int] = []
    threshold = max(1.25, float(np.nanquantile(score[np.isfinite(score)], .72)))
    for i in range(span, n - span):
        left, right = max(span, i - cfg.min_bars), min(n - span, i + cfg.min_bars + 1)
        if score[i] >= threshold and score[i] >= np.nanmax(score[left:right]):
            if not candidates or i - candidates[-1] >= cfg.min_bars:
                candidates.append(i)
            elif score[i] > score[candidates[-1]]:
                candidates[-1] = i
    return [Boundary(i, float(np.clip(.38 + .12 * score[i], .42, .82)), "adaptive_structure") for i in candidates]


def _fractal_boundaries(frame: pd.DataFrame, cfg: SegmentationConfig) -> list[Boundary]:
    """Local high/low fractals filtered by prominence relative to ATR."""
    window, n = max(1, cfg.fractal_window), len(frame)
    if n < 2 * window + 1:
        return []
    high, low = frame.high.to_numpy(float), frame.low.to_numpy(float)
    atr = average_true_range(frame, cfg.atr_period).to_numpy(float)
    candidates: list[Boundary] = []
    for i in range(window, n - window):
        neighbour_high = max(np.max(high[i - window:i]), np.max(high[i + 1:i + window + 1]))
        neighbour_low = min(np.min(low[i - window:i]), np.min(low[i + 1:i + window + 1]))
        peak_prominence = (high[i] - neighbour_high) / max(atr[i], 1e-12)
        trough_prominence = (neighbour_low - low[i]) / max(atr[i], 1e-12)
        prominence = max(peak_prominence, trough_prominence)
        if prominence >= cfg.fractal_min_prominence_atr:
            probability = float(np.clip(.38 + .18 * prominence, .40, .82))
            candidates.append(Boundary(i, probability, "fractal"))
    return candidates


def _bocpd_probabilities(values: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Student-t Bayesian online change probabilities for one standardized series."""
    from scipy.stats import t as student_t

    x = np.asarray(values, float)
    median = np.nanmedian(x)
    scale = np.nanmedian(np.abs(x - median)) * 1.4826 + 1e-9
    x = np.nan_to_num((x - median) / scale)
    maximum = max(cfg.bocpd_min_run_length + 1, cfg.bocpd_max_run_length)
    run = np.zeros(maximum + 1); run[0] = 1.0
    mu = np.zeros(maximum + 1)
    kappa = np.full(maximum + 1, .25)
    alpha = np.full(maximum + 1, 1.5)
    beta = np.full(maximum + 1, 1.0)
    output = np.zeros(len(x))
    hazard = float(np.clip(cfg.bocpd_hazard, 1e-5, .5))
    for pos, value in enumerate(x):
        predictive_scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        predictive = student_t.pdf(value, df=2 * alpha, loc=mu, scale=predictive_scale) + 1e-300
        prior_predictive = predictive[0]
        updated = np.zeros_like(run)
        updated[1:] = run[:-1] * (1 - hazard) * predictive[:-1]
        updated[0] = run.sum() * hazard * prior_predictive
        updated /= max(updated.sum(), 1e-300)
        output[pos] = updated[0]
        new_mu, new_kappa = np.zeros_like(mu), np.full_like(kappa, .25)
        new_alpha, new_beta = np.full_like(alpha, 1.5), np.full_like(beta, 1.0)
        old_kappa = kappa[:-1]
        new_kappa[1:] = old_kappa + 1
        new_mu[1:] = (old_kappa * mu[:-1] + value) / new_kappa[1:]
        new_alpha[1:] = alpha[:-1] + .5
        new_beta[1:] = beta[:-1] + old_kappa * (value - mu[:-1]) ** 2 / (2 * new_kappa[1:])
        run, mu, kappa, alpha, beta = updated, new_mu, new_kappa, new_alpha, new_beta
    return output


def _bocpd_boundaries(frame: pd.DataFrame, cfg: SegmentationConfig) -> list[Boundary]:
    if len(frame) < 2 * cfg.bocpd_min_run_length + 1:
        return []
    log_return = np.log(frame.close.clip(lower=1e-12)).diff().fillna(0).to_numpy()
    absolute_return = np.abs(log_return)
    probability = np.maximum(_bocpd_probabilities(log_return, cfg), _bocpd_probabilities(absolute_return, cfg))
    result = []
    radius = max(1, cfg.bocpd_min_run_length // 2)
    for i in range(cfg.bocpd_min_run_length, len(frame) - cfg.bocpd_min_run_length):
        local = probability[max(0, i - radius):min(len(frame), i + radius + 1)]
        if probability[i] >= cfg.bocpd_change_probability and probability[i] >= local.max():
            result.append(Boundary(i, float(np.clip(.4 + .55 * probability[i], .42, .9)), "bocpd"))
    return result


def _merge_boundaries(
    boundaries: Iterable[Boundary], n: int, tolerance: int, min_bars: int,
    min_probability: float = 0.0, min_votes: int = 1,
    single_source_min_probability: float = 1.0,
) -> list[Boundary]:
    ordered = sorted(boundaries, key=lambda x: x.index)
    groups: list[list[Boundary]] = []
    for b in ordered:
        if not groups or b.index - groups[-1][-1].index > tolerance:
            groups.append([b])
        else:
            groups[-1].append(b)
    merged: list[Boundary] = []
    for group in groups:
        # Agreement increases confidence, but no source is silently discarded.
        idx = int(round(np.average([x.index for x in group], weights=[x.probability for x in group])))
        p = 1 - float(np.prod([1 - x.probability for x in group]))
        merged.append(Boundary(idx, min(.99, p), "+".join(sorted({x.sources for x in group}))))
    # A weak boundary proposed by only one method is not a vote. Strong lone
    # pivots remain possible, which keeps abrupt event shocks detectable.
    filtered = []
    for boundary in merged:
        sources = set(boundary.sources.split("+")) - {"edge"}
        keep = boundary.index in {0, n - 1} or (
            boundary.probability >= min_probability and
            (len(sources) >= min_votes or boundary.probability >= single_source_min_probability)
        )
        if keep:
            filtered.append(boundary)
    merged = filtered
    if len(merged) < 2:
        return [Boundary(0, 1.0, "edge"), Boundary(n - 1, 1.0, "edge")]
    if len(merged) < 2:
        return [Boundary(0, 1.0, "edge"), Boundary(n - 1, 1.0, "edge")]
    merged[0] = Boundary(0, 1.0, "edge")
    merged[-1] = Boundary(n - 1, 1.0, "edge")
    # Drop too-short intermediate legs. Keep the more certain of competing pivots.
    changed = True
    while changed and len(merged) > 2:
        changed = False
        for i in range(1, len(merged) - 1):
            if merged[i].index - merged[i - 1].index < min_bars:
                if i > 1 and merged[i - 1].probability <= merged[i].probability:
                    merged.pop(i - 1)
                else:
                    merged.pop(i)
                changed = True
                break
    return merged


def segment_ohlcv(
    ohlcv: pd.DataFrame,
    config: SegmentationConfig | None = None,
    *,
    use_changepoints: bool = False,
    changepoint_penalty: float | None = None,
) -> pd.DataFrame:
    """Return candidate variable-length segments for every symbol in ``ohlcv``.

    Indices are positional within each symbol's sorted frame and are included to
    make downstream feature extraction reproducible. Boundaries are inclusive.
    """
    cfg = config or SegmentationConfig()
    if cfg.min_bars < 1:
        raise ValueError("min_bars must be at least one")
    data = normalize_ohlcv(ohlcv)
    rows: list[dict[str, object]] = []
    for symbol, group in data.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if not n:
            continue
        boundaries = _zigzag_boundaries(group, cfg)
        if use_changepoints:
            if cfg.use_adaptive_structure:
                boundaries += _structural_boundaries(group, cfg)
            if cfg.use_fractal_pivots:
                boundaries += _fractal_boundaries(group, cfg)
            if cfg.use_bocpd:
                boundaries += _bocpd_boundaries(group, cfg)
            if cfg.use_ruptures:
                boundaries += _ruptures_boundaries(group, changepoint_penalty)
        boundaries = _merge_boundaries(
            boundaries, n, cfg.merge_tolerance, cfg.min_bars,
            cfg.min_boundary_probability, cfg.min_boundary_votes,
            cfg.single_source_min_probability,
        )
        for seq, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # Adjacent legs share their turning bar, a useful intentional overlap.
            rows.append({
                "segment_id": f"{symbol}:{seq}", "symbol": symbol, "segment_no": seq,
                "start_idx": left.index, "end_idx": right.index,
                "start": group.loc[left.index, "timestamp"], "end": group.loc[right.index, "timestamp"],
                "start_boundary_probability": left.probability, "end_boundary_probability": right.probability,
                "boundary_probability": min(left.probability, right.probability),
                "boundary_uncertainty": 1 - min(left.probability, right.probability),
                "start_boundary_sources": left.sources, "end_boundary_sources": right.sources,
                "n_bars": right.index - left.index + 1,
            })
    columns = ["segment_id", "symbol", "segment_no", "start_idx", "end_idx", "start", "end", "start_boundary_probability", "end_boundary_probability", "boundary_probability", "boundary_uncertainty", "start_boundary_sources", "end_boundary_sources", "n_bars"]
    return pd.DataFrame(rows, columns=columns)


def segment_ohlcv_causal(
    ohlcv: pd.DataFrame,
    config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """Emit only ATR-ZigZag legs whose terminal pivot is confirmed in-sample.

    Unlike :func:`segment_ohlcv`, this online-compatible path never runs a
    retrospective detector or appends the latest bar as a confirmed endpoint.
    ``end`` is the pivot date; ``available_at``/``confirmation_idx`` record the
    later bar on which the reversal threshold made that pivot knowable. An open
    leg at the end of the input is intentionally omitted. Prefix runs should
    therefore only append confirmed rows, never revise prior ones.

    Boundary probabilities are heuristic strength scores, not calibrated
    probabilities of boundary correctness.
    """
    cfg = config or SegmentationConfig()
    if cfg.min_bars < 1:
        raise ValueError("min_bars must be at least one")
    data = normalize_ohlcv(ohlcv)
    rows: list[dict[str, object]] = []
    for symbol, group in data.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue
        close = group["close"].to_numpy(float)
        high = group["high"].to_numpy(float)
        low = group["low"].to_numpy(float)
        atr = average_true_range(group, cfg.atr_period).to_numpy(float)
        # (pivot bar, heuristic probability, confirmation bar)
        pivots: list[tuple[int, float, int]] = [(0, 1.0, 0)]
        direction = 0
        extreme_idx, extreme_price = 0, close[0]
        for i in range(1, n):
            threshold = max(float(atr[i]) * cfg.atr_reversal, abs(close[i]) * 1e-6)
            if direction == 0:
                rise, fall = high[i] - extreme_price, extreme_price - low[i]
                if rise >= threshold and rise >= fall:
                    direction, extreme_idx, extreme_price = 1, i, high[i]
                elif fall >= threshold:
                    direction, extreme_idx, extreme_price = -1, i, low[i]
                continue
            if direction > 0:
                if high[i] >= extreme_price:
                    extreme_idx, extreme_price = i, high[i]
                elif extreme_price - close[i] >= threshold:
                    strength = (extreme_price - close[i]) / threshold
                    pivots.append((extreme_idx, float(np.clip(.42 + .18 * min(strength, 3), .42, .96)), i))
                    direction, extreme_idx, extreme_price = -1, i, low[i]
            else:
                if low[i] <= extreme_price:
                    extreme_idx, extreme_price = i, low[i]
                elif close[i] - extreme_price >= threshold:
                    strength = (close[i] - extreme_price) / threshold
                    pivots.append((extreme_idx, float(np.clip(.42 + .18 * min(strength, 3), .42, .96)), i))
                    direction, extreme_idx, extreme_price = 1, i, high[i]
        for seq, (left, right) in enumerate(zip(pivots[:-1], pivots[1:])):
            left_idx, left_p, left_confirm = left
            right_idx, right_p, right_confirm = right
            if right_idx - left_idx < cfg.min_bars:
                continue
            rows.append({
                "segment_id": f"{symbol}:{seq}", "symbol": symbol, "segment_no": seq,
                "start_idx": left_idx, "end_idx": right_idx,
                "start": group.loc[left_idx, "timestamp"], "end": group.loc[right_idx, "timestamp"],
                "start_boundary_probability": left_p, "end_boundary_probability": right_p,
                "boundary_probability": min(left_p, right_p),
                "boundary_uncertainty": 1 - min(left_p, right_p),
                "start_boundary_sources": "edge" if seq == 0 else "causal_atr_zigzag",
                "end_boundary_sources": "causal_atr_zigzag",
                "n_bars": right_idx - left_idx + 1,
                "confirmation_idx": right_confirm,
                "available_at": group.loc[right_confirm, "timestamp"],
                "start_available_at": group.loc[left_confirm, "timestamp"],
                "segmenter": "causal_atr_zigzag",
            })
    return pd.DataFrame(rows)


class VariableLengthSegmenter:
    """Small estimator-like facade useful in notebooks and pipelines."""
    def __init__(self, config: SegmentationConfig | None = None, *, use_changepoints: bool = False, changepoint_penalty: float | None = None):
        self.config, self.use_changepoints, self.changepoint_penalty = config or SegmentationConfig(), use_changepoints, changepoint_penalty

    def fit_predict(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return segment_ohlcv(ohlcv, self.config, use_changepoints=self.use_changepoints, changepoint_penalty=self.changepoint_penalty)


segment_bars = segment_ohlcv
