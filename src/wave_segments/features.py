"""Segment-level OHLCV, candle-shape, and context feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .segmentation import average_true_range
from .schema import normalize_ohlcv


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    return float(a.corr(b)) if len(a) >= 3 and a.nunique() > 1 and b.nunique() > 1 else np.nan


def _context_features(bars: pd.DataFrame, context: pd.DataFrame | None, prefix: str) -> dict[str, object]:
    if context is None or context.empty:
        return {}
    ctx = context.copy()
    if "timestamp" not in ctx:
        raise ValueError("context must have a timestamp column")
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], errors="coerce")
    merged = bars[["timestamp"]].merge(ctx, on="timestamp", how="left")
    out: dict[str, object] = {}
    for col in ctx.columns:
        if col in {"timestamp", "symbol"}:
            continue
        name = f"{prefix}{col}"
        if pd.api.types.is_numeric_dtype(ctx[col]):
            values = pd.to_numeric(merged[col], errors="coerce")
            out[f"{name}_mean"] = values.mean()
            out[f"{name}_end"] = values.iloc[-1] if len(values) else np.nan
        else:
            vals = merged[col].dropna()
            out[f"{name}_end"] = vals.iloc[-1] if len(vals) else None
    return out


def extract_segment_features(
    ohlcv: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    market_context: pd.DataFrame | None = None,
    sector_context: pd.DataFrame | None = None,
    multi_timeframe_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create descriptive features; no forward returns or trading labels.

    Context tables are timestamp-keyed and may contain arbitrary numeric or
    categorical fields. Numeric fields get in-segment means and end values;
    categorical fields get their final observed state.
    """
    data = normalize_ohlcv(ohlcv)
    required = {"symbol", "start_idx", "end_idx"}
    if not required.issubset(segments.columns):
        raise ValueError(f"segments missing columns: {sorted(required - set(segments.columns))}")
    by_symbol = {s: g.reset_index(drop=True) for s, g in data.groupby("symbol", sort=False)}
    rows: list[dict[str, object]] = []
    for _, seg in segments.iterrows():
        symbol, start, end = seg["symbol"], int(seg["start_idx"]), int(seg["end_idx"])
        if symbol not in by_symbol or start < 0 or end < start or end >= len(by_symbol[symbol]):
            raise ValueError(f"invalid segment bounds for {symbol}: {start}..{end}")
        b = by_symbol[symbol].iloc[start:end + 1].copy()
        atr = average_true_range(by_symbol[symbol]).iloc[start:end + 1]
        o, h, l, c, v = (b[x] for x in ["open", "high", "low", "close", "volume"])
        body = (c - o).abs()
        full_range = (h - l).replace(0, np.nan)
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - l
        ret = c.pct_change().replace([np.inf, -np.inf], np.nan)
        cumulative_return = c.iloc[-1] / c.iloc[0] - 1 if c.iloc[0] else np.nan
        running_peak = c.cummax()
        drawdown = c / running_peak - 1
        x = np.arange(len(b), dtype=float)
        slope = float(np.polyfit(x, np.log(c.clip(lower=1e-12)), 1)[0]) if len(b) >= 2 else 0.0
        direction = "up" if cumulative_return > .002 else "down" if cumulative_return < -.002 else "flat"
        row = dict(seg.to_dict())
        row.update({
            "direction": direction, "cumulative_return": cumulative_return, "log_price_slope": slope,
            "amplitude": h.max() / l.min() - 1 if l.min() else np.nan,
            "max_drawdown": drawdown.min(), "duration_bars": len(b), "duration_days": (b.timestamp.iloc[-1] - b.timestamp.iloc[0]).days,
            "return_volatility": ret.std(ddof=0), "atr_mean": atr.mean(), "atr_pct_mean": (atr / c.replace(0, np.nan)).mean(),
            "volume_mean": v.mean(), "volume_median": v.median(), "volume_cv": v.std(ddof=0) / v.mean() if v.mean() else np.nan,
            "volume_trend": float(np.polyfit(x, np.log1p(v), 1)[0]) if len(b) >= 2 else 0.0,
            "price_volume_corr": _safe_corr(ret, v.pct_change()), "up_volume_ratio": v[c.diff() > 0].sum() / v.sum() if v.sum() else np.nan,
            "body_ratio_mean": (body / full_range).mean(), "upper_shadow_ratio_mean": (upper / full_range).mean(),
            "lower_shadow_ratio_mean": (lower / full_range).mean(), "doji_ratio": (body / full_range < .1).mean(),
            "bullish_candle_ratio": (c > o).mean(), "gap_ratio": (o / c.shift(1) - 1).abs().mean(),
            "realized_range_mean": ((h - l) / c.replace(0, np.nan)).mean(),
        })
        row.update(_context_features(b, market_context, "market_"))
        row.update(_context_features(b, sector_context, "sector_"))
        row.update(_context_features(b, multi_timeframe_context, "multitf_"))
        rows.append(row)
    return pd.DataFrame(rows)


# Short alias for interactive use.
compute_segment_features = extract_segment_features
