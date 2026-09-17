"""Causal market-context proxy from a fixed sample of daily OHLCV securities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import normalize_ohlcv


def sample_market_regime(
    bars: pd.DataFrame,
    *,
    window: int = 20,
    trend_threshold: float = 0.05,
) -> pd.DataFrame:
    """Build an equal-weight cross-sectional median-return regime proxy.

    The proxy uses each date's cross-sectional median close-to-close return,
    then trailing ``window`` observations inclusive of that date. It is not an
    official index, is not market-cap weighted, and has no look-ahead. The first
    ``window - 1`` observations are UNKNOWN due to warm-up.
    """
    if window < 2 or trend_threshold < 0:
        raise ValueError("window must be >=2 and trend_threshold non-negative")
    data = normalize_ohlcv(bars).sort_values(["symbol", "timestamp"], kind="stable")
    close = pd.to_numeric(data.close, errors="coerce")
    data["_daily_return"] = close.groupby(data.symbol, sort=False).pct_change(fill_method=None)
    proxy = data.groupby("timestamp", sort=True)["_daily_return"].median().dropna()
    trailing_return = (1 + proxy).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
    trailing_volatility = proxy.rolling(window, min_periods=window).std() * np.sqrt(252)
    regime = pd.Series("UNKNOWN", index=proxy.index, dtype=object)
    regime.loc[trailing_return > trend_threshold] = "BULL_PROXY"
    regime.loc[trailing_return < -trend_threshold] = "BEAR_PROXY"
    known = trailing_return.notna() & ~trailing_return.gt(trend_threshold) & ~trailing_return.lt(-trend_threshold)
    regime.loc[known] = "SIDEWAYS_PROXY"
    return pd.DataFrame({
        "timestamp": proxy.index,
        "sample_market_return": proxy.to_numpy(float),
        "market_proxy_trailing_return": trailing_return.to_numpy(float),
        "market_proxy_trailing_volatility": trailing_volatility.to_numpy(float),
        "market_proxy_regime": regime.to_numpy(object),
        "market_proxy_window": window,
        "market_proxy_threshold": trend_threshold,
    })
