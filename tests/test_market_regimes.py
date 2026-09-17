import numpy as np
import pandas as pd

from wave_segments.market_regimes import sample_market_regime


def _bars(returns, symbols=("A", "B")):
    rows = []
    dates = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    for offset, symbol in enumerate(symbols):
        close = 100 * np.cumprod(1 + np.asarray(returns, float) * (1 + .02 * offset))
        for date, price in zip(dates, close):
            rows.append({"timestamp": date, "symbol": symbol, "open": price,
                         "high": price * 1.01, "low": price * .99, "close": price,
                         "volume": 1000.})
    return pd.DataFrame(rows)


def test_market_regime_uses_trailing_returns_and_warmup_unknown():
    out = sample_market_regime(_bars(np.full(25, .006)), window=20, trend_threshold=.05)
    assert out.market_proxy_regime.iloc[:19].eq("UNKNOWN").all()
    assert out.market_proxy_regime.iloc[-1] == "BULL_PROXY"
    assert out.market_proxy_trailing_return.iloc[-1] > .05


def test_market_regime_is_prefix_invariant_and_detects_bear_proxy():
    returns = np.r_[np.full(20, .004), np.full(25, -.007)]
    bars = _bars(returns)
    full = sample_market_regime(bars, window=20, trend_threshold=.05)
    prefix_cutoff = bars.timestamp.unique()[32]
    prefix = sample_market_regime(bars.loc[bars.timestamp.le(prefix_cutoff)], window=20, trend_threshold=.05)
    expected = full.loc[full.timestamp.le(prefix_cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)
    assert full.market_proxy_regime.iloc[-1] == "BEAR_PROXY"
