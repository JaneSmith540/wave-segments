from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import normalize_ohlcv

TOKEN_ENV_NAMES = ("TUSHARE_API_TOKEN", "TUSHARE_TOKEN", "TS_TOKEN")


def _tushare_api(token: str | None, request_timeout: float):
    """Create a Tushare client without ever persisting its credential."""
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install 'wave-segments[data]'") from exc
    return ts.pro_api(resolve_tushare_token(token), timeout=request_timeout)


def resolve_tushare_token(token: str | None = None) -> str:
    value = token or next((os.environ.get(k) for k in TOKEN_ENV_NAMES if os.environ.get(k)), None)
    if not value:
        names = ", ".join(TOKEN_ENV_NAMES)
        raise RuntimeError(f"Tushare token not found. Set one of: {names}")
    return value


def load_local(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")
    return normalize_ohlcv(frame)


def fetch_tushare(
    symbols: Iterable[str],
    start: str,
    end: str,
    frequency: str = "D",
    adjustment: str | None = "qfq",
    token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch daily or minute bars using tushare.pro_bar.

    `frequency` accepts D/W/M or minute values supported by Tushare such as 5min.
    The token is read from the environment by default and is never persisted.
    """
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install 'wave-segments[data]'") from exc

    api_token = resolve_tushare_token(token)
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        kwargs = {
            "ts_code": symbol,
            "start_date": start.replace("-", ""),
            "end_date": end.replace("-", ""),
            "freq": frequency,
            "adj": adjustment,
            "api": ts.pro_api(api_token, timeout=request_timeout),
        }
        result = ts.pro_bar(**kwargs)
        if result is not None and not result.empty:
            frames.append(result)
    if not frames:
        raise RuntimeError("Tushare returned no rows for the requested symbols/date range")
    return normalize_ohlcv(pd.concat(frames, ignore_index=True))


def fetch_tushare_index(
    symbol: str,
    start: str,
    end: str,
    token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch an index daily series (for example 000300.SH) as market context."""
    api = _tushare_api(token, request_timeout)
    result = api.index_daily(ts_code=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    if result is None or result.empty:
        raise RuntimeError(f"Tushare returned no index rows for {symbol}")
    return normalize_ohlcv(result)


def fetch_tushare_stock_basic(token: str | None = None, request_timeout: float = 15.0) -> pd.DataFrame:
    """Fetch listed, delisted and paused securities for survivorship-safe universes."""
    api = _tushare_api(token, request_timeout)
    frames = []
    fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status"
    for status in ("L", "D", "P"):
        result = api.stock_basic(exchange="", list_status=status, fields=fields)
        if result is not None and not result.empty:
            frames.append(result)
    if not frames:
        raise RuntimeError("Tushare returned no stock_basic rows")
    data = pd.concat(frames, ignore_index=True)
    if "symbol" in data.columns:
        data = data.rename(columns={"symbol": "local_symbol"})
    return data.rename(columns={"ts_code": "symbol"}).drop_duplicates("symbol")


def _fetch_tushare_metadata(
    endpoint: str,
    symbols: Iterable[str],
    start: str,
    end: str,
    *,
    token: str | None = None,
    request_timeout: float = 15.0,
    extra_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fetch a symbol/date endpoint in bounded per-symbol requests."""
    api = _tushare_api(token, request_timeout)
    method = getattr(api, endpoint)
    frames: list[pd.DataFrame] = []
    for symbol in sorted(set(symbols)):
        result = method(
            ts_code=symbol,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            **(extra_params or {}),
        )
        if result is not None and not result.empty:
            frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_tushare_daily_basic(
    symbols: Iterable[str], start: str, end: str, *, token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch dated valuation/liquidity fields for point-in-time research joins."""
    return _fetch_tushare_metadata("daily_basic", symbols, start, end, token=token, request_timeout=request_timeout)


def fetch_tushare_stk_limit(
    symbols: Iterable[str], start: str, end: str, *, token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch historical price-limit bands; absence must be audited, not assumed."""
    return _fetch_tushare_metadata("stk_limit", symbols, start, end, token=token, request_timeout=request_timeout)


def fetch_tushare_suspend_d(
    symbols: Iterable[str], start: str, end: str, *, token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch dated suspension records (sparse: a missing row means no recorded suspension)."""
    return _fetch_tushare_metadata(
        "suspend_d", symbols, start, end, token=token,
        request_timeout=request_timeout, extra_params={"suspend_type": "S"},
    )


def fetch_tushare_sw_index_member_all(
    l1_codes: Iterable[str] | None = None,
    *,
    symbols: Iterable[str] | None = None,
    token: str | None = None,
    request_timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch Shenwan industry membership intervals.

    ``index_member_all`` is interval data rather than a daily time series.  Callers
    should normally provide the relevant L1 industry codes, which keeps requests
    bounded and makes the requested taxonomy explicit.  Passing ``None`` performs
    one endpoint request for installations whose Tushare entitlement supports it.
    """
    api = _tushare_api(token, request_timeout)
    if l1_codes is not None and symbols is not None:
        raise ValueError("Provide l1_codes or symbols, not both")
    requests = ([{"l1_code": code} for code in l1_codes] if l1_codes is not None else
                [{"ts_code": symbol} for symbol in symbols] if symbols is not None else [{}])
    frames: list[pd.DataFrame] = []
    for kwargs in requests:
        result = api.index_member_all(**kwargs)
        if result is not None and not result.empty:
            frames.append(result)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def attach_context(
    bars: pd.DataFrame,
    context: pd.DataFrame,
    prefix: str,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """As-of join market/sector/industry bars without leaking future observations.

    Context may have one common symbol, or `mapping` can map stock `symbol` to a
    `context_symbol`. Derived context return, volatility and relative strength are
    added with the supplied prefix.
    """
    base = normalize_ohlcv(bars)
    ctx = normalize_ohlcv(context)
    if mapping is None:
        unique = ctx["symbol"].dropna().unique()
        if len(unique) != 1:
            raise ValueError("mapping is required when context contains multiple symbols")
        base["context_symbol"] = unique[0]
    else:
        required = {"symbol", "context_symbol"}
        if not required.issubset(mapping.columns):
            raise ValueError(f"mapping must contain {sorted(required)}")
        base = base.merge(mapping[list(required)].drop_duplicates(), on="symbol", how="left")

    ctx = ctx.rename(columns={"symbol": "context_symbol"}).sort_values(["timestamp", "context_symbol"])
    ctx[f"{prefix}_return_20"] = ctx.groupby("context_symbol")["close"].pct_change(20)
    ctx[f"{prefix}_volatility_20"] = (
        ctx.groupby("context_symbol")["close"].pct_change().groupby(ctx["context_symbol"]).rolling(20).std().reset_index(level=0, drop=True)
    )
    keep = ["context_symbol", "timestamp", f"{prefix}_return_20", f"{prefix}_volatility_20"]
    parts = []
    for context_symbol, group in base.groupby("context_symbol", dropna=False):
        if pd.isna(context_symbol):
            parts.append(group)
            continue
        right = ctx.loc[ctx["context_symbol"] == context_symbol, keep].drop(columns="context_symbol")
        merged = pd.merge_asof(group.sort_values("timestamp"), right.sort_values("timestamp"), on="timestamp", direction="backward")
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp"])
    out[f"relative_strength_vs_{prefix}_20"] = out.groupby("symbol")["close"].pct_change(20) - out[f"{prefix}_return_20"]
    return out.reset_index(drop=True)
