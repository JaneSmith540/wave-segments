from __future__ import annotations

import pandas as pd

REQUIRED_OHLCV = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common vendor columns and preserve optional amount/context columns."""
    aliases = {"ts_code": "symbol", "trade_date": "timestamp", "vol": "volume"}
    out = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns}).copy()
    missing = [c for c in REQUIRED_OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"])
    out = out.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    out[numeric] = out.groupby("symbol", group_keys=False)[numeric].apply(lambda x: x.ffill())
    out["volume"] = out["volume"].fillna(0.0)
    return out.reset_index(drop=True)
