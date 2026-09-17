"""Resumable, point-in-time-aware market dataset construction."""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import fetch_tushare
from .schema import normalize_ohlcv


@dataclass(frozen=True)
class DatasetSpec:
    start: str
    end: str
    frequency: str = "D"
    adjustment: str | None = "qfq"
    chunk_months: int = 3
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    request_timeout_seconds: float = 15.0


def point_in_time_universe(stock_basic: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep securities listed during any part of the requested interval.

    Including delisted securities is essential; passing only today's constituents
    would create survivorship bias in downstream selection research.
    """
    data = stock_basic.rename(columns={"ts_code": "symbol"}).copy()
    if not {"symbol", "list_date"}.issubset(data):
        raise ValueError("stock_basic requires symbol/ts_code and list_date")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    data["list_date"] = pd.to_datetime(data["list_date"], errors="coerce")
    data["delist_date"] = pd.to_datetime(data.get("delist_date"), errors="coerce")
    mask = data.list_date.le(end_ts) & (data.delist_date.isna() | data.delist_date.ge(start_ts))
    return data.loc[mask].sort_values("symbol").reset_index(drop=True)


def date_chunks(start: str, end: str, months: int = 3) -> list[tuple[str, str]]:
    if months < 1:
        raise ValueError("months must be positive")
    left, final = pd.Timestamp(start), pd.Timestamp(end)
    if left > final:
        raise ValueError("start must not be after end")
    chunks = []
    while left <= final:
        right = min(left + pd.DateOffset(months=months) - pd.Timedelta(days=1), final)
        chunks.append((left.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
        left = right + pd.Timedelta(days=1)
    return chunks


class ResumableDatasetBuilder:
    """Download symbol/date partitions and safely resume completed chunks."""

    def __init__(
        self,
        cache_dir: str | Path,
        fetcher: Callable[..., pd.DataFrame] = fetch_tushare,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetcher = fetcher

    def build(
        self,
        symbols: Iterable[str],
        spec: DatasetSpec,
        *,
        force: bool = False,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> pd.DataFrame:
        symbols = sorted(set(symbols))
        if not symbols:
            raise ValueError("at least one symbol is required")
        partitions = self.cache_dir / "partitions"
        partitions.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        failures: list[dict[str, str]] = []
        for symbol in symbols:
            safe_symbol = symbol.replace(".", "_")
            for start, end in date_chunks(spec.start, spec.end, spec.chunk_months):
                path = partitions / f"{safe_symbol}_{start}_{end}.parquet"
                paths.append(path)
                if path.exists() and not force:
                    if progress: progress({"status": "cached", "symbol": symbol, "start": start, "end": end})
                    continue
                last_error: Exception | None = None
                for attempt in range(max(1, spec.max_retries)):
                    try:
                        frame = self.fetcher(
                            [symbol], start, end, frequency=spec.frequency,
                            adjustment=spec.adjustment,
                            request_timeout=spec.request_timeout_seconds,
                        )
                        temporary = path.with_suffix(".parquet.tmp")
                        normalize_ohlcv(frame).to_parquet(temporary, index=False)
                        temporary.replace(path)
                        if progress: progress({"status": "downloaded", "symbol": symbol, "start": start, "end": end, "rows": len(frame)})
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - injected/vendor fetcher failures are retried and recorded
                        last_error = exc
                        if attempt + 1 < max(1, spec.max_retries):
                            time.sleep(spec.retry_delay_seconds * (attempt + 1))
                if last_error is not None:
                    failures.append({"symbol": symbol, "start": start, "end": end, "error": str(last_error)})
                    if progress: progress({"status": "failed", "symbol": symbol, "start": start, "end": end, "error": str(last_error)})
        available = [pd.read_parquet(path) for path in paths if path.exists()]
        manifest = {
            "spec": asdict(spec), "symbols_requested": symbols,
            "partitions_expected": len(paths), "partitions_available": len(available),
            "failures": failures,
            "survivorship_warning": "Use point_in_time_universe including delisted securities.",
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not available:
            raise RuntimeError(f"No partitions available; failures={len(failures)}")
        return normalize_ohlcv(pd.concat(available, ignore_index=True))
