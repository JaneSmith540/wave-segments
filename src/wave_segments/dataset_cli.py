"""Build resumable multi-year datasets without embedding credentials."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import fetch_tushare_stock_basic
from .dataset import DatasetSpec, ResumableDatasetBuilder, point_in_time_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a partitioned Tushare OHLCV dataset")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--all-market", action="store_true")
    parser.add_argument("--frequency", default="D")
    parser.add_argument("--adjustment", choices=["qfq", "hfq", "none"], default="qfq")
    parser.add_argument("--chunk-months", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--cache", default="data/market_dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if bool(args.symbols) == bool(args.all_market):
        parser.error("choose exactly one of --symbols or --all-market")
    cache = Path(args.cache)
    if args.all_market:
        basic = fetch_tushare_stock_basic(request_timeout=args.request_timeout)
        universe = point_in_time_universe(basic, args.start, args.end)
        cache.mkdir(parents=True, exist_ok=True)
        universe.to_parquet(cache / "point_in_time_universe.parquet", index=False)
        symbols = universe.symbol.tolist()
    else:
        symbols = args.symbols
    spec = DatasetSpec(
        start=args.start, end=args.end, frequency=args.frequency,
        adjustment=None if args.adjustment == "none" else args.adjustment,
        chunk_months=args.chunk_months,
        request_timeout_seconds=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay,
    )
    data = ResumableDatasetBuilder(cache).build(
        symbols, spec, force=args.force,
        progress=lambda event: print(
            f"{event['status']} {event['symbol']} {event['start']}..{event['end']}"
            + (f" rows={event['rows']}" if "rows" in event else ""),
            flush=True,
        ),
    )
    output = cache / "dataset.parquet"
    data.to_parquet(output, index=False)
    print(f"symbols={data.symbol.nunique()} rows={len(data)} output={output}")


if __name__ == "__main__":
    main()
