"""CLI for the resumable bulk Tushare daily dataset builder."""
from __future__ import annotations

import argparse
import os

from .bulk_market_data import BulkMarketBuilder, read_security_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tushare full-market daily partitions by trade_date")
    parser.add_argument("--start", required=True); parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--universe")
    parser.add_argument("--include", choices=["daily_basic", "stk_limit", "suspend_d"], action="append", default=[])
    parser.add_argument("--retries", type=int, default=3); parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-dates", type=int, help="Process at most this many uncached dates, then exit cleanly")
    parser.add_argument("--show-cached", action="store_true", help="Print one event for every already cached date")
    parser.add_argument("--progress-every", type=int, default=1, help="Print every Nth completed new date")
    args = parser.parse_args()
    token = next((os.environ.get(k) for k in ("TUSHARE_API_TOKEN", "TUSHARE_TOKEN", "TS_TOKEN") if os.environ.get(k)), None)
    if not token: raise SystemExit("Set TUSHARE_API_TOKEN (or TUSHARE_TOKEN / TS_TOKEN); token is never written to output.")
    try:
        import tushare as ts
    except ImportError as exc: raise SystemExit("Install optional dependency: pip install -e '.[data]'") from exc
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    progress_state = {"finished": 0}

    def emit(event: dict[str, object]) -> None:
        kind = event.get("event")
        if kind == "date_cached":
            if args.show_cached:
                print(" ".join(f"{k}={v}" for k, v in event.items()), flush=True)
            return
        endpoint_failed = kind == "endpoint_finished" and event.get("status") != "ok"
        if kind == "date_finished":
            progress_state["finished"] += 1
            if progress_state["finished"] % args.progress_every == 0 or event.get("status") != "complete":
                print(" ".join(f"{k}={v}" for k, v in event.items()), flush=True)
        elif endpoint_failed:
            print(" ".join(f"{k}={v}" for k, v in event.items()), flush=True)

    manifest = BulkMarketBuilder(ts.pro_api(token, timeout=args.timeout), args.output, args.start, args.end,
        universe=read_security_universe(args.universe) if args.universe else None,
        optional_endpoints=args.include, retries=args.retries, timeout_seconds=args.timeout,
        max_dates_per_run=args.max_dates,
        progress=emit).build()
    complete = sum(x.get("status") == "complete" for x in manifest["dates"].values())
    print(f"complete_partitions={complete} build_complete={manifest['build_complete']} "
          f"processed_this_run={manifest['last_run_processed_dates']} manifest={args.output}/manifest.json")

if __name__ == "__main__": main()
