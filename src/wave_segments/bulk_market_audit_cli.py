"""CLI for proving completeness and integrity of a bulk market build."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .bulk_market_data import (
    BulkMarketBuilder,
    audit_bulk_market_manifest,
    read_security_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tushare bulk market partitions against the exchange calendar")
    parser.add_argument("--output", required=True, help="Bulk dataset directory containing manifest.json")
    parser.add_argument("--universe", help="Optional point-in-time security master")
    parser.add_argument("--start", help="Override calendar start date")
    parser.add_argument("--end", help="Override calendar end date")
    parser.add_argument("--quick", action="store_true", help="Skip parquet content checks")
    parser.add_argument("--minimum-adj-factor-match", type=float, default=0.999)
    parser.add_argument("--minimum-universe-coverage", type=float)
    parser.add_argument("--require-endpoint", choices=("daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d"),
                        action="append", default=None,
                        help="Require this endpoint on every expected session (repeatable)")
    parser.add_argument("--report", help="Write the complete JSON report here")
    args = parser.parse_args()

    root = Path(args.output)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("expected_open_dates")
    if expected is None or args.start or args.end:
        start = args.start or manifest.get("start_date")
        end = args.end or manifest.get("end_date")
        if not start or not end:
            raise SystemExit("Old manifest requires --start and --end to reconstruct the exchange calendar")
        token = next((os.environ.get(k) for k in ("TUSHARE_API_TOKEN", "TUSHARE_TOKEN", "TS_TOKEN")
                      if os.environ.get(k)), None)
        if not token:
            raise SystemExit("Set TUSHARE_API_TOKEN to reconstruct the exchange calendar for this manifest")
        try:
            import tushare as ts
        except ImportError as exc:
            raise SystemExit("Install optional dependency: pip install -e '.[data]'") from exc
        expected = BulkMarketBuilder(ts.pro_api(token, timeout=30), root, start, end).trade_dates()

    report = audit_bulk_market_manifest(
        root,
        expected_trade_dates=expected,
        universe=read_security_universe(args.universe) if args.universe else None,
        required_endpoints=args.require_endpoint,
        verify_partitions=not args.quick,
        minimum_adj_factor_match=args.minimum_adj_factor_match,
        minimum_universe_coverage=args.minimum_universe_coverage,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if not report["summary"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
