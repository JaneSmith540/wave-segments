"""Enrich OHLCV with point-in-time market metadata and write a quality audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .market_data import enrich_and_audit_market_data


def _load(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    source = Path(path)
    return pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-in-time market metadata enrichment and audit")
    parser.add_argument("--bars", required=True)
    parser.add_argument("--daily-basic")
    parser.add_argument("--stk-limit")
    parser.add_argument("--suspend-d")
    parser.add_argument("--suspend-query-complete", action="store_true")
    parser.add_argument("--sw-membership")
    parser.add_argument("--expected-universe")
    parser.add_argument("--output", default="data/enriched_market.parquet")
    parser.add_argument("--report", default="data/market_quality_report.json")
    args = parser.parse_args()
    result = enrich_and_audit_market_data(
        _load(args.bars), daily_basic=_load(args.daily_basic), stk_limit=_load(args.stk_limit),
        suspend_d=_load(args.suspend_d), suspend_query_complete=args.suspend_query_complete,
        sw_membership=_load(args.sw_membership), expected_universe=_load(args.expected_universe),
    )
    output, report = Path(args.output), Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True); report.parent.mkdir(parents=True, exist_ok=True)
    result.enriched.to_parquet(output, index=False)
    report.write_text(json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(result.enriched)} tradability_known={result.report['tradability']['known_coverage']:.1%} output={output}")


if __name__ == "__main__":
    main()
