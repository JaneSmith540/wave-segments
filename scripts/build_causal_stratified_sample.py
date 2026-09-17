"""Create a deterministic multi-year HFQ sample from completed symbol shards."""
from __future__ import annotations

import argparse
import json
from hashlib import blake2b
from pathlib import Path

import pandas as pd

from wave_segments.fullmarket_segments import _prices_for_segmentation


def _rank(seed: int, symbol: str) -> bytes:
    return blake2b(f"{seed}:{symbol}".encode(), digest_size=16).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=Path("outputs/fullmarket_candidates_hfq"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-bucket", type=int, default=1)
    parser.add_argument("--minimum-bars", type=int, default=1000,
                        help="require this many historical daily bars per selected stock")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--buckets", type=int, default=64)
    args = parser.parse_args()
    if args.sample_per_bucket < 1 or args.buckets < 1 or args.minimum_bars < 2:
        parser.error("sample-per-bucket/buckets must be positive and minimum-bars >= 2")
    manifest_path = args.candidate_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("build_complete"):
        parser.error("candidate build manifest is not complete")

    selected: dict[str, list[str]] = {}
    bars: list[pd.DataFrame] = []
    for bucket in range(args.buckets):
        key = f"{bucket:03d}"
        feature_path = args.candidate_root / "segment_features" / f"bucket={key}" / "data.parquet"
        shard_dir = args.candidate_root / "_symbol_shards" / f"bucket={key}"
        if not feature_path.is_file() or not shard_dir.is_dir():
            parser.error(f"missing feature or symbol shard for bucket {key}")
        features = pd.read_parquet(feature_path, columns=["symbol"])
        shards = sorted(shard_dir.glob("shard=*.parquet"))
        if not shards:
            parser.error(f"no source shards for bucket {key}")
        raw = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
        counts = raw.groupby("ts_code").size()
        eligible = set(features["symbol"].astype(str)) & set(counts[counts >= args.minimum_bars].index.astype(str))
        symbols = sorted(eligible, key=lambda s: _rank(args.seed, s))
        if len(symbols) < args.sample_per_bucket:
            parser.error(f"bucket {key} has only {len(symbols)} symbols with >= {args.minimum_bars} bars")
        chosen = symbols[:args.sample_per_bucket]
        selected[key] = chosen
        raw = raw.loc[raw["ts_code"].astype(str).isin(chosen)]
        if raw.empty:
            parser.error(f"selected symbols have no source bars in bucket {key}")
        adjusted = _prices_for_segmentation(raw, "hfq", None)
        adjusted = adjusted.rename(columns={"ts_code": "symbol", "trade_date": "timestamp", "vol": "volume"})
        bars.append(adjusted)
    sample = pd.concat(bars, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    bars_path = args.output / "bars_hfq.parquet"
    sample.to_parquet(bars_path, index=False)
    selection = [symbol for values in selected.values() for symbol in values]
    summary = {
        "status": "complete", "sampling": "deterministic hash-rank within stable symbol buckets",
        "seed": args.seed, "sample_per_bucket": args.sample_per_bucket,
        "minimum_bars": args.minimum_bars, "bucket_count": args.buckets, "selected_symbols": selection,
        "symbol_count": len(selection), "bar_rows": len(sample),
        "bars_per_symbol": {str(k): int(v) for k, v in sample.groupby("symbol").size().items()},
        "date_min": str(pd.to_datetime(sample.timestamp).min()),
        "date_max": str(pd.to_datetime(sample.timestamp).max()),
        "adjustment": "HFQ (raw OHLC multiplied by each bar's as-of adj_factor)",
        "source_candidate_manifest": str(manifest_path.resolve()),
        "bars_file": bars_path.name,
        "warning": "This is a deterministic bucket sample, not a full-market result or semantic label truth.",
    }
    (args.output / "sample_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
