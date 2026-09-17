"""CLI for bounded-memory full-market segment candidate construction."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import SegmentationConfig
from .fullmarket_segments import FullMarketSegmentBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable, non-semantic full-market segment candidates")
    parser.add_argument("--source", required=True, help="bulk_core directory containing manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--source-files-per-shard", type=int, default=20)
    parser.add_argument("--symbols-per-batch", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent bucket workers; parent alone writes manifest (default: 1)")
    parser.add_argument("--adjustment", choices=["hfq", "qfq"], default="hfq")
    parser.add_argument("--qfq-anchor-date", help="Required with qfq; only anchor-and-later bars are emitted")
    parser.add_argument("--max-buckets", type=int, help="Process at most N pending buckets, then exit cleanly")
    parser.add_argument("--max-shards", type=int, help="Process at most N pending date-to-symbol shards; buckets wait for all shards")
    parser.add_argument("--bootstrap-iterations", type=int, default=0,
                        help="Zero by default for scalable candidate construction")
    args = parser.parse_args()
    if args.adjustment == "qfq" and not args.qfq_anchor_date:
        parser.error("--qfq-anchor-date is required with --adjustment qfq")

    def emit(event: dict[str, object]) -> None:
        if event["event"] in {"shard_finished", "bucket_finished"}:
            print(" ".join(f"{k}={v}" for k, v in event.items()), flush=True)

    config = SegmentationConfig(bootstrap_iterations=args.bootstrap_iterations)
    result = FullMarketSegmentBuilder(
        args.source, args.output, bucket_count=args.buckets,
        source_files_per_shard=args.source_files_per_shard, symbols_per_batch=args.symbols_per_batch,
        bucket_workers=args.workers, adjustment=args.adjustment,
        qfq_anchor_date=args.qfq_anchor_date, segmentation=config,
        max_shards_per_run=args.max_shards, max_buckets_per_run=args.max_buckets, progress=emit,
    ).build()
    print(f"shards_complete={result.get('shards_complete')} buckets={len(result['buckets'])}/{args.buckets} "
          f"build_complete={result['build_complete']} "
          f"manifest={Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()
