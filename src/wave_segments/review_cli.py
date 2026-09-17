"""Build a reproducible, stratified human-review queue."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .review import build_stratified_review_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified wave annotation sample")
    parser.add_argument("--segments", required=True)
    parser.add_argument("--target-size", required=True, type=int)
    parser.add_argument("--unknown-share", type=float, default=0.35)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="outputs/annotation_sample.csv")
    args = parser.parse_args()
    path = Path(args.segments)
    if path.is_dir():
        # Full-market candidate construction emits a parquet dataset partitioned
        # by symbol bucket.  Read only the review-facing table, not raw bars or
        # the (wider) feature dataset.
        try:
            import pyarrow.dataset as ds
            data = ds.dataset(path, format="parquet").to_table().to_pandas()
        except Exception as exc:
            raise SystemExit(f"Could not read parquet dataset {path}: {exc}") from exc
    else:
        data = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    queue = build_stratified_review_sample(
        data, args.target_size, unknown_share=args.unknown_share, random_state=args.random_state
    )
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"sampled={len(queue)} unknown={(queue.predicted_label.str.upper() == 'UNKNOWN').sum()} output={output}")


if __name__ == "__main__":
    main()
