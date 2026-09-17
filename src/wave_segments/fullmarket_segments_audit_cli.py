"""CLI audit for full-market segment candidate artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fullmarket_segments import audit_fullmarket_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit full-market candidate segment buckets")
    parser.add_argument("--output", required=True, help="full-market candidate output directory")
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail the audit unless every configured bucket is complete")
    args = parser.parse_args()
    result = audit_fullmarket_segments(args.output, require_complete=args.require_complete)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = result["summary"]
    print(f"passed={result['passed']} buckets={summary['buckets_complete']}/{result['bucket_count']} "
          f"segments={summary['segments']} issues={len(result['issues'])}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
