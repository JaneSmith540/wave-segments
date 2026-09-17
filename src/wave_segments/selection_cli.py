"""Command-line research report for the isolated stock-selection layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .selection import (
    assess_selection_evidence,
    build_selection_dataset,
    evaluate_selection,
    walk_forward_cross_sectional_scores,
)


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _compact(report: dict[str, object]) -> dict[str, object]:
    keys = ("rank_ic", "icir", "rank_ic_hac_t", "rank_ic_positive_rate",
            "monotonicity", "mean_net_long_short", "max_drawdown")
    return {k: float(report[k]) if report[k] is not None and np.isfinite(report[k]) else None for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen wave-state scores; produces no orders")
    parser.add_argument("--bars", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--scores", nargs="+", default=[])
    parser.add_argument("--fit-features", nargs="+", default=[])
    parser.add_argument("--output", default="outputs/selection_validation.json")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--benchmark")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=0)
    parser.add_argument("--slippage-bps", type=float, default=0)
    parser.add_argument("--min-train-dates", type=int, default=252)
    parser.add_argument("--min-train-rows", type=int)
    parser.add_argument("--retrain-every", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    if not args.scores and not args.fit_features:
        parser.error("provide --scores and/or --fit-features")
    dataset = build_selection_dataset(_load(args.bars), _load(args.states), args.horizons, args.benchmark)
    results = {}
    for horizon in args.horizons:
        target = f"future_excess_{horizon}d"
        for score in args.scores:
            compact = _compact(evaluate_selection(dataset, score, target, args.groups, args.cost_bps, args.slippage_bps))
            compact["evidence_gate"] = assess_selection_evidence(compact)
            results[f"{score}@{horizon}d"] = compact
        if args.fit_features:
            scored = walk_forward_cross_sectional_scores(
                dataset, args.fit_features, target,
                min_train_dates=args.min_train_dates,
                min_train_rows=args.min_train_rows,
                retrain_every=args.retrain_every,
                ridge_alpha=args.ridge_alpha,
            )
            evaluation = evaluate_selection(
                scored, "oos_score", target, args.groups, args.cost_bps, args.slippage_bps
            )
            compact = _compact(evaluation)
            compact["scored_rows"] = int(scored["oos_score"].notna().sum())
            compact["causal_audit_passed"] = bool(
                (scored.loc[scored["oos_score"].notna(), "oos_train_max_target_available_at"]
                 < scored.loc[scored["oos_score"].notna(), "timestamp"]).all()
            )
            compact["evidence_gate"] = assess_selection_evidence(compact)
            results[f"ridge_oos@{horizon}d"] = compact
    report = {"results": results,
              "warning": "Research metrics only. Frozen --scores must already be out-of-sample; --fit-features uses strict causal walk-forward fitting."}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evaluations={len(results)} output={output}")


if __name__ == "__main__":
    main()
