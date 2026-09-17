"""Apply past-only cluster discovery to precomputed causal segment features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_diagnostics import build_causal_state_diagnostics
from .discovery import DiscoveryConfig, walk_forward_discovery_states

DEFAULT_FEATURES = [
    "cumulative_return", "log_price_slope", "amplitude", "max_drawdown",
    "duration_bars", "return_volatility", "atr_pct_mean", "volume_cv",
    "up_volume_ratio", "body_ratio_mean", "upper_shadow_ratio_mean",
    "lower_shadow_ratio_mean", "boundary_uncertainty",
]


def _verify_states(states: pd.DataFrame) -> dict[str, bool]:
    trained = states["oos_train_rows"].gt(0)
    purged = states["unknown_reason"].eq("purged_overlap")
    availability_match = bool((pd.to_datetime(states["available_at"])
                               == pd.to_datetime(states["state_available_at"])).all())
    train_prior = bool(not trained.any() or (
        pd.to_datetime(states.loc[trained, "oos_train_max_available_at"])
        < pd.to_datetime(states.loc[trained, "state_available_at"])
    ).all())
    same_symbol = states["oos_train_max_end"].notna()
    overlap_purged = bool(not (same_symbol & ~purged).any() or (
        pd.to_datetime(states.loc[same_symbol & ~purged, "oos_train_max_end"])
        < pd.to_datetime(states.loc[same_symbol & ~purged, "start"])
    ).all())
    columns = [c for c in states if c.startswith("ensemble_prob_")]
    rows = states[columns].notna().any(axis=1)
    probabilities_normalized = bool(not columns or not rows.any() or np.allclose(
        states.loc[rows, columns].sum(axis=1), 1.0, atol=1e-6,
    ))
    unknown_consistent = bool((states["is_unknown"] == states["label"].eq("UNKNOWN")).all())
    audit = {
        "state_availability_matches_segment": availability_match,
        "training_availability_strictly_prior": train_prior,
        "same_symbol_overlap_purged": overlap_purged,
        "ensemble_probabilities_normalized_when_present": probabilities_normalized,
        "unknown_flag_matches_label": unknown_consistent,
    }
    if not all(audit.values()):
        raise RuntimeError(f"causal state audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Parquet from causal segment pipeline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--retrain-every", type=int, default=20)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--min-cluster-samples", type=int, default=30)
    parser.add_argument("--min-cluster-symbols", type=int, default=3)
    parser.add_argument("--min-cluster-years", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not args.features.is_file():
        parser.error(f"features file not found: {args.features}")
    args.output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    config = DiscoveryConfig(
        n_components=args.components, ensemble_size=args.ensemble_size,
        min_cluster_samples=args.min_cluster_samples,
        min_cluster_symbols=args.min_cluster_symbols,
        min_cluster_years=args.min_cluster_years,
        max_iter=args.max_iter, random_state=args.random_state,
        include_hdbscan=False,
    )
    states = walk_forward_discovery_states(
        features, config=config, feature_columns=DEFAULT_FEATURES,
        min_train_rows=args.min_train_rows, retrain_every=args.retrain_every,
    )
    causal_audit = _verify_states(states)
    states_path = args.output / "causal_oos_discovery_states.parquet"
    states.to_parquet(states_path, index=False)
    diagnostics = {}
    for name, table in build_causal_state_diagnostics(states).items():
        path = args.output / f"{name}.csv"
        table.to_csv(path, index=False)
        diagnostics[name] = path.name
    trained = states.loc[states["oos_model_trained_at"].notna()]
    convergence = (
        trained.groupby("oos_model_trained_at")["mixture_convergence_valid"].first()
        if "mixture_convergence_valid" in trained else pd.Series(dtype=bool)
    )
    summary = {
        "status": "complete", "mode": "past_only_unsupervised_discovery_review_only",
        "features_path": str(args.features.resolve()), "rows": len(states),
        "identified_fraction": float((~states["is_unknown"]).mean()),
        "label_counts_not_semantic": {str(k): int(v) for k, v in states["label"].value_counts().items()},
        "unknown_reason_counts": {str(k): int(v) for k, v in states["unknown_reason"].value_counts().items()},
        "model_versions": len(convergence),
        "nonconverged_model_versions": int((~convergence.astype(bool)).sum()),
        "mixture_nonconverged_reason_rows": int(states["unknown_reason"].str.contains("mixture_nonconverged").sum()),
        "cluster_ids_comparable_across_versions": False,
        "selection_research_eligible": False,
        "config": {
            "min_train_rows": args.min_train_rows, "retrain_every_distinct_dates": args.retrain_every,
            "n_components": args.components, "ensemble_size": args.ensemble_size,
            "min_cluster_samples": args.min_cluster_samples,
            "min_cluster_symbols": args.min_cluster_symbols,
            "min_cluster_years": args.min_cluster_years,
            "max_iter": args.max_iter, "random_state": args.random_state,
            "features": DEFAULT_FEATURES,
        },
        "causal_audit": causal_audit,
        "outputs": {"states": states_path.name, "diagnostics": diagnostics},
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
