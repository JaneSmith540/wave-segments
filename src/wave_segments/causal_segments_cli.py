"""Build point-in-time ATR-ZigZag segments and descriptive features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SegmentationConfig
from .causal_diagnostics import build_causal_state_diagnostics
from .discovery import DiscoveryConfig, walk_forward_discovery_states
from .features import extract_segment_features
from .segmentation import segment_ohlcv_causal
from .stability import bootstrap_causal_boundary_stability


def _read_bars(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("bars input must be Parquet or CSV")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, type=Path, help="OHLCV Parquet or CSV")
    parser.add_argument("--output", required=True, type=Path, help="New or existing output directory")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-reversal", type=float, default=2.2)
    parser.add_argument("--min-bars", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=0,
                        help="causal segment boundary perturbation runs; 0 disables (default)")
    parser.add_argument("--bootstrap-tolerance", type=int, default=3)
    parser.add_argument("--discover-oos", action="store_true",
                        help="also emit past-only temporary discovery states (not calibrated classes)")
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--retrain-every", type=int, default=20,
                        help="refit every N distinct availability dates")
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--min-cluster-samples", type=int, default=30)
    parser.add_argument("--min-cluster-symbols", type=int, default=3)
    parser.add_argument("--min-cluster-years", type=int, default=2)
    args = parser.parse_args()
    if not args.bars.is_file():
        parser.error(f"bars file does not exist: {args.bars}")
    args.output.mkdir(parents=True, exist_ok=True)
    bars = _read_bars(args.bars)
    config = SegmentationConfig(
        atr_period=args.atr_period, atr_reversal=args.atr_reversal, min_bars=args.min_bars,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_tolerance=args.bootstrap_tolerance,
    )
    segments = segment_ohlcv_causal(bars, config)
    bootstrap_summary = None
    if config.bootstrap_iterations > 0 and not segments.empty:
        _, stability = bootstrap_causal_boundary_stability(
            bars, config, iterations=config.bootstrap_iterations,
            tolerance=config.bootstrap_tolerance,
            price_noise=config.bootstrap_price_noise,
            volume_noise=config.bootstrap_volume_noise,
            random_state=42,
        )
        lookup = stability.set_index(["symbol", "boundary_idx"])["stability_frequency"].to_dict()
        segments["detector_start_boundary_probability"] = segments["start_boundary_probability"]
        segments["detector_end_boundary_probability"] = segments["end_boundary_probability"]
        start_stability = [lookup.get((row.symbol, int(row.start_idx)), 0.0) for row in segments.itertuples()]
        end_stability = [lookup.get((row.symbol, int(row.end_idx)), 0.0) for row in segments.itertuples()]
        segments["start_boundary_stability"] = start_stability
        segments["end_boundary_stability"] = end_stability
        segments["bootstrap_boundary_stability"] = np.minimum(start_stability, end_stability)
        segments["start_boundary_probability"] = np.minimum(
            segments["detector_start_boundary_probability"], segments["start_boundary_stability"],
        )
        segments["end_boundary_probability"] = np.minimum(
            segments["detector_end_boundary_probability"], segments["end_boundary_stability"],
        )
        segments["boundary_probability"] = np.minimum(
            segments["start_boundary_probability"], segments["end_boundary_probability"],
        )
        segments["boundary_uncertainty"] = 1.0 - segments["boundary_probability"]
        bootstrap_summary = {
            "iterations": config.bootstrap_iterations,
            "tolerance_bars": config.bootstrap_tolerance,
            "boundary_rows": int(len(stability)),
            "mean_frequency": float(stability["stability_frequency"].mean()),
            "p10_frequency": float(stability["stability_frequency"].quantile(.10)),
            "fraction_below_0_5": float((stability["stability_frequency"] < .5).mean()),
            "computed_on_confirmation_prefix_only": True,
            "interprets_as_boundary_accuracy": False,
        }
    features = extract_segment_features(bars, segments) if not segments.empty else segments.copy()
    segments_path = args.output / "causal_segments.parquet"
    features_path = args.output / "causal_segment_features.parquet"
    segments.to_parquet(segments_path, index=False)
    features.to_parquet(features_path, index=False)
    outputs = {"segments": segments_path.name, "features": features_path.name}
    discovery_summary = None
    if args.discover_oos:
        if features.empty:
            raise ValueError("cannot discover OOS states from an empty causal feature table")
        discovery_features = [
            "cumulative_return", "log_price_slope", "amplitude", "max_drawdown",
            "duration_bars", "return_volatility", "atr_pct_mean", "volume_cv",
            "up_volume_ratio", "body_ratio_mean", "upper_shadow_ratio_mean",
            "lower_shadow_ratio_mean", "boundary_uncertainty",
        ]
        discovery = DiscoveryConfig(
            n_components=args.components, ensemble_size=args.ensemble_size,
            min_cluster_samples=args.min_cluster_samples,
            min_cluster_symbols=args.min_cluster_symbols,
            min_cluster_years=args.min_cluster_years,
            random_state=42, include_hdbscan=False,
        )
        states = walk_forward_discovery_states(
            features, config=discovery, feature_columns=discovery_features,
            min_train_rows=args.min_train_rows, retrain_every=args.retrain_every,
        )
        trained = states["oos_train_rows"].gt(0)
        purged = states["unknown_reason"].eq("purged_overlap")
        if not (pd.to_datetime(states["available_at"]) == pd.to_datetime(states["state_available_at"])).all():
            raise RuntimeError("causal audit failed: state availability differs from segment availability")
        if trained.any() and not (pd.to_datetime(states.loc[trained, "oos_train_max_available_at"])
                                  < pd.to_datetime(states.loc[trained, "state_available_at"])).all():
            raise RuntimeError("causal audit failed: training data is not strictly prior to scoring")
        same_symbol_train = states["oos_train_max_end"].notna()
        if (same_symbol_train & ~purged).any() and not (
            pd.to_datetime(states.loc[same_symbol_train & ~purged, "oos_train_max_end"])
            < pd.to_datetime(states.loc[same_symbol_train & ~purged, "start"])
        ).all():
            raise RuntimeError("causal audit failed: same-symbol training/scoring intervals overlap")
        probability_columns = [c for c in states if c.startswith("ensemble_prob_")]
        probability_rows = states[probability_columns].notna().any(axis=1)
        if probability_columns and probability_rows.any() and not np.allclose(
            states.loc[probability_rows, probability_columns].sum(axis=1), 1.0, atol=1e-6,
        ):
            raise RuntimeError("probability audit failed: ensemble probabilities do not sum to one")
        if not (states["is_unknown"] == states["label"].eq("UNKNOWN")).all():
            raise RuntimeError("state audit failed: UNKNOWN flag and label disagree")
        states_path = args.output / "causal_oos_discovery_states.parquet"
        states.to_parquet(states_path, index=False)
        outputs["oos_discovery_states"] = states_path.name
        diagnostic_paths = {}
        for name, table in build_causal_state_diagnostics(states).items():
            path = args.output / f"{name}.csv"
            table.to_csv(path, index=False)
            diagnostic_paths[name] = path.name
        outputs["diagnostics"] = diagnostic_paths
        discovery_summary = {
            "rows": int(len(states)),
            "identified_fraction": float((~states["is_unknown"]).mean()),
            "aggregate_label_counts_not_comparable": {
                str(k): int(v) for k, v in states["label"].value_counts(dropna=False).items()
            },
            "unknown_reason_counts": {str(k): int(v) for k, v in states["unknown_reason"].value_counts(dropna=False).items()},
            "first_trained_at": (str(states["oos_model_trained_at"].dropna().min())
                                 if states["oos_model_trained_at"].notna().any() else None),
            "last_trained_at": (str(states["oos_model_trained_at"].dropna().max())
                                if states["oos_model_trained_at"].notna().any() else None),
            "features": discovery_features,
            "classes_are_semantic_or_calibrated": False,
            "cluster_ids_comparable_across_model_versions": False,
            "aggregate_label_counts_are_fold_local_names": True,
            "config": {
                "min_train_rows": args.min_train_rows, "retrain_every_distinct_dates": args.retrain_every,
                "n_components": args.components, "ensemble_size": args.ensemble_size,
                "min_cluster_samples": args.min_cluster_samples,
                "min_cluster_symbols": args.min_cluster_symbols,
                "min_cluster_years": args.min_cluster_years, "random_state": 42,
                "include_hdbscan": False,
            },
            "causal_audit": {
                "state_availability_matches_segment": True,
                "training_availability_strictly_prior": True,
                "same_symbol_overlap_purged": True,
                "ensemble_probabilities_normalized_when_present": True,
                "unknown_flag_matches_label": True,
            },
        }
    summary = {
        "status": "complete",
        "mode": ("causal_atr_zigzag_with_past_only_discovery_states"
                 if args.discover_oos else "causal_atr_zigzag_candidates_only"),
        "bars_path": str(args.bars.resolve()),
        "bars_rows": int(len(bars)),
        "symbols": int(bars.get("symbol", bars.get("ts_code", pd.Series(dtype=str))).nunique()),
        "segments": int(len(segments)),
        "unclosed_tail_emitted": False,
        "availability_field": "available_at",
        "boundary_probability_is_calibrated": False,
        "selection_research_eligible": False,
        "config": {"atr_period": config.atr_period, "atr_reversal": config.atr_reversal,
                   "min_bars": config.min_bars, "bootstrap_iterations": config.bootstrap_iterations,
                   "bootstrap_tolerance": config.bootstrap_tolerance,
                   "bootstrap_price_noise": config.bootstrap_price_noise,
                   "bootstrap_volume_noise": config.bootstrap_volume_noise},
        "discovery": discovery_summary,
        "boundary_stability": bootstrap_summary,
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
