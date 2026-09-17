"""Test cluster-label stability when causal training folds are resampled by symbol."""
from __future__ import annotations

import argparse
import json
from hashlib import blake2b
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wave_segments.class_robustness import probability_total_variation
from wave_segments.cluster_bootstrap import (
    align_bootstrap_to_reference,
    remap_bootstrap_predictions,
    resample_symbols_with_replacement,
)
from wave_segments.discovery import DiscoveryConfig, MultiModelDiscoverer
from wave_segments.market_regimes import sample_market_regime


def _seed(base: int, model_time: str, replicate: int) -> int:
    value = f"{base}|{model_time}|symbol-bootstrap|{replicate}".encode()
    return int.from_bytes(blake2b(value, digest_size=4).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bars", required=True, help="daily sample bars used to derive causal market proxy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260917)
    parser.add_argument("--regime-window", type=int, default=20)
    parser.add_argument("--regime-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    features = pd.read_parquet(args.features)
    states = pd.read_parquet(args.states)
    bars = pd.read_parquet(args.bars)
    market_regimes = sample_market_regime(
        bars, window=args.regime_window, trend_threshold=args.regime_threshold,
    )
    market_regimes.to_csv(output / "sample_market_regime_proxy.csv", index=False, encoding="utf-8-sig")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    settings = manifest["config"]
    feature_columns = list(settings["features"])
    probability_columns = sorted(c for c in states if c.startswith("prob_CLUSTER_"))
    if not probability_columns:
        raise ValueError("No cluster probability fields found")

    features["available_at"] = pd.to_datetime(features["available_at"], errors="coerce")
    states["oos_model_trained_at"] = pd.to_datetime(states["oos_model_trained_at"], errors="coerce")
    scored = states.loc[states.oos_model_trained_at.notna()].copy()
    scored = scored.loc[scored[probability_columns].notna().any(axis=1)]
    by_id = features.set_index("segment_id", drop=False)
    config = DiscoveryConfig(
        n_components=int(settings["n_components"]), random_state=int(settings["random_state"]),
        ensemble_size=int(settings["ensemble_size"]),
        min_cluster_samples=int(settings["min_cluster_samples"]),
        min_cluster_symbols=int(settings["min_cluster_symbols"]),
        min_cluster_years=int(settings["min_cluster_years"]),
        max_iter=int(settings["max_iter"]), include_hdbscan=False,
    )

    metric_rows, class_rows, fold_rows, slice_rows = [], [], [], []
    versions = sorted(scored.oos_model_trained_at.dropna().unique())
    for version_value in versions:
        version = pd.Timestamp(version_value)
        version_text = version.isoformat()
        saved = scored.loc[scored.oos_model_trained_at.eq(version)].copy()
        expected = int(saved.oos_train_rows.dropna().iloc[0])
        training = features.loc[features.available_at < version].copy()
        training = training.sort_values(["available_at", "symbol", "segment_id"], kind="stable")
        if len(training) != expected:
            raise RuntimeError(f"fold {version_text}: training-row mismatch {len(training)} != {expected}")
        score_ids = saved.segment_id.astype(str).tolist()
        score = by_id.loc[score_ids].reset_index(drop=True)
        score_context = saved[["segment_id", "available_at"]].reset_index(drop=True).copy()
        score_context["available_at"] = pd.to_datetime(score_context.available_at, errors="coerce")
        score_context["year"] = score_context.available_at.dt.year.astype("Int64").astype("string")
        score_context = score_context.merge(
            market_regimes[["timestamp", "market_proxy_regime"]],
            left_on="available_at", right_on="timestamp", how="left", validate="many_to_one",
        )
        score_context["market_proxy_regime"] = score_context.market_proxy_regime.fillna("UNKNOWN")

        reference = MultiModelDiscoverer(config, feature_columns).fit(training)
        baseline = reference.predict(score)
        baseline_labels = baseline.label.astype(str).to_numpy()
        saved_labels = saved.label.astype(str).to_numpy()
        saved_prob = saved[probability_columns].to_numpy(float)
        baseline_prob = baseline[probability_columns].to_numpy(float)
        valid_prob = np.isfinite(saved_prob).all(axis=1) & np.isfinite(baseline_prob).all(axis=1)
        fold_rows.append({
            "oos_model_trained_at": version_text,
            "train_rows": len(training),
            "score_rows": len(score),
            "training_symbols": int(training.symbol.nunique()),
            "saved_label_reproduction": float(np.mean(saved_labels == baseline_labels)),
            "max_saved_probability_abs_delta": float(np.max(np.abs(saved_prob[valid_prob] - baseline_prob[valid_prob]))) if valid_prob.any() else np.nan,
            "gmm_converged": reference.gmm_converged_,
            "dpgmm_converged": reference.dpgmm_converged_,
        })

        for replicate in range(args.replicates):
            sample, distinct_drawn_symbols = resample_symbols_with_replacement(
                training, random_state=_seed(args.random_state, version_text, replicate),
            )
            bootstrap = MultiModelDiscoverer(config, feature_columns).fit(sample)
            mapping, prototype_rms = align_bootstrap_to_reference(reference, bootstrap)
            raw_prediction = bootstrap.predict(score)
            prediction = remap_bootstrap_predictions(raw_prediction, mapping, reference.cluster_labels_)
            perturbed_labels = prediction.label.astype(str).to_numpy()
            original_identified = baseline_labels != "UNKNOWN"
            tv = probability_total_variation(baseline, prediction, probability_columns)
            metric_rows.append({
                "oos_model_trained_at": version_text,
                "replicate": replicate,
                "rows": len(score),
                "baseline_identified_rows": int(original_identified.sum()),
                "bootstrap_rows": len(sample),
                "training_symbols": int(training.symbol.nunique()),
                "distinct_symbols_drawn": distinct_drawn_symbols,
                "prototype_alignment_rms_train_scale": prototype_rms,
                "gmm_converged": bootstrap.gmm_converged_,
                "dpgmm_converged": bootstrap.dpgmm_converged_,
                "cluster_valid_count": int(sum(bootstrap.cluster_valid_)),
                "label_agreement_all": float(np.mean(baseline_labels == perturbed_labels)),
                "candidate_agreement_all": float(np.mean(
                    baseline.candidate_label.astype(str).to_numpy() == prediction.candidate_label.astype(str).to_numpy()
                )),
                "label_agreement_baseline_identified": float(np.mean(
                    baseline_labels[original_identified] == perturbed_labels[original_identified]
                )) if original_identified.any() else np.nan,
                "coverage_after_bootstrap": float(np.mean(perturbed_labels != "UNKNOWN")),
                "baseline_identified_retention": float(np.mean(
                    perturbed_labels[original_identified] != "UNKNOWN"
                )) if original_identified.any() else np.nan,
                "mean_probability_total_variation": float(np.nanmean(tv)) if np.isfinite(tv).any() else np.nan,
            })
            for label in ["UNKNOWN", *reference.cluster_labels_]:
                mask = baseline_labels == label
                if not mask.any():
                    continue
                class_rows.append({
                    "oos_model_trained_at": version_text,
                    "replicate": replicate,
                    "baseline_class": label,
                    "rows": int(mask.sum()),
                    "same_label_rate": float(np.mean(baseline_labels[mask] == perturbed_labels[mask])),
                    "remains_identified_rate": float(np.mean(perturbed_labels[mask] != "UNKNOWN")),
                })
            tv = probability_total_variation(baseline, prediction, probability_columns)
            for dimension in ("year", "market_proxy_regime"):
                for slice_value, indices in score_context.groupby(dimension, dropna=False, sort=True).groups.items():
                    positions = np.asarray(list(indices), dtype=int)
                    selected_base = baseline_labels[positions]
                    selected_perturbed = perturbed_labels[positions]
                    known = selected_base != "UNKNOWN"
                    slice_rows.append({
                        "oos_model_trained_at": version_text,
                        "replicate": replicate,
                        "slice_dimension": dimension,
                        "slice_value": str(slice_value),
                        "rows": len(positions),
                        "baseline_identified_rows": int(known.sum()),
                        "baseline_coverage": float(known.mean()) if len(positions) else np.nan,
                        "label_agreement": float(np.mean(selected_base == selected_perturbed)) if len(positions) else np.nan,
                        "identified_label_agreement": float(np.mean(selected_base[known] == selected_perturbed[known])) if known.any() else np.nan,
                        "coverage_after_bootstrap": float(np.mean(selected_perturbed != "UNKNOWN")) if len(positions) else np.nan,
                        "baseline_identified_retention": float(np.mean(selected_perturbed[known] != "UNKNOWN")) if known.any() else np.nan,
                        "mean_probability_tv": float(np.nanmean(tv[positions])) if np.isfinite(tv[positions]).any() else np.nan,
                    })

    metric = pd.DataFrame(metric_rows)
    class_stability = pd.DataFrame(class_rows)
    folds = pd.DataFrame(fold_rows)
    slices = pd.DataFrame(slice_rows)
    metric.to_csv(output / "bootstrap_replicates.csv", index=False, encoding="utf-8-sig")
    class_stability.to_csv(output / "class_stability.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output / "fold_reproduction.csv", index=False, encoding="utf-8-sig")
    slices.to_csv(output / "stability_by_year_and_regime_replicates.csv", index=False, encoding="utf-8-sig")
    class_summary_rows = []
    for label, group in class_stability.groupby("baseline_class", sort=True):
        weights = group.rows.to_numpy(float)
        class_summary_rows.append({
            "baseline_class": label,
            "row_replicate_observations": int(group.rows.sum()),
            "same_label_rate_row_weighted": float(np.average(group.same_label_rate, weights=weights)),
            "remains_identified_rate_row_weighted": float(np.average(group.remains_identified_rate, weights=weights)),
        })
    class_summary = pd.DataFrame(class_summary_rows)
    class_summary.to_csv(output / "class_summary.csv", index=False, encoding="utf-8-sig")
    slice_summary_rows = []
    for (dimension, value), group in slices.groupby(["slice_dimension", "slice_value"], sort=True):
        row_count = float(group.rows.sum())
        identified_count = float(group.baseline_identified_rows.sum())
        slice_summary_rows.append({
            "slice_dimension": dimension,
            "slice_value": value,
            "model_versions": int(group.oos_model_trained_at.nunique()),
            "replicate_evaluations": len(group),
            "row_replicate_evaluations": int(row_count),
            "baseline_coverage_row_weighted": identified_count / row_count if row_count else np.nan,
            "label_agreement_row_weighted": float(np.average(group.label_agreement, weights=group.rows)),
            "identified_label_agreement_row_weighted": float(np.average(
                group.identified_label_agreement.fillna(0), weights=group.baseline_identified_rows,
            )) if identified_count else np.nan,
            "coverage_after_bootstrap_row_weighted": float(np.average(group.coverage_after_bootstrap, weights=group.rows)),
            "baseline_identified_retention_row_weighted": float(np.average(
                group.baseline_identified_retention.fillna(0), weights=group.baseline_identified_rows,
            )) if identified_count else np.nan,
            "mean_probability_tv_row_weighted": float(np.average(group.mean_probability_tv, weights=group.rows)),
        })
    slice_summary = pd.DataFrame(slice_summary_rows)
    slice_summary.loc[slice_summary.slice_dimension.eq("year")].to_csv(output / "stability_by_year.csv", index=False, encoding="utf-8-sig")
    slice_summary.loc[slice_summary.slice_dimension.eq("market_proxy_regime")].to_csv(output / "stability_by_market_regime.csv", index=False, encoding="utf-8-sig")
    total_rows = float(metric.rows.sum())
    identified_rows = float(metric.baseline_identified_rows.sum())
    summary = pd.DataFrame([{
        "folds": int(metric.oos_model_trained_at.nunique()),
        "replicates_per_fold": args.replicates,
        "unique_scored_rows_per_replicate": int(metric.groupby("replicate").rows.sum().iloc[0]),
        "row_bootstrap_evaluations": int(total_rows),
        "baseline_coverage_row_weighted": identified_rows / total_rows if total_rows else np.nan,
        "label_agreement_row_weighted": float(np.average(metric.label_agreement_all, weights=metric.rows)),
        "candidate_agreement_row_weighted": float(np.average(metric.candidate_agreement_all, weights=metric.rows)),
        "identified_label_agreement_row_weighted": float(np.average(
            metric.label_agreement_baseline_identified.fillna(0), weights=metric.baseline_identified_rows,
        )),
        "coverage_after_bootstrap_row_weighted": float(np.average(metric.coverage_after_bootstrap, weights=metric.rows)),
        "baseline_identified_retention_row_weighted": float(np.average(
            metric.baseline_identified_retention.fillna(0), weights=metric.baseline_identified_rows,
        )),
        "mean_probability_tv_row_weighted": float(np.average(metric.mean_probability_total_variation, weights=metric.rows)),
        "prototype_alignment_rms_median": float(metric.prototype_alignment_rms_train_scale.median()),
        "nonconverged_bootstrap_models": int((~metric.gmm_converged | ~metric.dpgmm_converged).sum()),
        "fold_label_agreement_p10_median_p90": [float(x) for x in metric.groupby("oos_model_trained_at").label_agreement_all.mean().quantile([.1, .5, .9])],
        "fold_identified_label_agreement_p10_median_p90": [float(x) for x in metric.groupby("oos_model_trained_at").label_agreement_baseline_identified.mean().quantile([.1, .5, .9])],
        "fold_coverage_p10_median_p90": [float(x) for x in metric.groupby("oos_model_trained_at").coverage_after_bootstrap.mean().quantile([.1, .5, .9])],
    }])
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(["all labels", "identified labels"], [
        summary.label_agreement_row_weighted.iloc[0],
        summary.identified_label_agreement_row_weighted.iloc[0],
    ], color=["#286a8e", "#238b8e"])
    axes[0].set(ylim=(0, 1), ylabel="Agreement with fold reference", title="Hard-label stability")
    axes[1].bar(["baseline", "resampled"], [
        summary.baseline_coverage_row_weighted.iloc[0],
        summary.coverage_after_bootstrap_row_weighted.iloc[0],
    ], color=["#7766aa", "#d17b2f"])
    axes[1].set(ylim=(0, 1), ylabel="Identified fraction", title="Coverage under symbol bootstrap")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Symbol-cluster refit robustness — not semantic accuracy")
    fig.savefig(output / "cluster_bootstrap_robustness.png", dpi=160)
    plt.close(fig)

    audit = {
        "features_path": str(Path(args.features).resolve()),
        "states_path": str(Path(args.states).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "model_versions_reconstructed": len(folds),
        "replicates_per_fold": args.replicates,
        "resampling_unit": "symbol cluster; draw unique symbols with replacement and retain their full training histories",
        "cluster_alignment": "Hungarian assignment of raw-space GMM reference means, distance scaled by frozen fold RobustScaler",
        "market_proxy": "equal-weight cross-sectional median daily return of supplied sample bars; trailing window inclusive of state available_at",
        "market_proxy_window": args.regime_window,
        "market_proxy_trend_threshold": args.regime_threshold,
        "boundaries_changed": False,
        "forward_return_fields_used": False,
        "semantic_accuracy_estimated": False,
        "probability_calibration_estimated": False,
        "archived_output_reproduced_exactly": bool(
            folds.saved_label_reproduction.eq(1).all()
            and folds.max_saved_probability_abs_delta.fillna(0).le(1e-8).all()
        ),
        "limitations": [
            "Cluster labels are aligned only within each frozen model version; no cross-version semantic identity is claimed.",
            "Symbol bootstrap preserves within-symbol histories but does not resample market years/regimes.",
            "Prototype matching can be ambiguous when clusters overlap; alignment RMS and per-class support are retained.",
            "This is model/refit stability, not semantic accuracy or calibrated confidence.",
        ],
        "outputs": ["bootstrap_replicates.csv", "class_stability.csv", "class_summary.csv", "fold_reproduction.csv",
                    "summary.csv", "stability_by_year.csv", "stability_by_market_regime.csv",
                    "stability_by_year_and_regime_replicates.csv", "sample_market_regime_proxy.csv",
                    "cluster_bootstrap_robustness.png"],
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Archived OOS states reproduced exactly: {audit['archived_output_reproduced_exactly']}")


if __name__ == "__main__":
    main()
