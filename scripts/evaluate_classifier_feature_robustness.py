"""Test frozen OOS cluster labels under causal-fold-scaled feature perturbations."""
from __future__ import annotations

import argparse
import json
from hashlib import blake2b
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wave_segments.class_robustness import perturb_features_from_train_scale, probability_total_variation
from wave_segments.discovery import DiscoveryConfig, MultiModelDiscoverer


def _seed(base: int, model_time: str, noise: float, replicate: int) -> int:
    value = f"{base}|{model_time}|{noise:.8g}|{replicate}".encode("utf-8")
    return int.from_bytes(blake2b(value, digest_size=4).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="causal_segment_features.parquet used by discovery")
    parser.add_argument("--states", required=True, help="frozen causal_oos_discovery_states.parquet")
    parser.add_argument("--manifest", required=True, help="discovery manifest with fold/model settings")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260917)
    args = parser.parse_args()
    if args.replicates < 1 or any(x < 0 for x in args.noise_fractions):
        parser.error("replicates must be positive and noise fractions non-negative")
    if len(set(args.noise_fractions)) != len(args.noise_fractions):
        parser.error("noise fractions must be unique")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    features = pd.read_parquet(args.features)
    states = pd.read_parquet(args.states)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    settings = manifest["config"]
    feature_columns = list(settings["features"])
    probability_columns = [f"prob_{name}" for name in settings.get("cluster_labels", [])]
    if not probability_columns or not set(probability_columns).issubset(states.columns):
        probability_columns = sorted(c for c in states if c.startswith("prob_CLUSTER_"))
    if not probability_columns:
        raise ValueError("No saved cluster probability columns in states")

    features["available_at"] = pd.to_datetime(features["available_at"], errors="coerce")
    states["oos_model_trained_at"] = pd.to_datetime(states["oos_model_trained_at"], errors="coerce")
    states["state_available_at"] = pd.to_datetime(states["state_available_at"], errors="coerce")
    feature_by_id = features.set_index("segment_id", drop=False)
    mixture_config = DiscoveryConfig(
        n_components=int(settings["n_components"]),
        random_state=int(settings["random_state"]),
        ensemble_size=int(settings["ensemble_size"]),
        min_cluster_samples=int(settings["min_cluster_samples"]),
        min_cluster_symbols=int(settings["min_cluster_symbols"]),
        min_cluster_years=int(settings["min_cluster_years"]),
        max_iter=int(settings["max_iter"]),
        include_hdbscan=False,
    )

    replicate_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    scored_states = states.loc[states["oos_model_trained_at"].notna()].copy()
    scored_states = scored_states.loc[scored_states[probability_columns].notna().any(axis=1)]
    versions = sorted(scored_states["oos_model_trained_at"].dropna().unique())
    for model_time_value in versions:
        model_time = pd.Timestamp(model_time_value)
        model_text = model_time.isoformat()
        archived_fold = scored_states.loc[scored_states["oos_model_trained_at"].eq(model_time)].copy()
        expected_train_rows = int(archived_fold["oos_train_rows"].dropna().iloc[0])
        train = features.loc[features["available_at"] < model_time].copy()
        train = train.sort_values(["available_at", "symbol", "segment_id"], kind="stable")
        if len(train) != expected_train_rows:
            raise RuntimeError(
                f"fold {model_text} training-row mismatch: reconstructed={len(train)}, saved={expected_train_rows}"
            )
        score_ids = archived_fold["segment_id"].astype(str).tolist()
        score = feature_by_id.loc[score_ids].reset_index(drop=True)
        model = MultiModelDiscoverer(mixture_config, feature_columns).fit(train)
        baseline = model.predict(score)
        archived_labels = archived_fold["label"].astype(str).to_numpy()
        baseline_labels = baseline["label"].astype(str).to_numpy()
        label_reproduction = float(np.mean(archived_labels == baseline_labels))
        archived_probs = archived_fold[probability_columns].to_numpy(float)
        recreated_probs = baseline[probability_columns].to_numpy(float)
        valid = np.isfinite(archived_probs).all(axis=1) & np.isfinite(recreated_probs).all(axis=1)
        max_probability_delta = float(np.max(np.abs(archived_probs[valid] - recreated_probs[valid]))) if valid.any() else float("nan")
        fold_rows.append({
            "oos_model_trained_at": model_text,
            "train_rows": len(train),
            "score_rows_with_probabilities": len(score),
            "archived_label_reproduction": label_reproduction,
            "max_saved_probability_abs_delta": max_probability_delta,
            "gmm_converged": model.gmm_converged_,
            "dpgmm_converged": model.dpgmm_converged_,
        })

        for noise in args.noise_fractions:
            for replicate in range(args.replicates):
                noisy = perturb_features_from_train_scale(
                    score, train, feature_columns, noise_fraction=noise,
                    random_state=_seed(args.random_state, model_text, noise, replicate),
                )
                prediction = model.predict(noisy)
                baseline_label = baseline["label"].astype(str).to_numpy()
                perturbed_label = prediction["label"].astype(str).to_numpy()
                identified = baseline_label != "UNKNOWN"
                tv = probability_total_variation(baseline, prediction, probability_columns)
                key = {"oos_model_trained_at": model_text, "noise_fraction": noise, "replicate": replicate}
                replicate_rows.append({
                    **key,
                    "rows": len(score),
                    "baseline_identified_rows": int(identified.sum()),
                    "baseline_coverage": float(identified.mean()),
                    "label_agreement_all": float(np.mean(baseline_label == perturbed_label)),
                    "candidate_agreement_all": float(np.mean(baseline.candidate_label.astype(str).to_numpy() == prediction.candidate_label.astype(str).to_numpy())),
                    "label_agreement_baseline_identified": float(np.mean(baseline_label[identified] == perturbed_label[identified])) if identified.any() else np.nan,
                    "coverage_after_perturbation": float(np.mean(perturbed_label != "UNKNOWN")),
                    "coverage_retention_baseline_identified": float(np.mean(perturbed_label[identified] != "UNKNOWN")) if identified.any() else np.nan,
                    "mean_probability_total_variation": float(np.nanmean(tv)) if np.isfinite(tv).any() else np.nan,
                    "mean_max_probability_delta": float(np.nanmean(np.abs(baseline.max_probability.to_numpy(float) - prediction.max_probability.to_numpy(float)))),
                })
                for original_class in sorted(set(baseline_label)):
                    mask = baseline_label == original_class
                    class_rows.append({
                        **key,
                        "baseline_class": original_class,
                        "rows": int(mask.sum()),
                        "same_label_rate": float(np.mean(baseline_label[mask] == perturbed_label[mask])),
                        "remains_identified_rate": float(np.mean(perturbed_label[mask] != "UNKNOWN")),
                    })

    replicate = pd.DataFrame(replicate_rows)
    classes = pd.DataFrame(class_rows)
    folds = pd.DataFrame(fold_rows)
    replicate.to_csv(output / "replicate_stability.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output / "class_stability.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output / "fold_reproduction.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for noise, group in replicate.groupby("noise_fraction", sort=True):
        total = float(group["rows"].sum())
        base_identified = float(group["baseline_identified_rows"].sum())
        summary_rows.append({
            "noise_fraction": noise,
            "folds": int(group["oos_model_trained_at"].nunique()),
            "replicates": int(group["replicate"].nunique()),
            "unique_scored_rows_per_replicate": int(group.groupby("replicate")["rows"].sum().iloc[0]),
            "row_perturbation_evaluations": int(total),
            "baseline_coverage_row_weighted": base_identified / total if total else np.nan,
            "label_agreement_row_weighted": float(np.average(group.label_agreement_all, weights=group.rows)),
            "identified_label_agreement_row_weighted": float(np.average(
                group.label_agreement_baseline_identified.fillna(0),
                weights=group.baseline_identified_rows,
            )) if base_identified else np.nan,
            "coverage_row_weighted": float(np.average(group.coverage_after_perturbation, weights=group.rows)),
            "baseline_identified_retention_row_weighted": float(np.average(
                group.coverage_retention_baseline_identified.fillna(0),
                weights=group.baseline_identified_rows,
            )) if base_identified else np.nan,
            "mean_probability_tv_row_weighted": float(np.average(group.mean_probability_total_variation, weights=group.rows)),
            "label_agreement_fold_median": float(group.label_agreement_all.median()),
            "identified_label_agreement_fold_median": float(group.label_agreement_baseline_identified.median()),
            "coverage_fold_median": float(group.coverage_after_perturbation.median()),
            "mean_probability_tv_fold_median": float(group.mean_probability_total_variation.median()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary_by_noise.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(summary.noise_fraction, summary.label_agreement_row_weighted, marker="o", label="all labels")
    axes[0].plot(summary.noise_fraction, summary.identified_label_agreement_row_weighted, marker="s", label="baseline identified only")
    axes[0].set(xlabel="Noise fraction × train-fold robust scale", ylabel="Median label agreement", ylim=(0, 1), title="Discrete state stability")
    axes[0].legend()
    axes[1].plot(summary.noise_fraction, summary.coverage_row_weighted, marker="o", label="coverage after perturbation")
    axes[1].plot(summary.noise_fraction, summary.mean_probability_tv_row_weighted, marker="s", label="probability TV distance")
    axes[1].set(xlabel="Noise fraction × train-fold robust scale", ylabel="Median score", ylim=(0, 1), title="Abstention and soft-label drift")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Frozen classifier feature robustness — not semantic accuracy")
    fig.savefig(output / "feature_robustness.png", dpi=160)
    plt.close(fig)

    audit = {
        "features_path": str(Path(args.features).resolve()),
        "states_path": str(Path(args.states).resolve()),
        "model_manifest": str(Path(args.manifest).resolve()),
        "model_versions_reconstructed": len(folds),
        "scored_rows_with_saved_probabilities": int(len(scored_states)),
        "replicates_per_noise_level": args.replicates,
        "noise_fractions": list(args.noise_fractions),
        "noise_scale_source": "robust MAD scale fit on each frozen training fold only",
        "fixed_features": ["duration_bars", "boundary_uncertainty"],
        "model_refit_under_perturbation": False,
        "boundaries_changed": False,
        "future_return_fields_used": False,
        "semantic_accuracy_estimated": False,
        "probability_calibration_estimated": False,
        "interpretation": "Measures local feature-noise sensitivity of each frozen fold model; it does not test semantic correctness, boundary sensitivity, or training-sample/model-refit stability.",
        "archived_output_reproduced_exactly": bool(
            folds.archived_label_reproduction.eq(1).all()
            and folds.max_saved_probability_abs_delta.fillna(0).le(1e-8).all()
        ),
        "outputs": ["replicate_stability.csv", "class_stability.csv", "fold_reproduction.csv",
                    "summary_by_noise.csv", "feature_robustness.png"],
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Archived OOS states reproduced exactly: {audit['archived_output_reproduced_exactly']}")


if __name__ == "__main__":
    main()
