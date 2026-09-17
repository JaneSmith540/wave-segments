"""Full-market transductive cluster-resampling stress test; never causal/OOS."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wave_segments.class_robustness import probability_total_variation
from wave_segments.cluster_bootstrap import (
    align_bootstrap_to_reference,
    align_by_probability_overlap,
    remap_bootstrap_predictions,
    resample_symbols_with_replacement,
)
from wave_segments.discovery import MultiModelDiscoverer
from wave_segments.fullmarket_discovery_cli import _feature_parts, _fit_sample


def _row_metrics(base: pd.DataFrame, altered: pd.DataFrame,
                 probability_columns: list[str], *, group: str, value: str,
                 replicate: int, bucket: str, start_year: pd.Series | None = None) -> dict[str, object]:
    y0, y1 = base.label.astype(str).to_numpy(), altered.label.astype(str).to_numpy()
    known = y0 != "UNKNOWN"
    tv = probability_total_variation(base, altered, probability_columns)
    return {
        "replicate": replicate,
        "bucket": bucket,
        "slice_dimension": group,
        "slice_value": value,
        "rows": len(base),
        "baseline_identified_rows": int(known.sum()),
        "label_agreement": float(np.mean(y0 == y1)) if len(y0) else np.nan,
        "candidate_agreement": float(np.mean(
            base.candidate_label.astype(str).to_numpy() == altered.candidate_label.astype(str).to_numpy()
        )) if len(y0) else np.nan,
        "identified_label_agreement": float(np.mean(y0[known] == y1[known])) if known.any() else np.nan,
        "baseline_coverage": float(known.mean()) if len(y0) else np.nan,
        "coverage_after": float(np.mean(y1 != "UNKNOWN")) if len(y0) else np.nan,
        "baseline_identified_retention": float(np.mean(y1[known] != "UNKNOWN")) if known.any() else np.nan,
        "mean_probability_tv": float(np.nanmean(tv)) if np.isfinite(tv).any() else np.nan,
        "baseline_class_counts": {str(k): int(v) for k, v in pd.Series(y0).value_counts().items()},
    }


def _weighted_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, values))
        total = float(group.rows.sum())
        known = float(group.baseline_identified_rows.sum())
        row.update({
            "buckets": int(group.bucket.nunique()),
            "replicates": int(group.replicate.nunique()),
            "row_replicate_evaluations": int(total),
            "baseline_coverage_row_weighted": known / total if total else np.nan,
            "label_agreement_row_weighted": float(np.average(group.label_agreement, weights=group.rows)),
            "candidate_agreement_row_weighted": float(np.average(group.candidate_agreement, weights=group.rows)),
            "identified_label_agreement_row_weighted": float(np.average(
                group.identified_label_agreement.fillna(0), weights=group.baseline_identified_rows,
            )) if known else np.nan,
            "coverage_after_row_weighted": float(np.average(group.coverage_after, weights=group.rows)),
            "baseline_identified_retention_row_weighted": float(np.average(
                group.baseline_identified_retention.fillna(0), weights=group.baseline_identified_rows,
            )) if known else np.nan,
            "mean_probability_tv_row_weighted": float(np.average(group.mean_probability_tv, weights=group.rows)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--discovery-output", required=True, help="completed offline full-market discovery directory")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260917)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    source, discovery, output = Path(args.features), Path(args.discovery_output), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((discovery / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((discovery / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or summary.get("mode") != "offline_transductive_review_only":
        raise ValueError("input must be a complete, explicitly offline/transductive discovery run")
    settings = manifest["settings"]
    parts = _feature_parts(source)
    training = _fit_sample(parts, int(settings["fit_sample_size"]), int(settings["random_state"]))
    training_hash = __import__("hashlib").sha256(
        "\n".join(training.segment_id.astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    if training_hash != summary.get("training_ids_sha256"):
        raise RuntimeError("reconstructed training IDs do not match saved discovery manifest")

    reference: MultiModelDiscoverer = joblib.load(discovery / summary["model_file"])
    if reference.feature_columns != summary["model_metadata"]["feature_columns"]:
        raise RuntimeError("saved discoverer feature schema differs from manifest")
    probability_columns = [f"prob_{label}" for label in reference.cluster_labels_]
    baseline_rows = []
    max_probability_delta = 0.0
    exact_labels = True
    full_labels: defaultdict[str, int] = defaultdict(int)

    # First prove the serialized discoverer reproduces its saved full-market output.
    for part in parts:
        bucket = part.parent.name
        features = pd.read_parquet(part)
        saved_path = discovery / "discovery_labels" / bucket / "data.parquet"
        saved = pd.read_parquet(saved_path).set_index("segment_id", drop=False).loc[features.segment_id.astype(str)].reset_index(drop=True)
        predicted = reference.predict(features)
        same = predicted.label.astype(str).to_numpy() == saved.label.astype(str).to_numpy()
        exact_labels &= bool(same.all())
        left, right = predicted[probability_columns].to_numpy(float), saved[probability_columns].to_numpy(float)
        valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        delta = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0
        max_probability_delta = max(max_probability_delta, delta)
        full_labels.update(predicted.label.astype(str).value_counts().to_dict())
        baseline_rows.append({"bucket": bucket, "rows": len(features),
                              "label_exact_match": bool(same.all()),
                              "max_probability_abs_delta": delta})
    if not exact_labels or max_probability_delta > 1e-8:
        raise RuntimeError("saved full-market labels were not exactly reproducible; aborting bootstrap comparison")

    metrics: list[dict[str, object]] = []
    class_metrics: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    config = reference.config
    columns = list(reference.feature_columns)
    training_symbols = training.symbol.dropna().drop_duplicates().to_numpy()
    reference_anchor = reference.predict(training)
    reference_anchor_prob = reference_anchor[probability_columns].to_numpy(float)

    for replicate in range(args.replicates):
        sample, unique_drawn = resample_symbols_with_replacement(
            training, random_state=args.random_state + replicate * 7919,
        )
        bootstrap = MultiModelDiscoverer(config, columns).fit(sample)
        anchor_prediction = bootstrap.predict(training)
        mapping, overlap_similarity = align_by_probability_overlap(
            reference_anchor_prob,
            anchor_prediction[probability_columns].to_numpy(float),
            reference.cluster_labels_, bootstrap.cluster_labels_,
        )
        _, prototype_rms = align_bootstrap_to_reference(reference, bootstrap)
        prototype_rows.append({
            "replicate": replicate,
            "bootstrap_rows": len(sample),
            "training_symbols": len(training_symbols),
            "distinct_symbols_drawn": unique_drawn,
            "training_overlap_alignment_mean_cosine": overlap_similarity,
            "prototype_alignment_rms_train_scale_diagnostic_only": prototype_rms,
            "gmm_converged": bootstrap.gmm_converged_,
            "dpgmm_converged": bootstrap.dpgmm_converged_,
            "valid_cluster_count": int(sum(bootstrap.cluster_valid_)),
        })
        for part in parts:
            bucket = part.parent.name
            features = pd.read_parquet(part)
            saved_path = discovery / "discovery_labels" / bucket / "data.parquet"
            baseline = pd.read_parquet(saved_path).set_index("segment_id", drop=False).loc[
                features.segment_id.astype(str)
            ].reset_index(drop=True)
            predicted = remap_bootstrap_predictions(bootstrap.predict(features), mapping, reference.cluster_labels_)
            year = pd.to_datetime(features.start, errors="coerce").dt.year
            metrics.append(_row_metrics(baseline, predicted, probability_columns,
                                        group="all", value="FULLMARKET", replicate=replicate, bucket=bucket))
            for year_value, indices in year.groupby(year, dropna=False, sort=True).groups.items():
                idx = np.asarray(list(indices), dtype=int)
                metrics.append(_row_metrics(
                    baseline.iloc[idx], predicted.iloc[idx], probability_columns,
                    group="start_year", value=str(year_value), replicate=replicate, bucket=bucket,
                ))
            for label, group in baseline.groupby("label", sort=True):
                idx = group.index.to_numpy(int)
                changed = predicted.iloc[idx]
                original = group.label.astype(str).to_numpy()
                altered = changed.label.astype(str).to_numpy()
                class_metrics.append({
                    "replicate": replicate, "bucket": bucket, "baseline_class": str(label),
                    "rows": len(group), "same_label_rate": float(np.mean(original == altered)),
                    "remains_identified_rate": float(np.mean(altered != "UNKNOWN")),
                })

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "replicate_by_bucket_and_year.csv", index=False, encoding="utf-8-sig")
    class_frame = pd.DataFrame(class_metrics)
    class_frame.to_csv(output / "class_stability_by_bucket.csv", index=False, encoding="utf-8-sig")
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(output / "baseline_reproduction_by_bucket.csv", index=False, encoding="utf-8-sig")
    prototype_frame = pd.DataFrame(prototype_rows)
    prototype_frame.to_csv(output / "bootstrap_models.csv", index=False, encoding="utf-8-sig")

    overall = _weighted_summary(metric_frame.loc[metric_frame.slice_dimension.eq("all")], ["slice_dimension", "slice_value"])
    yearly = _weighted_summary(metric_frame.loc[metric_frame.slice_dimension.eq("start_year")], ["slice_dimension", "slice_value"])
    overall.to_csv(output / "overall_stability.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "stability_by_start_year.csv", index=False, encoding="utf-8-sig")
    class_summary = []
    for label, group in class_frame.groupby("baseline_class", sort=True):
        class_summary.append({
            "baseline_class": label,
            "row_replicate_evaluations": int(group.rows.sum()),
            "same_label_rate_row_weighted": float(np.average(group.same_label_rate, weights=group.rows)),
            "remains_identified_rate_row_weighted": float(np.average(group.remains_identified_rate, weights=group.rows)),
        })
    pd.DataFrame(class_summary).to_csv(output / "class_summary.csv", index=False, encoding="utf-8-sig")

    year_plot = yearly.copy()
    year_plot["year_num"] = pd.to_numeric(year_plot.slice_value, errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(["all labels", "known labels"], [
        overall.label_agreement_row_weighted.iloc[0],
        overall.identified_label_agreement_row_weighted.iloc[0],
    ], color=["#286a8e", "#238b8e"])
    axes[0].set(title="Full-market in-sample agreement", ylabel="Agreement", ylim=(0, 1))
    axes[1].plot(year_plot.year_num, year_plot.identified_label_agreement_row_weighted,
                 marker="o", color="#b45f06")
    axes[1].set(title="By candidate start year", xlabel="Year", ylabel="Known-label agreement", ylim=(0, 1))
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Full-market offline discovery symbol bootstrap — transductive, not OOS accuracy")
    fig.savefig(output / "fullmarket_resample_stability.png", dpi=160)
    plt.close(fig)

    audit = {
        "mode": "offline_transductive_review_only",
        "features_path": str(source.resolve()),
        "discovery_path": str(discovery.resolve()),
        "input_rows": int(summary["predicted_rows"]),
        "training_rows": len(training),
        "training_symbols": len(training_symbols),
        "training_id_sha256_reconstructed": training_hash,
        "training_id_hash_matches": True,
        "baseline_labels_reproduced_exactly": exact_labels,
        "baseline_max_probability_abs_delta": max_probability_delta,
        "replicates": args.replicates,
        "resampling_unit": "symbol clusters drawn with replacement, retaining each sampled symbol's full fit-sample history",
        "cluster_alignment": "Primary: Hungarian assignment maximizing cosine-normalized soft-membership overlap on the reconstructed common training sample; prototype distance is diagnostic only.",
        "cluster_ids_semantic": False,
        "training_and_scoring_temporally_disjoint": False,
        "causal_oos": False,
        "forward_return_fields_used": False,
        "semantic_accuracy_estimated": False,
        "limitations": [
            "The fitted model was trained on a sample spanning the full 2015-2024 history, then scored on the same full-market candidate set.",
            "This is broad-sample structural robustness only; it cannot be reported as causal OOS or semantic accuracy.",
            "Symbol resampling changes cross-sectional composition but does not resample year/regime blocks.",
            "Prototype-distance assignment was ill-conditioned by extreme feature centers; overlap alignment is used for reported labels.",
            "HDBSCAN was unavailable in the saved model and remains untested here.",
        ],
        "outputs": ["replicate_by_bucket_and_year.csv", "class_stability_by_bucket.csv", "baseline_reproduction_by_bucket.csv",
                    "bootstrap_models.csv", "overall_stability.csv", "stability_by_start_year.csv",
                    "class_summary.csv", "fullmarket_resample_stability.png"],
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print("By start year:\n", yearly.to_string(index=False))
    print("Baseline labels reproduced exactly:", exact_labels)


if __name__ == "__main__":
    main()
