"""Label-free structural audit of an existing causal classification output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wave_segments.classifier_diagnostics import classification_structure_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, help="causal OOS states parquet")
    parser.add_argument("--output", required=True, help="new/empty output directory")
    parser.add_argument("--min-duration-segments", type=int, default=2)
    parser.add_argument("--max-duration-segments", type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    states = pd.read_parquet(args.states)
    summary, transitions, hsmm = classification_structure_summary(
        states,
        min_duration_segments=args.min_duration_segments,
        max_duration_segments=args.max_duration_segments,
    )
    summary.to_csv(output / "model_version_structure.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(output / "transitions_by_model_version.csv", index=False, encoding="utf-8-sig")
    hsmm.to_csv(output / "hsmm_duration_sensitivity.csv", index=False, encoding="utf-8-sig")

    features = [c for c in (
        "cumulative_return", "amplitude", "max_drawdown", "duration_bars",
        "return_volatility", "atr_pct_mean", "volume_cv", "price_volume_corr",
        "upper_shadow_ratio_mean", "lower_shadow_ratio_mean", "boundary_uncertainty",
    ) if c in states]
    profile = states.groupby(["oos_model_trained_at", "label"], dropna=False)[features].agg(
        ["count", "median", "mean", "std"]
    ) if features else pd.DataFrame()
    if not profile.empty:
        profile.columns = [f"{feature}_{stat}" for feature, stat in profile.columns]
        profile.reset_index().to_csv(output / "descriptive_profiles_by_version.csv", index=False, encoding="utf-8-sig")

    view = summary.copy()
    view["version_order"] = range(len(view))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(view.version_order, view.unknown_fraction, marker="o", ms=3, color="#7766aa")
    axes[0].set_ylabel("UNKNOWN fraction")
    axes[0].set_ylim(0, 1)
    axes[1].plot(view.version_order, view.adjacent_self_transition_rate, marker="o", ms=3, label="adjacent self-transition", color="#238b8e")
    axes[1].set_ylabel("Self-transition rate")
    axes[1].set_ylim(0, 1)
    ax2 = axes[1].twinx()
    ax2.plot(view.version_order, view.median_segment_bars_identified, marker="s", ms=3, label="median segment bars", color="#d17b2f")
    ax2.set_ylabel("Median identified segment length (bars)")
    axes[1].set_xlabel("OOS model version (chronological index; cluster IDs are version-local)")
    axes[0].grid(alpha=.2)
    axes[1].grid(alpha=.2)
    fig.suptitle("Classifier structure diagnostics — not semantic accuracy")
    fig.savefig(output / "classifier_structure_over_time.png", dpi=160)
    plt.close(fig)

    report = {
        "input": str(Path(args.states).resolve()),
        "rows": len(states),
        "model_versions": len(summary),
        "forward_return_fields_used": False,
        "semantic_accuracy_estimated": False,
        "probability_calibration_estimated": False,
        "notes": [
            "Temporary cluster IDs are only compared within each fitted model version.",
            "UNKNOWN is kept as a sequence break and is never counted as a transition.",
            "HSMM output is retrospective sensitivity analysis, not causal state and not accuracy evidence.",
            "Without adjudicated human labels, accuracy, Brier, log loss, ECE, and coverage-risk error cannot be measured.",
        ],
        "outputs": [
            "model_version_structure.csv", "transitions_by_model_version.csv",
            "hsmm_duration_sensitivity.csv", "descriptive_profiles_by_version.csv",
            "classifier_structure_over_time.png",
        ],
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
