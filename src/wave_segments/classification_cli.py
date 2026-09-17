"""Command-line walk-forward validation for human-labelled wave classes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .validation import (CalibratedSegmentClassifier, PurgedWalkForwardSplit,
                         annotation_agreement_report)
from .visualization import plot_coverage_risk, plot_reliability_diagram


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _jsonable(value):
    if isinstance(value, pd.DataFrame): return value.to_dict(orient="records")
    if isinstance(value, pd.Series): return value.to_dict()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Purged walk-forward validation of human wave labels")
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="outputs/classification_validation.json")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--target-error", type=float, default=.10)
    parser.add_argument("--min-coverage", type=float, default=.65)
    parser.add_argument("--inclusion-probability-col", default="sampling_probability")
    args = parser.parse_args()

    features, annotations = _load(args.features), _load(args.annotations)
    annotation_label = "label" if "label" in annotations else "true_label"
    agreement_report = annotation_agreement_report(annotations, label_col=annotation_label)
    consensus, agreement = agreement_report["consensus"], agreement_report["label_agreement"]
    consensus_columns = ["segment_id", "consensus_label"]
    probability_col = args.inclusion_probability_col
    if probability_col in consensus:
        consensus_columns.append(probability_col)
    elif probability_col in annotations:
        probabilities = annotations.groupby("segment_id", as_index=False)[probability_col].first()
        consensus = consensus.merge(probabilities, on="segment_id", how="left")
        consensus_columns.append(probability_col)
    else:
        probability_col = None
    data = features.merge(consensus[consensus_columns], on="segment_id", how="inner").reset_index(drop=True)
    folds, failures = [], []
    for number, fold in enumerate(PurgedWalkForwardSplit(args.splits).split(data), 1):
        train, calibration, test = data.loc[fold.train], data.loc[fold.calibration], data.loc[fold.test]
        try:
            model = CalibratedSegmentClassifier(method=args.method).fit(train, calibration)
            threshold = model.choose_unknown_threshold(
                calibration, target_error=args.target_error, min_coverage=args.min_coverage,
                inclusion_probability_col=probability_col,
            )
            model.fit_aps(calibration)
            metrics = model.evaluate(test, inclusion_probability_col=probability_col)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            for suffix, axis in (
                ("reliability", plot_reliability_diagram(metrics["reliability"])),
                ("coverage_risk", plot_coverage_risk(metrics["coverage_risk"])),
            ):
                figure = axis.figure; figure.tight_layout()
                figure.savefig(output.with_name(f"{output.stem}_fold{number}_{suffix}.png"), dpi=150)
                import matplotlib.pyplot as plt
                plt.close(figure)
            metrics.update({"fold": number, "unknown_threshold": threshold,
                            "mean_prediction_set_size": float(np.mean([len(x) for x in model.predict_sets(test)]))})
            folds.append(_jsonable(metrics))
        except ValueError as exc:
            failures.append({"fold": number, "error": str(exc)})
    report = {"agreement": _jsonable(agreement),
              "boundary_agreement": _jsonable(agreement_report["boundary_agreement"]),
              "folds": folds, "failures": failures,
              "warning": "Only test-fold metrics are evidence; thresholds are chosen on calibration folds."}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"folds={len(folds)} failures={len(failures)} output={output}")


if __name__ == "__main__":
    main()
