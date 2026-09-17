"""Scan ATR reversal thresholds on a fixed OHLCV sample; agreement is not truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wave_segments.config import SegmentationConfig
from wave_segments.schema import normalize_ohlcv
from wave_segments.segmentation import segment_ohlcv_causal
from wave_segments.segmentation_sensitivity import compare_causal_boundaries, segment_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, help="OHLCV parquet; one row per symbol/date")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--atr-reversals", type=float, nargs="+", default=[1.8, 2.2, 2.6])
    parser.add_argument("--reference", type=float, default=2.2)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--min-bars", type=int, default=4)
    parser.add_argument("--short-threshold", type=int, default=10,
                        help="report fraction of segments shorter than this many bars")
    args = parser.parse_args()
    if args.reference not in args.atr_reversals:
        raise SystemExit("--reference must be included in --atr-reversals")
    if len(set(args.atr_reversals)) != len(args.atr_reversals):
        raise SystemExit("--atr-reversals must not contain duplicates")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bars = normalize_ohlcv(pd.read_parquet(args.bars))
    configs = {
        value: SegmentationConfig(atr_reversal=value, min_bars=args.min_bars)
        for value in args.atr_reversals
    }
    segments = {value: segment_ohlcv_causal(bars, config) for value, config in configs.items()}
    reference = segments[args.reference]
    summary_rows = []
    symbol_frames, year_frames = [], []
    for value in args.atr_reversals:
        current = segments[value]
        overall, per_symbol, per_year = compare_causal_boundaries(
            reference, current, tolerance=args.tolerance,
        )
        summary_rows.append({
            "atr_reversal": value,
            "is_reference": value == args.reference,
            **segment_distribution(current, short_threshold=args.short_threshold),
            **overall,
        })
        per_symbol.insert(0, "atr_reversal", value)
        per_year.insert(0, "atr_reversal", value)
        symbol_frames.append(per_symbol)
        year_frames.append(per_year)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat(symbol_frames, ignore_index=True).to_csv(out / "agreement_by_symbol.csv", index=False, encoding="utf-8-sig")
    pd.concat(year_frames, ignore_index=True).to_csv(out / "agreement_by_year_symbol.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        frame.assign(atr_reversal=value)
        for value, frame in segments.items()
    ], ignore_index=True).to_parquet(out / "segments_by_parameter.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(summary.atr_reversal, summary.segments, marker="o", color="#286a8e")
    axes[0].set(xlabel="ATR reversal multiplier", ylabel="Confirmed segments", title="Segmentation count")
    axes[1].plot(summary.atr_reversal, summary.f1_agreement, marker="o", label="± tolerance agreement F1", color="#b45f06")
    axes[1].plot(summary.atr_reversal, summary.median_bars, marker="s", label="Median segment length", color="#238b8e")
    axes[1].set(xlabel="ATR reversal multiplier", title=f"Relative stability vs {args.reference:g} reference")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Causal ATR segmentation sensitivity — agreement is not boundary accuracy")
    fig.savefig(out / "parameter_sensitivity.png", dpi=160)
    plt.close(fig)

    audit = {
        "input": str(Path(args.bars).resolve()),
        "input_rows": int(len(bars)),
        "symbols": int(bars.symbol.nunique()),
        "date_min": str(bars.timestamp.min()),
        "date_max": str(bars.timestamp.max()),
        "segmenter": "segment_ohlcv_causal / ATR ZigZag",
        "reference_atr_reversal": args.reference,
        "atr_reversals": list(args.atr_reversals),
        "tolerance_bars": args.tolerance,
        "future_return_fields_used": False,
        "human_boundary_truth_used": False,
        "semantic_accuracy_estimated": False,
        "interpretation": "Precision/recall/F1 here measure agreement with the selected reference parameterization, not correctness.",
        "outputs": ["parameter_sensitivity.csv", "agreement_by_symbol.csv",
                    "agreement_by_year_symbol.csv", "segments_by_parameter.parquet",
                    "parameter_sensitivity.png"],
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
