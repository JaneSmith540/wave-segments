"""Render sampled causal wave candidates for blinded visual boundary review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wave_segments.data import load_local
from wave_segments.visualization import _plt, plot_segmented_candles


COLORS = {
    "CLUSTER_A": "#2e86de", "CLUSTER_B": "#e67e22", "CLUSTER_C": "#8e44ad",
    "CLUSTER_D": "#16a085", "CLUSTER_E": "#c0392b", "UNKNOWN": "#7f8c8d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--context-calendar-days", type=int, default=45)
    args = parser.parse_args()
    if args.per_page < 1:
        parser.error("per-page must be positive")
    bars = load_local(args.bars)
    queue = pd.read_csv(args.queue, dtype={"segment_id": str})
    if queue.empty:
        parser.error("review queue is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    page_count = (len(queue) + args.per_page - 1) // args.per_page
    files = []
    audit_rows = []
    by_symbol = {str(k): g.sort_values("timestamp").reset_index(drop=True)
                 for k, g in bars.groupby("symbol", sort=False)}
    for page in range(page_count):
        subset = queue.iloc[page * args.per_page:(page + 1) * args.per_page]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
        for ax, (_, row) in zip(axes.flat, subset.iterrows()):
            symbol = str(row["symbol"])
            symbol_bars = by_symbol.get(symbol)
            if symbol_bars is None or symbol_bars.empty:
                ax.set_title(f"{symbol} bars missing")
                ax.axis("off")
                audit_rows.append({"segment_id": row.segment_id, "chart_status": "bars_missing"})
                continue
            start = pd.Timestamp(row["start_timestamp"])
            end = pd.Timestamp(row["end_timestamp"])
            view = symbol_bars.loc[
                symbol_bars.timestamp.between(start - pd.Timedelta(days=args.context_calendar_days), end)
            ].copy()
            # Deliberately omit positional indices: these bars are a cropped
            # view, so plotting positions must be resolved from timestamps.
            segment = pd.DataFrame([{
                "symbol": symbol, "start": start, "end": end,
                "label": str(row.get("predicted_label", "UNKNOWN")),
            }])
            plot_segmented_candles(
                view, segment, color_map=COLORS,
                title=f"{symbol} | {start:%Y-%m-%d} → {end:%Y-%m-%d} | {row.get('predicted_label', 'UNKNOWN')}",
                ax=ax,
            )
            ax.text(.01, .98, f"status={row.get('candidate_status', '')}\nreason={str(row.get('unknown_reason', ''))[:80]}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none"})
            audit_rows.append({"segment_id": row.segment_id, "chart_status": "rendered",
                               "symbol": symbol, "start": str(start), "end": str(end),
                               "candidate_label": str(row.get("predicted_label", "UNKNOWN")),
                               "candidate_status": str(row.get("candidate_status", "")),
                               "unknown_reason": str(row.get("unknown_reason", "")),
                               "sampling_stratum": row.get("sampling_stratum"),
                               "sampling_probability": row.get("sampling_probability")})
        for ax in axes.flat[len(subset):]:
            ax.axis("off")
        fig.suptitle("Causal segment visual review — context ends at segment endpoint", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .97))
        path = args.output / f"review_{page + 1:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output / "review_sample_manifest.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete", "sample_rows": len(queue), "rendered_rows": int(audit.chart_status.eq("rendered").sum()),
        "chart_pages": files, "future_bars_included": False,
        "visual_review_is_formal_truth": False,
        "review_template": "review_sample_manifest.csv; add independent annotator events separately",
        "bars_path": str(args.bars.resolve()), "queue_path": str(args.queue.resolve()),
        "context_calendar_days_before_start": args.context_calendar_days,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
