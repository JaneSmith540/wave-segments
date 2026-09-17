from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from wave_segments.config import PipelineConfig, SegmentationConfig
from wave_segments.data import fetch_tushare, fetch_tushare_index
from wave_segments.pipeline import WavePipeline
from wave_segments.segmentation import segment_ohlcv


SYMBOLS = [
    "000001.SZ", "000333.SZ", "000858.SZ", "300750.SZ",
    "600036.SH", "600276.SH", "600519.SH", "601318.SH",
]


def boundary_match_rate(reference: pd.DataFrame, candidate: pd.DataFrame, tolerance: int = 3) -> float:
    matches: list[bool] = []
    for symbol, group in reference.groupby("symbol"):
        left = group["end_idx"].iloc[:-1].to_numpy(int)
        right = candidate.loc[candidate.symbol == symbol, "end_idx"].iloc[:-1].to_numpy(int)
        matches.extend(bool(len(right) and np.min(np.abs(right - point)) <= tolerance) for point in left)
    return float(np.mean(matches)) if matches else float("nan")


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    bars_path, index_path = data_dir / "real_bars_2024.parquet", data_dir / "csi300_2024.parquet"
    if bars_path.exists() and index_path.exists():
        bars, market = pd.read_parquet(bars_path), pd.read_parquet(index_path)
    else:
        # A bounded calendar year makes this diagnostic quick and repeatable.
        # The production loader's qfq path is tested separately; raw bars avoid
        # Tushare's slower long-range adjustment endpoint during diagnostics.
        bars = fetch_tushare(SYMBOLS, "20240101", "20241231", adjustment=None)
        market = fetch_tushare_index("000300.SH", "20240101", "20241231")
        bars.to_parquet(bars_path, index=False)
        market.to_parquet(index_path, index=False)

    config = PipelineConfig(output_dir="outputs_real")
    pipeline = WavePipeline(config)
    result = pipeline.fit_run(bars, market_context=market)
    combined = result.features.merge(result.labels, on="segment_id")

    recognised = combined[~combined["is_unknown"]]
    silhouette = None
    if len(recognised) > recognised["label"].nunique() > 1:
        matrix = pipeline.model._matrix(recognised)
        silhouette = float(silhouette_score(matrix, recognised["label"]))

    base = result.segments
    sensitivity = []
    for atr in (1.8, 2.0, 2.4, 2.6):
        cfg = SegmentationConfig(**{**asdict(config.segmentation), "atr_reversal": atr})
        candidate = segment_ohlcv(bars, cfg, use_changepoints=True)
        sensitivity.append({"atr_reversal": atr, "segments": len(candidate), "boundary_match_within_3_bars": boundary_match_rate(base, candidate)})
    sensitivity_frame = pd.DataFrame(sensitivity)
    sensitivity_frame.to_csv(Path(config.output_dir) / "boundary_sensitivity.csv", index=False)

    reasons = combined.loc[combined.is_unknown, "unknown_reason"].str.get_dummies(sep=";").sum().sort_values(ascending=False)
    class_counts = combined["label"].value_counts()
    metrics = {
        "symbols": int(bars.symbol.nunique()), "bars": int(len(bars)), "segments": int(len(combined)),
        "coverage": float((~combined.is_unknown).mean()), "unknown_rate": float(combined.is_unknown.mean()),
        "median_duration_bars": float(combined.duration_bars.median()),
        "duration_p10_p90": [float(combined.duration_bars.quantile(.1)), float(combined.duration_bars.quantile(.9))],
        "mean_max_probability": float(combined.max_probability.mean()),
        "mean_normalized_entropy": float(combined.posterior_entropy.mean()),
        "mean_model_disagreement": float(combined.model_disagreement.mean()),
        "recognised_silhouette": silhouette,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "unknown_reasons": {str(k): int(v) for k, v in reasons.items()},
        "boundary_sensitivity": sensitivity,
    }
    out = Path(config.output_dir)
    (out / "evaluation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
