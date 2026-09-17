from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .analytics import build_descriptions, build_transition_matrix, build_tree_rules
from .config import PipelineConfig
from .features import extract_segment_features
from .model import ProbabilisticWaveModel
from .discovery import DiscoveryConfig, MultiModelDiscoverer
from .duration import causal_duration_filter, decode_hsmm
from .review import create_review_table
from .schema import normalize_ohlcv
from .segmentation import segment_bars
from .stability import bootstrap_boundary_stability, merge_consecutive_same_label
from .visualization import create_all_visualizations


@dataclass
class PipelineResult:
    segments: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    descriptions: pd.DataFrame
    transitions: pd.DataFrame
    rules: str
    review: pd.DataFrame
    merged_segments: pd.DataFrame


class WavePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        if self.config.discovery_backend == "ensemble":
            model = self.config.model
            self.model = MultiModelDiscoverer(DiscoveryConfig(
                n_components=model.n_components, random_state=model.random_state,
                ensemble_size=model.ensemble_size,
                min_cluster_samples=model.min_cluster_samples,
                min_cluster_symbols=model.min_cluster_symbols,
                min_cluster_years=model.min_cluster_years,
                min_density_quantile=model.min_density_quantile,
                max_disagreement=model.max_disagreement,
                min_consensus_probability=model.min_max_probability,
                max_normalized_entropy=model.max_normalized_entropy,
                max_boundary_uncertainty=model.max_boundary_uncertainty,
                max_context_conflict=model.max_context_conflict,
                min_recognizability=model.min_recognizability,
            ))
        elif self.config.discovery_backend == "gmm":
            self.model = ProbabilisticWaveModel(self.config.model)
        else:
            raise ValueError("discovery_backend must be 'ensemble' or 'gmm'")

    def fit_run(
        self,
        bars: pd.DataFrame,
        write_outputs: bool = True,
        *,
        market_context: pd.DataFrame | None = None,
        sector_context: pd.DataFrame | None = None,
        multi_timeframe_context: pd.DataFrame | None = None,
    ) -> PipelineResult:
        bars = normalize_ohlcv(bars)
        segments = segment_bars(bars, self.config.segmentation, use_changepoints=True)
        stability_cfg = self.config.segmentation
        if stability_cfg.bootstrap_iterations > 0:
            _, stability = bootstrap_boundary_stability(
                bars, stability_cfg, iterations=stability_cfg.bootstrap_iterations,
                tolerance=stability_cfg.bootstrap_tolerance,
                price_noise=stability_cfg.bootstrap_price_noise,
                volume_noise=stability_cfg.bootstrap_volume_noise,
                random_state=self.config.model.random_state, use_changepoints=True,
            )
            lookup = stability.set_index(["symbol", "boundary_idx"])["stability_frequency"].to_dict()
            start_stability = [lookup.get((row.symbol, int(row.start_idx)), 1.0) for row in segments.itertuples()]
            end_stability = [lookup.get((row.symbol, int(row.end_idx)), 1.0) for row in segments.itertuples()]
            segments["detector_start_boundary_probability"] = segments["start_boundary_probability"]
            segments["detector_end_boundary_probability"] = segments["end_boundary_probability"]
            segments["detector_boundary_probability"] = segments["boundary_probability"]
            segments["start_boundary_stability"] = start_stability
            segments["end_boundary_stability"] = end_stability
            segments["bootstrap_boundary_stability"] = np.minimum(start_stability, end_stability)
            segments["start_boundary_probability"] = np.minimum(
                segments["detector_start_boundary_probability"], segments["start_boundary_stability"]
            )
            segments["end_boundary_probability"] = np.minimum(
                segments["detector_end_boundary_probability"], segments["end_boundary_stability"]
            )
            segments["boundary_probability"] = np.minimum(
                segments["start_boundary_probability"], segments["end_boundary_probability"]
            )
            segments["boundary_uncertainty"] = 1 - segments["boundary_probability"]
        features = extract_segment_features(
            bars,
            segments,
            market_context=market_context,
            sector_context=sector_context,
            multi_timeframe_context=multi_timeframe_context,
        )
        combined = self.model.fit_predict(features)
        combined = causal_duration_filter(
            combined, min_dwell_bars=self.config.duration.causal_min_dwell_bars
        )
        prediction_columns = [
            c for c in combined.columns
            if c == "segment_id" or c.startswith(("prob_", "gmm_", "dpgmm_", "ensemble_", "hdbscan_")) or c in {
                "soft_label", "max_probability", "posterior_entropy", "model_disagreement",
                "density_score", "density_is_low", "recognizability", "unknown_reason",
                "candidate_label", "label", "is_unknown", "context_conflict",
                "suggested_semantic_label", "cluster_support_valid",
                "boundary_unstable",
                "causal_duration_label", "hsmm_label", "hsmm_changed", "hsmm_path_score",
            }
        ]
        if self.config.duration.enabled:
            historical = decode_hsmm(
                combined, max_duration=self.config.duration.max_segments,
                min_duration=self.config.duration.min_segments,
            )
            historical["pre_hsmm_label"] = historical["label"]
            historical["label"] = historical["hsmm_label"]
        else:
            historical = combined
        prediction_columns = list(dict.fromkeys(prediction_columns + [
            c for c in ("causal_duration_label", "pre_hsmm_label", "hsmm_label", "hsmm_changed", "hsmm_path_score")
            if c in historical
        ]))
        labels = historical[prediction_columns].copy()
        merged = merge_consecutive_same_label(historical)
        # Merging changes the path itself; recompute path-dependent features
        # instead of retaining the first child segment's return/drawdown values.
        merged = extract_segment_features(
            bars, merged, market_context=market_context, sector_context=sector_context,
            multi_timeframe_context=multi_timeframe_context,
        )
        descriptions = build_descriptions(merged)
        transitions = build_transition_matrix(merged)
        rules = build_tree_rules(merged)
        review = create_review_table(combined)
        result = PipelineResult(segments, features, labels, descriptions, transitions, rules, review, merged)
        if write_outputs:
            self.write_outputs(result, bars)
        return result

    def write_outputs(self, result: PipelineResult, bars: pd.DataFrame) -> Path:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tables = {
            "candidate_segments.parquet": result.segments,
            "segment_features.parquet": result.features,
            "probabilistic_labels.parquet": result.labels,
            "class_descriptions.csv": result.descriptions,
            "transition_matrix.csv": result.transitions,
            "human_review.csv": result.review,
            "merged_labeled_segments.parquet": result.merged_segments,
        }
        for name, frame in tables.items():
            path = out / name
            if path.suffix == ".parquet":
                frame.to_parquet(path, index=False)
            else:
                # A transition matrix carries state names in its index; ordinary
                # record tables already have explicit identifiers.
                frame.to_csv(path, index=name == "transition_matrix.csv")
        (out / "decision_tree_rules.txt").write_text(result.rules, encoding="utf-8")
        (out / "run_config.json").write_text(json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        create_all_visualizations(bars, result.merged_segments, result.transitions, out)
        return out
