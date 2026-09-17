from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class SegmentationConfig:
    min_bars: int = 4
    atr_period: int = 14
    atr_reversal: float = 2.2
    return_penalty: float = 5.0
    volume_penalty: float = 7.0
    merge_tolerance: int = 2
    min_boundary_probability: float = 0.34
    min_boundary_votes: int = 2
    single_source_min_probability: float = 0.75
    bootstrap_iterations: int = 0
    bootstrap_tolerance: int = 3
    bootstrap_price_noise: float = 0.002
    bootstrap_volume_noise: float = 0.08
    # Independent candidate generators; enabled only with ``use_changepoints``.
    use_adaptive_structure: bool = True
    use_ruptures: bool = True
    use_fractal_pivots: bool = True
    fractal_window: int = 2
    fractal_min_prominence_atr: float = 0.35
    use_bocpd: bool = True
    bocpd_hazard: float = 0.04
    bocpd_min_run_length: int = 5
    bocpd_change_probability: float = 0.52
    bocpd_max_run_length: int = 80


@dataclass
class ModelConfig:
    n_components: int = 5
    ensemble_size: int = 5
    random_state: int = 42
    min_max_probability: float = 0.52
    max_normalized_entropy: float = 0.78
    max_disagreement: float = 0.22
    min_density_quantile: float = 0.04
    min_recognizability: float = 0.48
    max_boundary_uncertainty: float = 0.44
    max_context_conflict: float = 0.75
    min_cluster_samples: int = 5
    min_cluster_symbols: int = 2
    min_cluster_years: int = 1


@dataclass
class DurationConfig:
    enabled: bool = True
    min_segments: int = 2
    max_segments: int = 20
    causal_min_dwell_bars: int = 5


@dataclass
class PipelineConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    duration: DurationConfig = field(default_factory=DurationConfig)
    discovery_backend: str = "ensemble"
    output_dir: str = "outputs"

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            segmentation=SegmentationConfig(**raw.get("segmentation", {})),
            model=ModelConfig(**raw.get("model", {})),
            duration=DurationConfig(**raw.get("duration", {})),
            discovery_backend=raw.get("discovery_backend", "ensemble"),
            output_dir=raw.get("output_dir", "outputs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
