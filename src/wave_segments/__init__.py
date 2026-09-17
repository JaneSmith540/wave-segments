"""Probabilistic, variable-length market wave description toolkit."""

from .config import PipelineConfig
from .pipeline import WavePipeline
from .stability import (
    bootstrap_boundary_stability,
    boundary_precision_recall_f1,
    merge_consecutive_same_label,
)
from .validation import CalibratedSegmentClassifier, PurgedWalkForwardSplit

__all__ = [
    "CalibratedSegmentClassifier",
    "PipelineConfig",
    "PurgedWalkForwardSplit",
    "WavePipeline",
    "bootstrap_boundary_stability",
    "boundary_precision_recall_f1",
    "merge_consecutive_same_label",
]
__version__ = "0.1.0"
