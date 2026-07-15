"""Evaluation contracts and metrics for Semantic-Motion."""

from src.evaluation.corruption import CorruptedMotion, inject_motion_corruption
from src.evaluation.gold import GoldAnnotation, load_gold, validate_gold
from src.evaluation.metrics import (
    binary_classification_metrics,
    boundary_metrics,
    calibration_metrics,
    coverage_accuracy,
    distinct_n,
    labeled_segment_metrics,
    match_temporal_segments,
    normalized_edit_score,
    paired_bootstrap_ci,
    segmental_metrics,
    self_bleu_overlap,
    slot_f1,
    token_f1,
)

__all__ = [
    "CorruptedMotion",
    "GoldAnnotation",
    "binary_classification_metrics",
    "boundary_metrics",
    "calibration_metrics",
    "coverage_accuracy",
    "distinct_n",
    "inject_motion_corruption",
    "labeled_segment_metrics",
    "load_gold",
    "match_temporal_segments",
    "normalized_edit_score",
    "paired_bootstrap_ci",
    "segmental_metrics",
    "self_bleu_overlap",
    "slot_f1",
    "token_f1",
    "validate_gold",
]
