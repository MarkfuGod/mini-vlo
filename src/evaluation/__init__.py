"""Evaluation contracts and metrics for Semantic-Motion."""

from src.evaluation.corruption import CorruptedMotion, inject_motion_corruption
from src.evaluation.gold import GoldAnnotation, load_gold, validate_gold
from src.evaluation.metrics import (
    binary_classification_metrics,
    boundary_metrics,
    calibration_metrics,
    coverage_accuracy,
    distinct_n,
    normalized_edit_score,
    segmental_metrics,
    self_bleu_overlap,
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
    "load_gold",
    "normalized_edit_score",
    "segmental_metrics",
    "self_bleu_overlap",
    "token_f1",
    "validate_gold",
]
