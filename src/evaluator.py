"""Metrics engine for evaluating VLM predictions against ground truth."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from rouge_score import rouge_scorer

from src.scenario import GroundTruth, Prediction


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.strip().lower()


def object_f1(pred: Prediction, gt: GroundTruth) -> dict[str, float]:
    """Token-level precision / recall / F1 for identified objects."""
    pred_set = {_normalize(o) for o in pred.objects}
    gt_set = {_normalize(o) for o in gt.objects}

    if not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def task_accuracy(pred: Prediction, gt: GroundTruth) -> float:
    """1.0 if predicted task_type matches ground truth, else 0.0."""
    return 1.0 if _normalize(pred.task_type) == _normalize(gt.task_type) else 0.0


def action_rouge_l(pred: Prediction, gt: GroundTruth) -> float:
    """ROUGE-L F1 between predicted and ground-truth action sequences."""
    pred_text = " ; ".join(pred.action_sequence)
    gt_text = " ; ".join(gt.action_sequence)
    if not gt_text:
        return 1.0 if not pred_text else 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gt_text, pred_text)
    return scores["rougeL"].fmeasure


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity (pure Python, no numpy/torch)."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    all_keys = set(counter_a) | set(counter_b)

    dot = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def semantic_similarity(pred: Prediction, gt: GroundTruth) -> float:
    """Cosine similarity between predicted and ground-truth descriptions."""
    pred_text = (
        f"objects: {', '.join(pred.objects)}. "
        f"task: {pred.task_type}. "
        f"actions: {', '.join(pred.action_sequence)}. "
        f"target: {pred.target_object}."
    )
    gt_text = (
        f"objects: {', '.join(gt.objects)}. "
        f"task: {gt.task_type}. "
        f"actions: {', '.join(gt.action_sequence)}. "
        f"target: {gt.target_object}."
    )
    return _tfidf_cosine(pred_text, gt_text)


def spatial_accuracy(pred: Prediction, gt: GroundTruth) -> float:
    """Fraction of ground-truth spatial relations matched (case-insensitive substring)."""
    if not gt.spatial_relations:
        return 1.0

    pred_blob = " | ".join(_normalize(r) for r in pred.spatial_relations)
    matched = 0
    for rel in gt.spatial_relations:
        key_parts = _normalize(rel).split()
        if all(part in pred_blob for part in key_parts):
            matched += 1

    return matched / len(gt.spatial_relations)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

WEIGHTS = {
    "object_f1": 0.20,
    "task_accuracy": 0.20,
    "action_rouge_l": 0.20,
    "semantic_similarity": 0.20,
    "spatial_accuracy": 0.20,
}


@dataclass
class ScenarioResult:
    scenario_id: str
    object_f1: float = 0.0
    task_accuracy: float = 0.0
    action_rouge_l: float = 0.0
    semantic_similarity: float = 0.0
    spatial_accuracy: float = 0.0
    composite: float = 0.0


@dataclass
class EvalReport:
    results: list[ScenarioResult] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_object_f1(self) -> float:
        return self._mean("object_f1")

    @property
    def mean_task_accuracy(self) -> float:
        return self._mean("task_accuracy")

    @property
    def mean_action_rouge_l(self) -> float:
        return self._mean("action_rouge_l")

    @property
    def mean_semantic_similarity(self) -> float:
        return self._mean("semantic_similarity")

    @property
    def mean_spatial_accuracy(self) -> float:
        return self._mean("spatial_accuracy")

    @property
    def mean_composite(self) -> float:
        return self._mean("composite")

    def summary_table(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  Mini-VLO  Evaluation Report",
            "=" * 60,
            f"  Scenarios evaluated : {len(self.results)}",
            "-" * 60,
            f"  Object Recognition F1      : {self.mean_object_f1:.3f}",
            f"  Task Classification Acc     : {self.mean_task_accuracy:.3f}",
            f"  Action Sequence ROUGE-L     : {self.mean_action_rouge_l:.3f}",
            f"  Semantic Similarity         : {self.mean_semantic_similarity:.3f}",
            f"  Spatial Reasoning Acc       : {self.mean_spatial_accuracy:.3f}",
            "-" * 60,
            f"  Composite Score             : {self.mean_composite:.3f}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)


def evaluate_single(scenario_id: str, pred: Prediction, gt: GroundTruth) -> ScenarioResult:
    """Compute all metrics for one (prediction, ground_truth) pair."""
    of1 = object_f1(pred, gt)["f1"]
    ta = task_accuracy(pred, gt)
    ar = action_rouge_l(pred, gt)
    ss = semantic_similarity(pred, gt)
    sa = spatial_accuracy(pred, gt)

    composite = (
        WEIGHTS["object_f1"] * of1
        + WEIGHTS["task_accuracy"] * ta
        + WEIGHTS["action_rouge_l"] * ar
        + WEIGHTS["semantic_similarity"] * ss
        + WEIGHTS["spatial_accuracy"] * sa
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        object_f1=of1,
        task_accuracy=ta,
        action_rouge_l=ar,
        semantic_similarity=ss,
        spatial_accuracy=sa,
        composite=composite,
    )
