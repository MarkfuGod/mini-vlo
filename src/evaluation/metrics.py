"""Independent metrics for temporal, semantic, refinement, and calibration quality."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Callable, Iterable


def _tokens(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    text = value if isinstance(value, str) else " ".join(str(item) for item in value)
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str | Iterable[str], truth: str | Iterable[str]) -> float:
    pred = Counter(_tokens(prediction))
    gold = Counter(_tokens(truth))
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / max(precision + recall, 1e-12)


def boundary_metrics(
    predicted_sec: list[float],
    gold_sec: list[float],
    tolerance_sec: float = 0.5,
) -> dict[str, float]:
    """Greedily one-to-one match boundaries within the stated tolerance."""
    unmatched = set(range(len(gold_sec)))
    matches = 0
    errors: list[float] = []
    for prediction in sorted(predicted_sec):
        candidates = [
            (abs(prediction - gold_sec[index]), index)
            for index in unmatched
            if abs(prediction - gold_sec[index]) <= tolerance_sec
        ]
        if not candidates:
            continue
        error, index = min(candidates)
        unmatched.remove(index)
        matches += 1
        errors.append(error)
    precision = matches / len(predicted_sec) if predicted_sec else float(not gold_sec)
    recall = matches / len(gold_sec) if gold_sec else float(not predicted_sec)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(matches),
        "fp": float(len(predicted_sec) - matches),
        "fn": float(len(gold_sec) - matches),
        "mean_abs_error_sec": sum(errors) / len(errors) if errors else 0.0,
        "tolerance_sec": tolerance_sec,
    }


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def segmental_metrics(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Match labeled or unlabeled temporal segments by maximum IoU."""
    unmatched = set(range(len(gold)))
    matched_ious: list[float] = []
    for pred in predicted:
        pred_interval = (float(pred["start_sec"]), float(pred["end_sec"]))
        pred_label = str(pred.get("label", "")).lower()
        candidates = []
        for index in unmatched:
            gold_item = gold[index]
            gold_label = str(gold_item.get("label", "")).lower()
            if pred_label and gold_label and pred_label != gold_label:
                continue
            iou = temporal_iou(
                pred_interval,
                (float(gold_item["start_sec"]), float(gold_item["end_sec"])),
            )
            candidates.append((iou, index))
        if not candidates:
            continue
        iou, index = max(candidates)
        if iou >= iou_threshold:
            unmatched.remove(index)
            matched_ious.append(iou)
    tp = len(matched_ious)
    precision = tp / len(predicted) if predicted else float(not gold)
    recall = tp / len(gold) if gold else float(not predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "threshold": iou_threshold,
        "tp": float(tp),
        "fp": float(len(predicted) - tp),
        "fn": float(len(gold) - tp),
    }


def match_temporal_segments(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Greedily match segment pairs by descending temporal IoU."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        pred_interval = (float(pred["start_sec"]), float(pred["end_sec"]))
        for gold_index, truth in enumerate(gold):
            iou = temporal_iou(
                pred_interval,
                (float(truth["start_sec"]), float(truth["end_sec"])),
            )
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gold_index))
    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, gold_index in sorted(candidates, reverse=True):
        if pred_index in matched_predicted or gold_index in matched_gold:
            continue
        matched_predicted.add(pred_index)
        matched_gold.add(gold_index)
        matches.append((pred_index, gold_index, iou))
    return sorted(matches, key=lambda item: item[1])


def labeled_segment_metrics(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    label_match: Callable[[int, int], bool],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute end-to-end segment F1 requiring temporal and semantic matches."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        pred_interval = (float(pred["start_sec"]), float(pred["end_sec"]))
        for gold_index, truth in enumerate(gold):
            if not label_match(pred_index, gold_index):
                continue
            iou = temporal_iou(
                pred_interval,
                (float(truth["start_sec"]), float(truth["end_sec"])),
            )
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gold_index))
    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, gold_index in sorted(candidates, reverse=True):
        if pred_index in matched_predicted or gold_index in matched_gold:
            continue
        matched_predicted.add(pred_index)
        matched_gold.add(gold_index)
        matches.append((pred_index, gold_index, iou))
    tp = len(matches)
    precision = tp / len(predicted) if predicted else float(not gold)
    recall = tp / len(gold) if gold else float(not predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(len(predicted) - tp),
        "fn": float(len(gold) - tp),
        "mean_iou": (
            sum(match[2] for match in matches) / tp if matches else 0.0
        ),
        "threshold": iou_threshold,
    }


def slot_f1(
    predicted: list[dict[str, str]],
    gold: list[dict[str, str]],
    *,
    slots: Iterable[str] = ("action", "object", "destination"),
) -> dict[str, Any]:
    """Compute token F1 per semantic slot for aligned label pairs."""
    slot_names = list(slots)
    pair_count = min(len(predicted), len(gold))
    if pair_count == 0:
        return {
            "per_slot": {slot: 0.0 for slot in slot_names},
            "macro_f1": 0.0,
            "pair_count": 0,
        }
    predicted = predicted[:pair_count]
    gold = gold[:pair_count]
    per_slot = {
        slot: token_f1(
            [row.get(slot, "") for row in predicted],
            [row.get(slot, "") for row in gold],
        )
        for slot in slot_names
    }
    return {
        "per_slot": per_slot,
        "macro_f1": (
            sum(per_slot.values()) / len(per_slot) if per_slot else 0.0
        ),
        "pair_count": pair_count,
    }


def paired_bootstrap_ci(
    paired_deltas: list[float],
    *,
    confidence: float = 0.95,
    draws: int = 4000,
    seed: int = 20260715,
) -> dict[str, float] | None:
    """Bootstrap a confidence interval for the mean paired difference."""
    if len(paired_deltas) < 2:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(max(100, draws)):
        values = [
            generator.choice(paired_deltas) for _ in range(len(paired_deltas))
        ]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = int(alpha * (len(estimates) - 1))
    upper_index = int((1.0 - alpha) * (len(estimates) - 1))
    return {
        "mean_delta": sum(paired_deltas) / len(paired_deltas),
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "confidence": confidence,
        "draws": float(len(estimates)),
        "n": float(len(paired_deltas)),
    }


def normalized_edit_score(predicted: list[str], gold: list[str]) -> float:
    """One minus normalized Levenshtein distance over ordered action labels."""
    rows, cols = len(predicted) + 1, len(gold) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            cost = int(predicted[row - 1].lower() != gold[col - 1].lower())
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + cost,
            )
    denominator = max(len(predicted), len(gold), 1)
    return 1.0 - matrix[-1][-1] / denominator


def binary_classification_metrics(
    truth: list[str],
    prediction: list[str],
    *,
    positive_label: str = "keep",
    scores: list[float] | None = None,
) -> dict[str, float]:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    positive = [item == positive_label for item in truth]
    predicted_positive = [item == positive_label for item in prediction]
    tp = sum(a and b for a, b in zip(positive, predicted_positive))
    fp = sum((not a) and b for a, b in zip(positive, predicted_positive))
    fn = sum(a and (not b) for a, b in zip(positive, predicted_positive))
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    result = {
        "accuracy": (tp + tn) / len(truth) if truth else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_keep_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
    if scores is not None:
        result["auroc"] = binary_auroc(positive, scores)
    return result


def binary_auroc(truth: list[bool], scores: list[float]) -> float:
    """Pairwise AUROC; returns NaN when only one class is present."""
    if len(truth) != len(scores):
        raise ValueError("truth and score lengths differ")
    positives = [score for label, score in zip(truth, scores) if label]
    negatives = [score for label, score in zip(truth, scores) if not label]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def calibration_metrics(
    correctness: list[bool],
    confidence: list[float],
    bins: int = 10,
) -> dict[str, Any]:
    if len(correctness) != len(confidence):
        raise ValueError("correctness and confidence lengths differ")
    if not correctness:
        return {"brier": 0.0, "ece": 0.0, "bins": []}
    clipped = [max(0.0, min(1.0, value)) for value in confidence]
    brier = sum(
        (score - float(correct)) ** 2
        for score, correct in zip(clipped, correctness)
    ) / len(correctness)
    rows = []
    ece = 0.0
    for index in range(max(1, bins)):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            position
            for position, score in enumerate(clipped)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(clipped[position] for position in members) / len(members)
        accuracy = sum(correctness[position] for position in members) / len(members)
        ece += len(members) / len(correctness) * abs(mean_confidence - accuracy)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"brier": brier, "ece": ece, "bins": rows}


def coverage_accuracy(
    correctness: list[bool],
    confidence: list[float],
    thresholds: Iterable[float] = (0.0, 0.5, 0.7, 0.8, 0.9),
) -> list[dict[str, float]]:
    rows = []
    total = max(len(correctness), 1)
    for threshold in thresholds:
        selected = [
            index for index, score in enumerate(confidence) if score >= threshold
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": len(selected) / total,
                "accuracy": (
                    sum(correctness[index] for index in selected) / len(selected)
                    if selected
                    else 0.0
                ),
            }
        )
    return rows


def distinct_n(texts: list[str], n: int = 2) -> float:
    ngrams = []
    for text in texts:
        tokens = _tokens(text)
        ngrams.extend(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def self_bleu_overlap(texts: list[str], n: int = 2) -> float:
    """Pairwise n-gram overlap proxy; lower values indicate greater diversity."""
    if len(texts) < 2:
        return 0.0
    scores = []
    for index, text in enumerate(texts):
        tokens = _tokens(text)
        candidate = {
            tuple(tokens[position : position + n])
            for position in range(len(tokens) - n + 1)
        }
        references = set()
        for other_index, other in enumerate(texts):
            if other_index == index:
                continue
            other_tokens = _tokens(other)
            references.update(
                tuple(other_tokens[position : position + n])
                for position in range(len(other_tokens) - n + 1)
            )
        scores.append(len(candidate & references) / len(candidate) if candidate else 0.0)
    return sum(scores) / len(scores)
