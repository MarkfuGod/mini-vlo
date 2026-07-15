#!/usr/bin/env python3
"""Evaluate one VideoTaskRecord against adjudicated paired-view gold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    boundary_metrics,
    distinct_n,
    labeled_segment_metrics,
    load_gold,
    match_temporal_segments,
    normalized_edit_score,
    segmental_metrics,
    self_bleu_overlap,
    slot_f1,
    token_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--allow-pending-debug",
        action="store_true",
        help="Inspect pending packets; results are marked non-formal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold, formal=not args.allow_pending_debug)
    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    predicted_sample_id = str(
        prediction.get("view_bundle", {}).get(
            "sample_id",
            prediction.get("metadata", {}).get("sample_id", ""),
        )
    )
    if predicted_sample_id and predicted_sample_id != gold.source_sample_id:
        raise ValueError(
            f"Gold source {gold.source_sample_id!r} does not match prediction "
            f"{predicted_sample_id!r}"
        )
    segments = [
        segment
        for segment in prediction.get("task_segments", [])
        if float(segment.get("end_time_sec", 0.0)) > gold.clip_start_sec
        and float(segment.get("start_time_sec", 0.0)) < gold.clip_end_sec
    ]
    predicted_boundaries = [
        float(segment["start_time_sec"])
        for segment in segments
        if gold.clip_start_sec < float(segment["start_time_sec"]) < gold.clip_end_sec
    ]
    predicted_intervals = [
        {
            "start_sec": float(segment["start_time_sec"]),
            "end_sec": float(segment["end_time_sec"]),
            "label": str(segment.get("macro_intent", {}).get("task_type", "")),
        }
        for segment in segments
    ]
    gold_intervals = [
        {
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "label": segment.task_type,
        }
        for segment in gold.segments
    ]
    temporal_matches = match_temporal_segments(
        predicted_intervals,
        gold_intervals,
        iou_threshold=0.5,
    )
    label_matches = {
        (pred_index, gold_index): (
            str(
                segments[pred_index]
                .get("macro_intent", {})
                .get("task_type", "")
            ).lower()
            == gold.segments[gold_index].task_type.lower()
        )
        for pred_index, gold_index, _ in temporal_matches
    }

    macro_scores = []
    micro_scores = []
    order_scores = []
    augmentation_texts = []
    augmentation_audit = []
    for pred, truth in zip(segments, gold.segments):
        intent = pred.get("macro_intent", {})
        macro_scores.append(
            {
                "task_accuracy": float(
                    str(intent.get("task_type", "")).lower()
                    == truth.task_type.lower()
                ),
                "target_f1": token_f1(
                    str(intent.get("target_object", "")),
                    truth.target_object,
                ),
                "destination_f1": token_f1(
                    str(intent.get("destination", "")),
                    truth.destination,
                ),
            }
        )
        pred_micro = pred.get("micro_instructions", [])
        pred_text = [str(item.get("text", "")) for item in pred_micro]
        gold_text = [item.text for item in truth.micro_actions]
        order_scores.append(normalized_edit_score(pred_text, gold_text))
        micro_scores.append(
            {
                "step_f1": token_f1(pred_text, gold_text),
                "body_part_f1": token_f1(
                    [str(item.get("body_part", "")) for item in pred_micro],
                    [item.body_part for item in truth.micro_actions],
                ),
                "contact_state_f1": token_f1(
                    [str(item.get("contact_state", "")) for item in pred_micro],
                    [item.contact_state for item in truth.micro_actions],
                ),
            }
        )
        augmentation_texts.extend(
            str(item.get("text", ""))
            for item in pred.get("augmented_instructions", [])
        )
        augmentation_audit.extend(
            item
            for item in pred.get("metadata", {}).get("augmentation_audit", [])
            if isinstance(item, dict)
        )

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    payload = {
        "formal": gold.annotation_status == "adjudicated",
        "sample_id": gold.sample_id,
        "boundary": {
            str(tolerance): boundary_metrics(
                predicted_boundaries,
                [
                    boundary
                    for boundary in gold.boundaries_sec
                    if gold.clip_start_sec < boundary < gold.clip_end_sec
                ],
                tolerance_sec=tolerance,
            )
            for tolerance in (0.5, 1.0)
        },
        "segmental_f1_at_iou": {
            str(threshold): segmental_metrics(
                predicted_intervals,
                gold_intervals,
                threshold,
            )
            for threshold in (0.5, 0.75)
        },
        "semantic_label_accuracy": (
            sum(label_matches.values()) / len(label_matches)
            if label_matches
            else 0.0
        ),
        "semantic_label_temporal_coverage": (
            len(temporal_matches) / len(gold_intervals)
            if gold_intervals
            else 1.0
        ),
        "labeled_end_to_end_segment_f1": {
            str(threshold): labeled_segment_metrics(
                predicted_intervals,
                gold_intervals,
                lambda pred_index, gold_index: label_matches.get(
                    (pred_index, gold_index),
                    False,
                ),
                iou_threshold=threshold,
            )
            for threshold in (0.5, 0.75)
        },
        "macro": {
            "task_accuracy": mean(macro_scores, "task_accuracy"),
            "target_f1": mean(macro_scores, "target_f1"),
            "destination_f1": mean(macro_scores, "destination_f1"),
        },
        "micro": {
            "step_f1": mean(micro_scores, "step_f1"),
            "body_part_f1": mean(micro_scores, "body_part_f1"),
            "contact_state_f1": mean(micro_scores, "contact_state_f1"),
            "order_edit_score": (
                sum(order_scores) / len(order_scores) if order_scores else 0.0
            ),
        },
        "slot_f1": slot_f1(
            [
                {
                    "action": str(
                        segment.get("macro_intent", {}).get("task_type", "")
                    ),
                    "object": str(
                        segment.get("macro_intent", {}).get(
                            "target_object",
                            "",
                        )
                    ),
                    "destination": str(
                        segment.get("macro_intent", {}).get(
                            "destination",
                            "",
                        )
                    ),
                }
                for segment in segments
            ],
            [
                {
                    "action": segment.task_type,
                    "object": segment.target_object,
                    "destination": segment.destination,
                }
                for segment in gold.segments
            ],
        ),
        "augmentation": {
            "code_level_validation": "disabled",
            "included_variant_count": sum(
                bool(item.get("included")) for item in augmentation_audit
            ),
            "distinct_1": distinct_n(augmentation_texts, 1),
            "distinct_2": distinct_n(augmentation_texts, 2),
            "self_bleu_2_overlap": self_bleu_overlap(augmentation_texts, 2),
        },
        "provenance": {
            "gold": str(args.gold),
            "prediction": str(args.prediction),
            "pending_debug": args.allow_pending_debug,
        },
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
