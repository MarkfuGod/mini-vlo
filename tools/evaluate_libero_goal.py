#!/usr/bin/env python3
"""Evaluate Semantic-Motion on LIBERO Goal using titles as weak gold.

Each demonstration is intentionally treated as one full-video task segment
labeled by its official LIBERO task title. This supports the requested
title-as-ground-truth experiment, but it does not create human subtask gold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    boundary_metrics,
    labeled_segment_metrics,
    normalized_edit_score,
    paired_bootstrap_ci,
    segmental_metrics,
    slot_f1,
    token_f1,
)
from src.runtime_utils import (
    file_sha256,
    git_revision,
    utc_now_iso,
    write_json,
)
from src.semantic_motion import (
    SourceInstructionRewriter,
    VLMRecognitionModel,
    VideoTaskPipeline,
    load_view_bundle,
)


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LIBERO Goal title-as-single-segment weak-ground-truth "
            "experiment with repeated Fixed/Ego/Fused inference."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/libero_goal/processed/manifest.json",
    )
    parser.add_argument(
        "--views",
        choices=["fixed", "ego", "fused", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--macro-window-sec", type=float, default=16.0)
    parser.add_argument("--macro-step-sec", type=float, default=8.0)
    parser.add_argument("--micro-window-sec", type=float, default=2.0)
    parser.add_argument("--micro-step-sec", type=float, default=1.0)
    parser.add_argument("--micro-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=_optional_float_env("VLM_INPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=_optional_float_env("VLM_OUTPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output",
        default="results/libero_goal_title_as_single_segment.json",
    )
    parser.add_argument(
        "--records-dir",
        default="results/libero_goal_title_records",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in {"a", "an", "the", "to", "of"}
    }


def title_slots(title: str) -> dict[str, str]:
    """Derive deterministic weak slots from a task title."""
    normalized = _normalize(title)
    patterns = [
        (
            r"^put (?:the )?(.+?) on (?:the )?(.+)$",
            "pick_and_place",
        ),
        (
            r"^place (?:the )?(.+?) on (?:the )?(.+)$",
            "pick_and_place",
        ),
        (
            r"^turn on (?:the )?(.+)$",
            "turn_on",
        ),
        (
            r"^turn off (?:the )?(.+)$",
            "turn_off",
        ),
        (
            r"^open (?:the )?(.+)$",
            "open",
        ),
        (
            r"^close (?:the )?(.+)$",
            "close",
        ),
    ]
    for pattern, action in patterns:
        match = re.match(pattern, normalized)
        if not match:
            continue
        groups = match.groups()
        if action == "pick_and_place":
            return {
                "action": action,
                "object": groups[0],
                "destination": groups[1],
            }
        destination = "on state" if action == "turn_on" else ""
        if action == "turn_off":
            destination = "off state"
        return {
            "action": action,
            "object": groups[0],
            "destination": destination,
        }
    return {
        "action": normalized.split(maxsplit=1)[0] if normalized else "",
        "object": normalized,
        "destination": "",
    }


def _prediction_text(record: dict[str, Any]) -> str:
    return " ; ".join(
        str(segment.get("task_instruction", "")).strip()
        for segment in record.get("task_segments", [])
        if str(segment.get("task_instruction", "")).strip()
    )


def _majority(values: list[str]) -> str:
    normalized = [_normalize(value) for value in values if value]
    return Counter(normalized).most_common(1)[0][0] if normalized else ""


def predicted_slots(record: dict[str, Any]) -> dict[str, str]:
    intents = [
        segment.get("macro_intent", {})
        for segment in record.get("task_segments", [])
    ]
    return {
        "action": _majority(
            [str(intent.get("task_type", "")) for intent in intents]
        ),
        "object": " ; ".join(
            dict.fromkeys(
                str(intent.get("target_object", "")).strip()
                for intent in intents
                if str(intent.get("target_object", "")).strip()
            )
        ),
        "destination": " ; ".join(
            dict.fromkeys(
                str(intent.get("destination", "")).strip()
                for intent in intents
                if str(intent.get("destination", "")).strip()
            )
        ),
    }


def _slot_contains(prediction: str, truth: str) -> bool:
    truth_tokens = _content_tokens(truth)
    return not truth_tokens or truth_tokens.issubset(_content_tokens(prediction))


def semantic_title_match(prediction: dict[str, str], truth: dict[str, str]) -> bool:
    return (
        _normalize(prediction.get("action", ""))
        == _normalize(truth.get("action", ""))
        and _slot_contains(prediction.get("object", ""), truth.get("object", ""))
        and _slot_contains(
            prediction.get("destination", ""),
            truth.get("destination", ""),
        )
    )


def _predicted_segments(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": float(segment.get("start_time_sec", 0.0)),
            "end_sec": float(segment.get("end_time_sec", 0.0)),
            "label": str(segment.get("task_instruction", "")),
        }
        for segment in record.get("task_segments", [])
    ]


def _usage_delta(
    before: dict[str, int],
    after: dict[str, int],
    *,
    input_price: float | None,
    output_price: float | None,
) -> dict[str, Any]:
    usage = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("requests", "input_tokens", "output_tokens")
    }
    if input_price is None or output_price is None:
        usage["estimated_cost_usd"] = None
    else:
        usage["estimated_cost_usd"] = (
            usage["input_tokens"] * input_price
            + usage["output_tokens"] * output_price
        ) / 1_000_000
    return usage


def evaluate_record(
    record: dict[str, Any],
    *,
    title: str,
    duration_sec: float,
) -> dict[str, Any]:
    truth_slots = title_slots(title)
    prediction_slots = predicted_slots(record)
    semantic_match = semantic_title_match(prediction_slots, truth_slots)
    predicted = _predicted_segments(record)
    temporal_predicted = [
        {**segment, "label": ""}
        for segment in predicted
    ]
    gold = [
        {
            "start_sec": 0.0,
            "end_sec": duration_sec,
            "label": "",
        }
    ]
    boundaries = [
        float(segment["start_sec"])
        for segment in predicted[1:]
        if 0.0 < float(segment["start_sec"]) < duration_sec
    ]
    label_match = lambda _pred_index, _gold_index: semantic_match
    predicted_labels = [str(segment["label"]) for segment in predicted]
    return {
        "ground_truth_policy": "official_title_as_one_full_video_segment",
        "ground_truth_slots": truth_slots,
        "predicted_slots": prediction_slots,
        "title_instruction_token_f1": token_f1(
            _prediction_text(record),
            title,
        ),
        "semantic_label_accuracy": float(semantic_match),
        "slot_f1": slot_f1([prediction_slots], [truth_slots]),
        "boundary_f1": {
            "0.5s": boundary_metrics(boundaries, [], tolerance_sec=0.5),
            "1.0s": boundary_metrics(boundaries, [], tolerance_sec=1.0),
        },
        "segment_f1_at_iou": {
            str(threshold): segmental_metrics(
                temporal_predicted,
                gold,
                iou_threshold=threshold,
            )
            for threshold in (0.5, 0.75)
        },
        "labeled_end_to_end_segment_f1": {
            str(threshold): labeled_segment_metrics(
                predicted,
                gold,
                label_match,
                iou_threshold=threshold,
            )
            for threshold in (0.5, 0.75)
        },
        "action_edit": normalized_edit_score(predicted_labels, [title]),
        "predicted_segment_count": len(predicted),
        "gold_segment_count": 1,
    }


METRIC_PATHS = {
    "title_instruction_token_f1": ("title_instruction_token_f1",),
    "semantic_label_accuracy": ("semantic_label_accuracy",),
    "slot_macro_f1": ("slot_f1", "macro_f1"),
    "boundary_f1_0.5s": ("boundary_f1", "0.5s", "f1"),
    "boundary_f1_1.0s": ("boundary_f1", "1.0s", "f1"),
    "segment_f1_iou_0.5": ("segment_f1_at_iou", "0.5", "f1"),
    "segment_f1_iou_0.75": ("segment_f1_at_iou", "0.75", "f1"),
    "labeled_e2e_f1_iou_0.5": (
        "labeled_end_to_end_segment_f1",
        "0.5",
        "f1",
    ),
    "labeled_e2e_f1_iou_0.75": (
        "labeled_end_to_end_segment_f1",
        "0.75",
        "f1",
    ),
    "action_edit": ("action_edit",),
}


def _metric_value(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row["metrics"]
    for key in path:
        value = value[key]
    return float(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    views = sorted({str(row["view"]) for row in valid})
    summary: dict[str, Any] = {
        "run_count": len(rows),
        "valid_run_count": len(valid),
        "error_run_count": len(rows) - len(valid),
        "by_view": {},
        "paired_view_deltas": {},
    }
    for view in views:
        members = [row for row in valid if row["view"] == view]
        summary["by_view"][view] = {
            name: (
                sum(_metric_value(row, path) for row in members) / len(members)
                if members
                else 0.0
            )
            for name, path in METRIC_PATHS.items()
        }
        summary["by_view"][view]["mean_latency_s"] = (
            sum(float(row["latency_s"]) for row in members) / len(members)
            if members
            else 0.0
        )
        costs = [
            float(row["usage"]["estimated_cost_usd"])
            for row in members
            if row["usage"].get("estimated_cost_usd") is not None
        ]
        summary["by_view"][view]["mean_estimated_cost_usd"] = (
            sum(costs) / len(costs) if costs else None
        )

    indexed = {
        (str(row["sample_id"]), int(row["repeat"]), str(row["view"])): row
        for row in valid
    }
    for comparison in (("fused", "fixed"), ("fused", "ego")):
        proposed, baseline = comparison
        metric_intervals: dict[str, Any] = {}
        for name, path in METRIC_PATHS.items():
            deltas = []
            for sample_id, repeat, view in indexed:
                if view != proposed:
                    continue
                counterpart = indexed.get((sample_id, repeat, baseline))
                if counterpart is None:
                    continue
                deltas.append(
                    _metric_value(indexed[(sample_id, repeat, proposed)], path)
                    - _metric_value(counterpart, path)
                )
            metric_intervals[name] = paired_bootstrap_ci(deltas)
        summary["paired_view_deltas"][f"{proposed}_minus_{baseline}"] = (
            metric_intervals
        )
    return summary


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["sample_id"]),
        str(row["view"]),
        int(row["repeat"]),
    )


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("--repeats must be >= 3 for this experiment")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest["samples"])
    if args.limit > 0:
        samples = samples[: args.limit]
    selected_views = (
        ["fixed", "ego", "fused"] if args.views == "all" else [args.views]
    )
    recognizer = VLMRecognitionModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    pipeline = VideoTaskPipeline(
        recognizer=recognizer,
        rewriter=SourceInstructionRewriter(),
    )
    output_path = Path(args.output)
    records_dir = Path(args.records_dir)
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {
            _row_key(row): row
            for row in previous.get("results", [])
            if isinstance(row, dict)
        }
    results = list(existing.values())

    def save() -> None:
        results.sort(key=_row_key)
        write_json(
            output_path,
            {
                "schema_version": "semantic-motion-libero-title-eval/v2",
                "formal": False,
                "evaluation": "title_as_single_segment",
                "warning": (
                    "Official LIBERO titles are intentionally used as weak "
                    "full-video single-segment ground truth. Temporal scores "
                    "measure over-segmentation against that experimental proxy, "
                    "not human atomic-subtask boundaries."
                ),
                "generated_at": utc_now_iso(),
                "code_revision": git_revision(ROOT),
                "model": recognizer.model,
                "base_url": recognizer.base_url,
                "request_timeout_s": recognizer.timeout,
                "input": {
                    "manifest": str(manifest_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "sample_count": len(samples),
                    "views": selected_views,
                    "repeats": args.repeats,
                },
                "pricing": {
                    "input_usd_per_million_tokens": args.input_cost_per_million,
                    "output_usd_per_million_tokens": args.output_cost_per_million,
                },
                "summary": summarize(results),
                "results": results,
            },
        )

    for sample in samples:
        bundle = load_view_bundle(args.manifest, sample["sample_id"])
        title = str(sample["ground_truth_instruction"])
        duration_sec = bundle.timebase.duration_sec
        for view in selected_views:
            for repeat in range(args.repeats):
                key = str(sample["sample_id"]), view, repeat
                if key in existing and existing[key].get("status") == "ok":
                    print(f"{key} already complete", flush=True)
                    continue
                before = recognizer.engine.usage_snapshot()
                started = time.monotonic()
                try:
                    record = pipeline.run_view_bundle(
                        bundle=bundle,
                        work_dir=(
                            Path(".semantic_motion_work")
                            / "libero_goal_title_eval"
                            / str(sample["sample_id"])
                            / view
                            / f"repeat_{repeat:02d}"
                        ),
                        source_instruction="",
                        view_mode=view,
                        macro_window_sec=args.macro_window_sec,
                        macro_step_sec=args.macro_step_sec,
                        macro_frames=args.max_frames,
                        micro_window_sec=args.micro_window_sec,
                        micro_step_sec=args.micro_step_sec,
                        micro_frames=args.micro_frames,
                        num_variants=0,
                    )
                    latency = time.monotonic() - started
                    record_dict = record.model_dump()
                    record_path = (
                        records_dir
                        / str(sample["sample_id"])
                        / view
                        / f"repeat_{repeat:02d}.json"
                    )
                    write_json(record_path, record_dict)
                    row = {
                        "sample_id": sample["sample_id"],
                        "view": view,
                        "repeat": repeat,
                        "status": "ok",
                        "ground_truth_title": title,
                        "ground_truth_source": sample["ground_truth_source"],
                        "ground_truth_scope": (
                            "experimental weak title-as-single-segment"
                        ),
                        "duration_sec": duration_sec,
                        "latency_s": latency,
                        "usage": _usage_delta(
                            before,
                            recognizer.engine.usage_snapshot(),
                            input_price=args.input_cost_per_million,
                            output_price=args.output_cost_per_million,
                        ),
                        "metrics": evaluate_record(
                            record_dict,
                            title=title,
                            duration_sec=duration_sec,
                        ),
                        "prediction": _prediction_text(record_dict),
                        "record_path": str(record_path),
                    }
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    row = {
                        "sample_id": sample["sample_id"],
                        "view": view,
                        "repeat": repeat,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": time.monotonic() - started,
                        "usage": _usage_delta(
                            before,
                            recognizer.engine.usage_snapshot(),
                            input_price=args.input_cost_per_million,
                            output_price=args.output_cost_per_million,
                        ),
                    }
                results = [
                    previous for previous in results if _row_key(previous) != key
                ]
                results.append(row)
                save()
                if row["status"] == "ok":
                    print(
                        f"{sample['sample_id']} [{view}] repeat={repeat} "
                        f"titleF1={row['metrics']['title_instruction_token_f1']:.3f} "
                        f"labelAcc={row['metrics']['semantic_label_accuracy']:.0f} "
                        f"latency={row['latency_s']:.1f}s",
                        flush=True,
                    )
                else:
                    print(f"{key} ERROR {row['error']}", flush=True)
    save()
    print(f"Saved LIBERO title experiment: {output_path}")


if __name__ == "__main__":
    main()
