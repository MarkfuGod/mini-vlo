#!/usr/bin/env python3
"""Score a vendored Video2Tasks run against weak folder-name gold."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer

from src.evaluation import (
    boundary_metrics,
    distinct_n,
    normalized_edit_score,
    segmental_metrics,
    self_bleu_overlap,
    token_f1,
)
from src.evaluator import _tfidf_cosine
from src.runtime_utils import file_sha256, git_revision, utc_now_iso, write_json


STOPWORDS = {"a", "an", "the", "to", "of", "on", "in", "and", "or"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Video2Tasks predictions using motion folder names as "
            "weak single-segment gold."
        )
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--output",
        default="",
        help="Defaults to <prediction_stem>_metrics.json",
    )
    return parser.parse_args()


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in STOPWORDS
    }


def action_title(sample_id: str) -> str:
    return sample_id.replace("_", " ").strip()


def prediction_text(row: dict[str, Any]) -> str:
    segments = row.get("prediction", {}).get("segments", [])
    parts = [
        str(segment.get("instruction", "")).strip()
        for segment in segments
        if str(segment.get("instruction", "")).strip()
    ]
    return " ; ".join(parts)


def action_rouge_l(prediction: str, gold: str) -> float:
    if not gold:
        return 1.0 if not prediction else 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold, prediction)["rougeL"].fmeasure


def keyword_hit(prediction: str, gold: str) -> float:
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return 1.0
    pred_tokens = _content_tokens(prediction)
    return float(bool(gold_tokens & pred_tokens))


def loose_title_match(prediction: str, gold: str) -> float:
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return 1.0
    pred_tokens = _content_tokens(prediction)
    return float(gold_tokens.issubset(pred_tokens))


def strict_title_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def composite_score(metrics: dict[str, float]) -> float:
    return (
        0.30 * metrics["action_name_token_f1"]
        + 0.20 * metrics["action_rouge_l"]
        + 0.20 * metrics["semantic_similarity"]
        + 0.15 * metrics["keyword_hit"]
        + 0.10 * metrics["loose_title_match"]
        + 0.05 * metrics["segment_count_accuracy"]
    )


def evaluate_row(row: dict[str, Any], *, gold_title: str) -> dict[str, Any]:
    prediction = prediction_text(row)
    fps = float(row.get("fps", 0.0) or 0.0)
    frame_count = int(row.get("frame_count", 0) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    segments = row.get("prediction", {}).get("segments", [])
    predicted_intervals = [
        {
            "start_sec": float(segment.get("start_frame", 0)) / fps,
            "end_sec": float(segment.get("end_frame", 0)) / fps,
            "label": "",
        }
        for segment in segments
        if fps > 0
    ]
    gold_intervals = [
        {
            "start_sec": 0.0,
            "end_sec": duration_sec,
            "label": "",
        }
    ]
    boundaries = [
        float(segment.get("start_frame", 0)) / fps
        for segment in segments[1:]
        if fps > 0
    ]
    metrics = {
        "action_name_token_f1": token_f1(prediction, gold_title),
        "action_rouge_l": action_rouge_l(prediction, gold_title),
        "semantic_similarity": _tfidf_cosine(prediction, gold_title),
        "keyword_hit": keyword_hit(prediction, gold_title),
        "loose_title_match": loose_title_match(prediction, gold_title),
        "strict_title_match": strict_title_match(prediction, gold_title),
        "segment_count_accuracy": float(len(segments) == 1),
        "predicted_segment_count": float(len(segments)),
    }
    metrics["composite"] = composite_score(metrics)
    return {
        "sample_id": row.get("sample_id", ""),
        "gold_title": gold_title,
        "prediction_text": prediction,
        "status": "failed" if row.get("error") else "ok",
        "error": row.get("error", ""),
        "metrics": metrics,
        "temporal": {
            "duration_sec": duration_sec,
            "boundary_f1": {
                "0.5s": boundary_metrics(boundaries, [], tolerance_sec=0.5),
                "1.0s": boundary_metrics(boundaries, [], tolerance_sec=1.0),
            },
            "segment_f1_at_iou": {
                str(threshold): segmental_metrics(
                    predicted_intervals,
                    gold_intervals,
                    iou_threshold=threshold,
                )
                for threshold in (0.5, 0.75)
            },
            "label_edit": normalized_edit_score(
                [str(segment.get("instruction", "")) for segment in segments],
                [gold_title],
            ),
        },
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row["metrics"][key])
        for row in rows
        if row.get("status") == "ok" and key in row.get("metrics", {})
    ]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest) if args.manifest else None
    gold_by_id = {}
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("samples", []):
            sample_id = str(row.get("id", row.get("sample_id", "")))
            if sample_id:
                gold_by_id[sample_id] = action_title(sample_id)

    per_sample = []
    for row in payload.get("results", []):
        sample_id = str(row.get("sample_id", ""))
        gold_title = gold_by_id.get(sample_id, action_title(sample_id))
        per_sample.append(evaluate_row(row, gold_title=gold_title))

    ok_rows = [row for row in per_sample if row["status"] == "ok"]
    predictions = [row["prediction_text"] for row in ok_rows if row["prediction_text"]]
    summary = {
        "action_name_token_f1": _mean(ok_rows, "action_name_token_f1"),
        "action_rouge_l": _mean(ok_rows, "action_rouge_l"),
        "semantic_similarity": _mean(ok_rows, "semantic_similarity"),
        "keyword_hit": _mean(ok_rows, "keyword_hit"),
        "loose_title_match": _mean(ok_rows, "loose_title_match"),
        "strict_title_match": _mean(ok_rows, "strict_title_match"),
        "segment_count_accuracy": _mean(ok_rows, "segment_count_accuracy"),
        "composite": _mean(ok_rows, "composite"),
        "pipeline_success_rate": len(ok_rows) / max(len(per_sample), 1),
        "distinct_1": distinct_n(predictions, n=1),
        "distinct_2": distinct_n(predictions, n=2),
        "self_bleu_2": self_bleu_overlap(predictions, n=2),
    }
    summary["mean_segment_f1_iou_0.5"] = (
        sum(
            row["temporal"]["segment_f1_at_iou"]["0.5"]["f1"]
            for row in ok_rows
        )
        / len(ok_rows)
        if ok_rows
        else 0.0
    )

    output = (
        Path(args.output)
        if args.output
        else prediction_path.with_name(f"{prediction_path.stem}_metrics.json")
    )
    report = {
        "schema_version": "video2tasks-run-metrics/v1",
        "evaluation_kind": "weak_folder_name_single_segment",
        "formal": False,
        "generated_at": utc_now_iso(),
        "code_revision": git_revision(ROOT),
        "input": {
            "prediction": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
            "manifest": str(manifest_path) if manifest_path else None,
            "manifest_sha256": (
                file_sha256(manifest_path) if manifest_path and manifest_path.exists() else None
            ),
            "ground_truth_policy": (
                "motion_folder_name_as_one_full_video_label; "
                "underscores replaced with spaces"
            ),
            "model": payload.get("model"),
            "view_mode": payload.get("input", {}).get("view_mode"),
        },
        "counts": {
            "total": len(per_sample),
            "ok": len(ok_rows),
            "failed": len(per_sample) - len(ok_rows),
        },
        "summary": summary,
        "per_sample": per_sample,
        "limitations": [
            "folder names are weak gold, not human adjudication",
            "generic Video2Tasks prompt targets object manipulation, not human motion",
            "no temporal boundary gold; single-segment IoU only checks full-clip coverage",
            "semantic similarity is bag-of-words cosine, not an embedding judge",
        ],
    }
    write_json(output, report)
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
