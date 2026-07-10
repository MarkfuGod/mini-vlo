#!/usr/bin/env python3
"""Run the video-to-task model on prepared LIBERO Goal samples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.semantic_motion import (
    SourceInstructionRewriter,
    VLMRecognitionModel,
    VideoTaskPipeline,
    load_view_bundle,
)


STOPWORDS = {"a", "an", "the", "to", "of"}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def token_f1(prediction: str, ground_truth: str) -> float:
    pred = _tokens(prediction)
    truth = _tokens(ground_truth)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    overlap = len(pred & truth)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)


def _prediction_text(record_dict: dict[str, Any]) -> str:
    segments = record_dict.get("task_segments", [])
    return " ; ".join(
        str(segment.get("task_instruction", "")).strip()
        for segment in segments
        if str(segment.get("task_instruction", "")).strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen video-to-task labels on prepared LIBERO Goal demos."
    )
    parser.add_argument(
        "--manifest",
        default="data/libero_goal/processed/manifest.json",
    )
    parser.add_argument(
        "--views",
        choices=["fixed", "ego", "fused", "all"],
        default="fixed",
        help="'all' runs the Fixed/Ego/Fused ablation",
    )
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/libero_goal_qwen3_vl_flash.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest["samples"])
    if args.limit > 0:
        samples = samples[: args.limit]

    selected_views = (
        ["fixed", "ego", "fused"] if args.views == "all" else [args.views]
    )
    recognizer = VLMRecognitionModel()
    pipeline = VideoTaskPipeline(
        recognizer=recognizer,
        rewriter=SourceInstructionRewriter(),
    )
    results: list[dict[str, Any]] = []

    for sample in samples:
        ground_truth = str(sample["ground_truth_instruction"])
        for view_name in selected_views:
            bundle = load_view_bundle(args.manifest, sample["sample_id"])
            work_dir = (
                Path(".semantic_motion_work")
                / "libero_goal"
                / sample["sample_id"]
                / view_name
            )
            record = pipeline.run_view_bundle(
                bundle=bundle,
                work_dir=work_dir,
                source_instruction="",
                view_mode=view_name,
                macro_frames=max(1, args.max_frames),
                num_variants=max(1, args.variants),
            )
            record_dict = record.model_dump()
            prediction = _prediction_text(record_dict)
            result = {
                "sample_id": sample["sample_id"],
                "view": view_name,
                "video_paths": bundle.video_paths(),
                "ground_truth_instruction": ground_truth,
                "ground_truth_source": sample["ground_truth_source"],
                "ground_truth_scope": sample["ground_truth_scope"],
                "prediction": prediction,
                "instruction_token_f1": token_f1(prediction, ground_truth),
                "record": record_dict,
            }
            results.append(result)
            print(
                f"{sample['sample_id']} [{view_name}] "
                f"F1={result['instruction_token_f1']:.3f} "
                f"prediction={prediction!r}",
                flush=True,
            )

    scores = [float(item["instruction_token_f1"]) for item in results]
    payload = {
        "model": recognizer.model,
        "evaluation": "weak task-title instruction token F1",
        "warning": (
            "LIBERO titles are task-level labels only. These scores do not measure "
            "temporal boundaries, micro-actions, contact, or refinement correctness."
        ),
        "sample_view_count": len(results),
        "mean_instruction_token_f1": sum(scores) / len(scores) if scores else 0.0,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved evaluation: {output_path}", flush=True)


if __name__ == "__main__":
    main()
