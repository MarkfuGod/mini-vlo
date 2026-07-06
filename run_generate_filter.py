#!/usr/bin/env python3
"""Run Mini-VLO generation followed by Module C refinement."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.module_c.prepare_samples import (
    PrepareSamplesOptions,
    convert_perception_objects,
    write_samples,
)
from src.module_c.refinement import (
    load_config,
    load_samples,
    refine_samples,
    save_results,
    save_results_pretty,
)
from src.semantic_motion import VLMRecognitionModel, VideoTaskPipeline


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
WORK_DIR = ROOT / ".semantic_motion_work"
DEFAULT_CONFIG = ROOT / "configs" / "module_c_default.yaml"


def _load_motion_plugin(spec: str):
    if ":" not in spec:
        raise ValueError("--motion-plugin format must be 'python.module:function_name'")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Motion plugin function not found or not callable: {spec}")
    return func


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video task labels and filter them with Module C."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to an input video")
    source.add_argument(
        "--frame-dir",
        help="Directory of pre-extracted frames, sorted by filename",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="Optional coarse prompt/hint for the video task",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=12,
        help="Uniformly sampled frames for --video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame rate used to timestamp --frame-dir inputs",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Augmented instruction variants per detected segment",
    )
    parser.add_argument("--api-key", default=None, help="VLM API key")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for the generation model",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Generation model name, e.g. qwen-vl-plus, qwen-vl-max",
    )
    parser.add_argument(
        "--refine-config",
        default=str(DEFAULT_CONFIG),
        help="Module C refinement YAML config",
    )
    parser.add_argument(
        "--semantic-verifier",
        default=None,
        choices=["mock", "qwen3-vl-plus"],
        help="Optional override for semantic.verifier in the refinement config",
    )
    parser.add_argument(
        "--sample-level",
        choices=["segment", "video"],
        default="segment",
        help="Filter one sample per segment or one aggregated sample per video",
    )
    parser.add_argument(
        "--text-source",
        choices=["task_instruction", "augmented_first"],
        default="task_instruction",
        help="Which segment text field to filter",
    )
    parser.add_argument(
        "--motion-dir",
        default="",
        help="Optional motion directory. Expect <video_stem>.json.",
    )
    parser.add_argument(
        "--libero-traj-file",
        default="",
        help="Optional LIBERO demo_0_traj.json path",
    )
    parser.add_argument(
        "--allow-dummy-motion",
        action="store_true",
        help="Allow synthetic placeholder motion for flow debugging",
    )
    parser.add_argument(
        "--allow-missing-motion",
        action="store_true",
        help="Allow semantic-only filtering when motion is missing",
    )
    parser.add_argument(
        "--motion-plugin",
        default="",
        help="Optional plugin in 'python.module:function_name' format",
    )
    parser.add_argument(
        "--perception-output",
        default="",
        help="Optional path for generated VideoTaskRecord JSON",
    )
    parser.add_argument(
        "--samples-output",
        default="",
        help="Optional path for generated Module C samples JSONL",
    )
    parser.add_argument(
        "--samples-pretty-output",
        default="",
        help="Optional path for pretty samples JSON",
    )
    parser.add_argument(
        "--refined-output",
        default="",
        help="Optional path for refinement output JSONL",
    )
    parser.add_argument(
        "--refined-pretty-output",
        default="",
        help="Optional path for pretty refinement JSON",
    )
    return parser.parse_args()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recognizer = VLMRecognitionModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.vlm_model,
    )
    pipeline = VideoTaskPipeline(recognizer=recognizer)
    print(
        "Generate-filter ready "
        f"model={recognizer.model} base_url={recognizer.base_url}"
    )

    if args.video:
        source_path = Path(args.video)
        record = pipeline.run_video(
            source_path,
            work_dir=WORK_DIR / source_path.stem,
            source_instruction=args.instruction,
            max_frames=args.max_frames,
            num_variants=args.variants,
        )
    else:
        source_path = Path(args.frame_dir)
        record = pipeline.run_frame_dir(
            source_path,
            source_instruction=args.instruction,
            fps=args.fps,
            num_variants=args.variants,
        )

    perception_path = (
        Path(args.perception_output)
        if args.perception_output
        else RESULTS_DIR / f"video_task_{timestamp}.json"
    )
    record_dict = record.model_dump()
    _write_json(perception_path, record_dict)
    print(f"Generated perception output: {perception_path}")

    samples_path = (
        Path(args.samples_output)
        if args.samples_output
        else RESULTS_DIR / f"module_c_samples_{timestamp}.jsonl"
    )
    samples_pretty_path = (
        Path(args.samples_pretty_output)
        if args.samples_pretty_output
        else RESULTS_DIR / f"module_c_samples_{timestamp}.pretty.json"
    )
    sample_options = PrepareSamplesOptions(
        text_source=args.text_source,
        sample_level=args.sample_level,
        motion_dir=args.motion_dir or None,
        libero_traj_file=args.libero_traj_file or None,
        allow_dummy_motion=args.allow_dummy_motion,
        allow_missing_motion=args.allow_missing_motion,
        motion_plugin=_load_motion_plugin(args.motion_plugin)
        if args.motion_plugin
        else None,
    )
    prepare_result = convert_perception_objects([record_dict], sample_options)
    write_samples(
        prepare_result.samples,
        output_path=samples_path,
        pretty_output_path=samples_pretty_path,
    )
    print(
        f"Prepared {prepare_result.written} sample(s), "
        f"skipped {prepare_result.skipped}: {samples_path}"
    )

    cfg = load_config(args.refine_config)
    if args.semantic_verifier:
        cfg.semantic_cfg.verifier = args.semantic_verifier
    samples = load_samples(samples_path)
    results = refine_samples(samples, cfg)

    refined_path = (
        Path(args.refined_output)
        if args.refined_output
        else RESULTS_DIR / f"refined_{timestamp}.jsonl"
    )
    refined_pretty_path = (
        Path(args.refined_pretty_output)
        if args.refined_pretty_output
        else RESULTS_DIR / f"refined_{timestamp}.pretty.json"
    )
    save_results(results, refined_path)
    save_results_pretty(results, refined_pretty_path)
    print(f"Refined {len(results)} sample(s): {refined_path}")
    print(f"Pretty refinement output: {refined_pretty_path}")

    keep_count = sum(1 for result in results if result.decision == "keep")
    drop_count = sum(1 for result in results if result.decision == "drop")
    print(f"Decisions: keep={keep_count} drop={drop_count}")


if __name__ == "__main__":
    main()

