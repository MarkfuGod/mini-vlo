#!/usr/bin/env python3
"""Video-to-task runner for Semantic-Motion.

Compared with a plain Video2Tasks-style segment labeler, this runner keeps
frame-level semantic evidence, aggregates temporal task segments, and emits
macro intents, micro instructions, and rewritten task variants.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.semantic_motion import (
    LLMInstructionRewriter,
    SourceInstructionRewriter,
    TemplateInstructionRewriter,
    VLMRecognitionModel,
    VideoTaskPipeline,
    load_view_bundle,
)


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
WORK_DIR = ROOT / ".semantic_motion_work"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer manipulation task segments from a video"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to an input video")
    source.add_argument(
        "--frame-dir",
        help="Directory of pre-extracted frames, sorted by filename",
    )
    source.add_argument(
        "--manifest",
        help="ViewBundle or dataset manifest containing synchronized fixed/ego views",
    )
    parser.add_argument("--sample-id", default="", help="Sample id inside --manifest")
    parser.add_argument(
        "--view-mode",
        choices=["fixed", "ego", "fused"],
        default="fused",
        help="Camera ablation or paired-view early fusion for --manifest",
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
        help="Frames sampled in each overlapping macro window",
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
    parser.add_argument(
        "--rewriter",
        choices=["llm", "template", "none"],
        default="llm",
        help="LLM is production mode; template is deterministic debug-only mode",
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Optional independent text-rewrite model (defaults to --model)",
    )
    parser.add_argument("--macro-window-sec", type=float, default=16.0)
    parser.add_argument("--macro-step-sec", type=float, default=8.0)
    parser.add_argument("--micro-window-sec", type=float, default=2.0)
    parser.add_argument("--micro-step-sec", type=float, default=1.0)
    parser.add_argument("--micro-frames", type=int, default=4)
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the existing recognition model",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for the recognition model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Recognition model name, e.g. qwen-vl-plus, qwen-vl-max",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save video-to-task JSON output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = VLMRecognitionModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    if args.rewriter == "llm":
        rewriter = LLMInstructionRewriter(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.rewrite_model or args.model,
        )
    elif args.rewriter == "template":
        rewriter = TemplateInstructionRewriter()
    else:
        rewriter = SourceInstructionRewriter()
    pipeline = VideoTaskPipeline(recognizer=recognizer, rewriter=rewriter)
    print(
        "Video-to-task ready "
        f"model={recognizer.model} base_url={recognizer.base_url}"
    )

    if args.manifest:
        bundle = load_view_bundle(args.manifest, sample_id=args.sample_id or None)
        source_path = Path(args.manifest)
        record = pipeline.run_view_bundle(
            bundle,
            work_dir=WORK_DIR / bundle.sample_id / args.view_mode,
            source_instruction=args.instruction,
            view_mode=args.view_mode,
            num_variants=args.variants,
            macro_window_sec=args.macro_window_sec,
            macro_step_sec=args.macro_step_sec,
            macro_frames=args.max_frames,
            micro_window_sec=args.micro_window_sec,
            micro_step_sec=args.micro_step_sec,
            micro_frames=args.micro_frames,
        )
    elif args.video:
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"video_task_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record.model_dump(), f, indent=2, ensure_ascii=False)

    print(
        f"Processed {len(record.frames) + len(record.multi_view_frames)} evidence frames into "
        f"{len(record.task_segments)} task segment(s)."
    )
    for segment in record.task_segments:
        print(
            f"- {segment.segment_id} "
            f"{segment.start_time_sec:.2f}s-{segment.end_time_sec:.2f}s: "
            f"{segment.task_instruction}"
        )
    print(f"Video-to-task output saved to {output_path}")


if __name__ == "__main__":
    main()
