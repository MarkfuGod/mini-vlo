#!/usr/bin/env python3
"""Run generation followed by ungated Module C diagnostic refinement."""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

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
from src.runtime_utils import git_revision, utc_now_iso, write_json
from src.semantic_motion import (
    VLMRecognitionModel,
    VideoTaskPipeline,
    build_view_bundle,
    load_view_bundle,
)
from src.semantic_motion.cli import add_rewriter_arguments, build_rewriter


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


def _parse_motion_tracks(value: str):
    tracks = [item.strip() for item in value.split(",") if item.strip()]
    return tracks or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate video task labels and attach Module C diagnostics. "
            "Quality gates are disabled, so prepared samples are retained."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to an input video")
    source.add_argument(
        "--frame-dir",
        help="Directory of pre-extracted frames, sorted by filename",
    )
    source.add_argument(
        "--manifest",
        help="ViewBundle or dataset manifest with synchronized fixed/ego views",
    )
    parser.add_argument("--sample-id", default="")
    parser.add_argument(
        "--view-mode",
        choices=["fixed", "ego", "fused"],
        default="fused",
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
    add_rewriter_arguments(parser)
    parser.add_argument("--api-key", default=None, help="VLM API key")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for the generation model",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Generation model name, e.g. qwen3-vl-flash, qwen3-vl-plus",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--macro-window-sec", type=float, default=16.0)
    parser.add_argument("--macro-step-sec", type=float, default=8.0)
    parser.add_argument("--micro-window-sec", type=float, default=2.0)
    parser.add_argument("--micro-step-sec", type=float, default=1.0)
    parser.add_argument("--micro-frames", type=int, default=4)
    parser.add_argument(
        "--work-dir",
        default=str(WORK_DIR),
        help="Directory for extracted frame evidence.",
    )
    parser.add_argument(
        "--refine-config",
        default=str(DEFAULT_CONFIG),
        help="Module C refinement YAML config",
    )
    parser.add_argument(
        "--semantic-verifier",
        default=None,
        help="Optional override for semantic.verifier in the refinement config",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Independent semantic judge model; do not reuse generator by default.",
    )
    parser.add_argument(
        "--motion-aggregation",
        choices=["min", "mean"],
        default="",
        help="Optional override for motion_quality.aggregation.",
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
        "--motion-path",
        default="",
        help=(
            "Unified motion JSON file or directory. Files are auto-detected as "
            "LIBERO, Module D, or standard tracks; directories are searched by "
            "video stem."
        ),
    )
    parser.add_argument(
        "--motion-dir",
        default="",
        help="Deprecated alias for a motion directory. Prefer --motion-path.",
    )
    parser.add_argument(
        "--motion-fps",
        type=float,
        default=24.0,
        help="FPS used to timestamp Module D frame_N motion files.",
    )
    parser.add_argument(
        "--motion-tracks",
        default="",
        help="Comma-separated track names to keep, e.g. Root,Hand_R,Hand_L.",
    )
    parser.add_argument(
        "--libero-traj-file",
        default="",
        help="Deprecated alias for a LIBERO demo_0_traj.json path. Prefer --motion-path.",
    )
    parser.add_argument(
        "--debug-dummy-motion",
        "--allow-dummy-motion",
        dest="allow_dummy_motion",
        action="store_true",
        default=False,
        help="Generate an explicit placeholder trajectory marked is_dummy=true.",
    )
    parser.add_argument(
        "--allow-missing-motion",
        action="store_true",
        default=True,
        help="Deprecated no-op: missing motion is retained as a diagnostic.",
    )
    parser.add_argument(
        "--allow-single-view-debug",
        action="store_true",
        help="Disable the paired-view diagnostic requirement.",
    )
    parser.add_argument(
        "--allow-mock-debug",
        action="store_true",
        help="Set the legacy allow_mock_keep diagnostic config flag.",
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


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recognizer = VLMRecognitionModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.vlm_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    rewriter = build_rewriter(args, perception_model=recognizer.model)
    pipeline = VideoTaskPipeline(recognizer=recognizer, rewriter=rewriter)
    print(
        "Generate-filter ready "
        f"model={recognizer.model} base_url={recognizer.base_url}"
    )

    bundle = None
    if args.manifest:
        bundle = load_view_bundle(args.manifest, sample_id=args.sample_id or None)
        source_path = Path(args.manifest)
        record = pipeline.run_view_bundle(
            bundle,
            work_dir=work_root / bundle.sample_id / args.view_mode,
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
        bundle = build_view_bundle(
            sample_id=source_path.stem,
            views={"fixed": source_path},
        )
        record = pipeline.run_view_bundle(
            bundle,
            work_dir=work_root / source_path.stem / "fixed",
            source_instruction=args.instruction,
            view_mode="fixed",
            num_variants=args.variants,
            macro_window_sec=args.macro_window_sec,
            macro_step_sec=args.macro_step_sec,
            macro_frames=args.max_frames,
            micro_window_sec=args.micro_window_sec,
            micro_step_sec=args.micro_step_sec,
            micro_frames=args.micro_frames,
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
    record.metadata.update(
        {
            "entry_point": "run_generate_filter.py",
            "generated_at": utc_now_iso(),
            "code_revision": git_revision(ROOT),
            "base_url": recognizer.base_url,
            "request_timeout_s": recognizer.timeout,
            "max_retries": args.max_retries,
            "code_level_augmentation_validation": "disabled",
            "refinement_policy": "diagnostic_only_quality_gates_disabled",
        }
    )
    record_dict = record.model_dump()
    write_json(perception_path, record_dict)
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
        motion_path=(
            args.motion_path
            or (
                bundle.trajectory.path
                if bundle is not None and bundle.trajectory is not None
                else ""
            )
            or None
        ),
        motion_dir=args.motion_dir or None,
        libero_traj_file=args.libero_traj_file or None,
        allow_dummy_motion=args.allow_dummy_motion,
        allow_missing_motion=args.allow_missing_motion,
        motion_plugin=_load_motion_plugin(args.motion_plugin)
        if args.motion_plugin
        else None,
        motion_fps=args.motion_fps,
        motion_tracks=_parse_motion_tracks(args.motion_tracks),
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
    cfg.semantic_cfg.request_timeout_s = args.timeout
    if args.semantic_verifier:
        cfg.semantic_cfg.verifier = args.semantic_verifier
    if args.judge_model:
        cfg.semantic_cfg.qwen_vl_model = args.judge_model
    if args.motion_aggregation:
        cfg.motion_cfg.aggregation = args.motion_aggregation
    if args.allow_single_view_debug:
        cfg.sync_cfg.require_paired_views = False
    if args.allow_mock_debug:
        cfg.allow_mock_keep = True
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

