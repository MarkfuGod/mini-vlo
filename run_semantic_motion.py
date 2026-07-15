#!/usr/bin/env python3
"""Run the first two Semantic-Motion streams on static benchmark scenarios.

Streams implemented:
1. Perception: macro-intent and micro-instruction labeling with a VLM.
2. Augmentation: instruction rewriting from the structured perception labels.

For temporal Fixed/Ego/Fused video input use ``run_video_task.py``.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from src.prompts import SYSTEM_PROMPT
from src.runtime_utils import (
    file_sha256,
    git_revision,
    text_sha256,
    utc_now_iso,
    write_json,
)
from src.scenario import load_scenarios
from src.semantic_motion import (
    SemanticMotionPipeline,
    VLMRecognitionModel,
)
from src.semantic_motion.cli import add_rewriter_arguments, build_rewriter


ROOT = Path(__file__).parent
BENCHMARK_PATH = ROOT / "benchmark" / "scenarios.json"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Semantic-Motion perception + augmentation streams"
    )
    parser.add_argument(
        "--scenarios",
        default=str(BENCHMARK_PATH),
        help="Path to scenarios.json",
    )
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
        help="Recognition model name, e.g. qwen3-vl-flash",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N scenarios",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Number of augmented instruction variants per scenario",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first missing image or provider error.",
    )
    add_rewriter_arguments(parser)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save Semantic-Motion JSON records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")

    recognizer = VLMRecognitionModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    rewriter = build_rewriter(args, perception_model=recognizer.model)
    pipeline = SemanticMotionPipeline(recognizer=recognizer, rewriter=rewriter)
    print(
        "Semantic-Motion ready "
        f"model={recognizer.model} base_url={recognizer.base_url}"
    )

    records: list[dict] = []
    failed = 0
    skipped = 0
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n[{idx}/{len(scenarios)}] {scenario.id}: {scenario.instruction}")
        image_path = ROOT / scenario.image_path
        if not image_path.exists():
            if args.fail_fast:
                raise FileNotFoundError(image_path)
            print(f"  WARNING: missing image at {image_path}, skipping")
            skipped += 1
            records.append(
                {
                    "scenario_id": scenario.id,
                    "status": "skipped",
                    "error": f"missing image: {image_path}",
                }
            )
            continue

        try:
            t0 = time.time()
            record = pipeline.run_one(
                scenario,
                image_root=ROOT,
                num_variants=args.variants,
            )
            elapsed = time.time() - t0
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"  ERROR: {exc}")
            failed += 1
            records.append(
                {
                    "scenario_id": scenario.id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        print(
            "  labeled "
            f"{len(record.annotation.micro_instructions)} micro steps, "
            f"generated {len(record.augmented_instructions)} variants "
            f"in {elapsed:.1f}s"
        )
        record_dict = record.model_dump()
        record_dict["status"] = "ok"
        record_dict["latency_s"] = elapsed
        records.append(record_dict)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"semantic_motion_{timestamp}.json"
    )
    scenario_path = Path(args.scenarios)
    succeeded = sum(record.get("status") == "ok" for record in records)
    write_json(
        out_path,
        {
            "schema_version": "semantic-motion-static/v2",
            "evaluation_kind": "diagnostic_static_perception_augmentation",
            "formal": False,
            "generated_at": utc_now_iso(),
            "code_revision": git_revision(ROOT),
            "streams": ["perception", "augmentation"],
            "recognition_model": recognizer.model,
            "base_url": recognizer.base_url,
            "timeout_s": recognizer.timeout,
            "max_retries": args.max_retries,
            "rewriter": args.rewriter,
            "rewrite_model": args.rewrite_model or recognizer.model,
            "code_level_augmentation_validation": "disabled",
            "prompt_sha256": text_sha256(SYSTEM_PROMPT),
            "input": {
                "scenarios": str(scenario_path),
                "sha256": file_sha256(scenario_path),
                "limit": args.limit,
            },
            "counts": {
                "requested": len(scenarios),
                "succeeded": succeeded,
                "failed": failed,
                "skipped": skipped,
            },
            "num_records": succeeded,
            "records": records,
            "limitations": [
                "single synthetic image per scenario",
                "not the temporal multi-view VideoTaskPipeline",
            ],
        },
    )
    print(f"\nSemantic-Motion records saved to {out_path}")


if __name__ == "__main__":
    main()
