#!/usr/bin/env python3
"""Run the first two Semantic-Motion project streams.

Streams implemented:
1. Perception: macro-intent and micro-instruction labeling with an existing VLM.
2. Augmentation: instruction rewriting from the structured perception labels.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.scenario import Prediction, load_scenarios
from src.semantic_motion import (
    LLMInstructionRewriter,
    SemanticMotionPipeline,
    SourceInstructionRewriter,
    TemplateInstructionRewriter,
    VLMRecognitionModel,
)


ROOT = Path(__file__).parent
BENCHMARK_PATH = ROOT / "benchmark" / "scenarios.json"
RESULTS_DIR = ROOT / "results"


def parse_args():
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
        help="Recognition model name, e.g. qwen-vl-plus, qwen-vl-max",
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
    parser.add_argument(
        "--rewriter",
        choices=["llm", "template", "none"],
        default="llm",
    )
    parser.add_argument("--rewrite-model", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save Semantic-Motion JSON records",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")

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
    pipeline = SemanticMotionPipeline(recognizer=recognizer, rewriter=rewriter)
    print(
        "Semantic-Motion ready "
        f"model={recognizer.model} base_url={recognizer.base_url}"
    )

    records = []
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n[{idx}/{len(scenarios)}] {scenario.id}: {scenario.instruction}")
        image_path = ROOT / scenario.image_path
        if not image_path.exists():
            print(f"  WARNING: missing image at {image_path}, skipping")
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
            print(f"  ERROR: {exc}")
            annotation = pipeline.perception.from_prediction(
                scenario,
                image_path,
                Prediction(raw_text=f"ERROR: {exc}"),
            )
            records.append(
                {
                    "annotation": annotation.model_dump(),
                    "augmented_instructions": [],
                }
            )
            continue

        print(
            "  labeled "
            f"{len(record.annotation.micro_instructions)} micro steps, "
            f"generated {len(record.augmented_instructions)} variants "
            f"in {elapsed:.1f}s"
        )
        records.append(record.model_dump())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"semantic_motion_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "streams": ["perception", "augmentation"],
                "recognition_model": recognizer.model,
                "rewriter": args.rewriter,
                "rewrite_model": args.rewrite_model or args.model,
                "num_records": len(records),
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSemantic-Motion records saved to {out_path}")


if __name__ == "__main__":
    main()
