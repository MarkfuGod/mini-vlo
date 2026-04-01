#!/usr/bin/env python3
"""
Mini-VLO: Robot Task Understanding Evaluator
=============================================
Evaluates a VLM's ability to understand robot task scenes and instructions,
inspired by Being-H's Vision-Language-Action pipeline.

Usage
-----
  # 1. Generate benchmark (once)
  python generate_benchmark.py

  # 2. Run evaluation
  export DASHSCOPE_API_KEY="sk-..."        # or OPENAI_API_KEY
  python run_eval.py

  # Optional overrides
  python run_eval.py --model qwen-vl-max --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.evaluator import EvalReport, evaluate_single
from src.scenario import Prediction, load_scenarios
from src.vlm_engine import VLMEngine


BENCHMARK_PATH = Path(__file__).parent / "benchmark" / "scenarios.json"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Mini-VLO evaluation runner")
    p.add_argument("--scenarios", default=str(BENCHMARK_PATH),
                   help="Path to scenarios.json")
    p.add_argument("--api-key", default=None,
                   help="API key (overrides env var)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL")
    p.add_argument("--model", default=None,
                   help="Model name, e.g. qwen-vl-plus, qwen-vl-max")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate first N scenarios (for quick testing)")
    p.add_argument("--output", default=None,
                   help="Path to save JSON results")
    return p.parse_args()


def main():
    args = parse_args()

    # ── load scenarios ────────────────────────────────────────────────
    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")

    # ── init VLM engine ───────────────────────────────────────────────
    engine = VLMEngine(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    print(f"VLM engine ready  model={engine.model}  base_url={engine.base_url}")

    # ── run evaluation ────────────────────────────────────────────────
    report = EvalReport()
    predictions_log: list[dict] = []

    for i, scenario in enumerate(scenarios):
        tag = f"[{i + 1}/{len(scenarios)}] {scenario.id}"
        print(f"\n{tag}  instruction: {scenario.instruction}")

        image_path = Path(__file__).parent / scenario.image_path
        if not image_path.exists():
            print(f"  WARNING: image not found at {image_path}, skipping")
            continue

        try:
            t0 = time.time()
            pred = engine.analyze(image_path, scenario.instruction)
            elapsed = time.time() - t0
            print(f"  VLM responded in {elapsed:.1f}s")
        except Exception as exc:
            print(f"  ERROR calling VLM: {exc}")
            pred = Prediction(raw_text=f"ERROR: {exc}")

        result = evaluate_single(scenario.id, pred, scenario.ground_truth)
        report.results.append(result)

        print(f"  obj_f1={result.object_f1:.2f}  task_acc={result.task_accuracy:.0f}"
              f"  rouge={result.action_rouge_l:.2f}  sem_sim={result.semantic_similarity:.2f}"
              f"  spatial={result.spatial_accuracy:.2f}  composite={result.composite:.2f}")

        predictions_log.append({
            "scenario_id": scenario.id,
            "instruction": scenario.instruction,
            "prediction": pred.model_dump(),
            "metrics": {
                "object_f1": result.object_f1,
                "task_accuracy": result.task_accuracy,
                "action_rouge_l": result.action_rouge_l,
                "semantic_similarity": result.semantic_similarity,
                "spatial_accuracy": result.spatial_accuracy,
                "composite": result.composite,
            },
        })

    # ── print summary ─────────────────────────────────────────────────
    print(report.summary_table())

    # ── save results ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": engine.model,
            "num_scenarios": len(report.results),
            "summary": {
                "object_f1": report.mean_object_f1,
                "task_accuracy": report.mean_task_accuracy,
                "action_rouge_l": report.mean_action_rouge_l,
                "semantic_similarity": report.mean_semantic_similarity,
                "spatial_accuracy": report.mean_spatial_accuracy,
                "composite": report.mean_composite,
            },
            "per_scenario": predictions_log,
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
