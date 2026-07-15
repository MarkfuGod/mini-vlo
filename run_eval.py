#!/usr/bin/env python3
"""Evaluate a VLM on the diagnostic static-image Mini-VLO benchmark."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluator import EvalReport, evaluate_single
from src.prompts import SYSTEM_PROMPT
from src.runtime_utils import (
    file_sha256,
    git_revision,
    text_sha256,
    utc_now_iso,
    write_json,
)
from src.scenario import load_scenarios
from src.vlm_engine import VLMEngine


BENCHMARK_PATH = Path(__file__).parent / "benchmark" / "scenarios.json"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini-VLO evaluation runner")
    p.add_argument("--scenarios", default=str(BENCHMARK_PATH),
                   help="Path to scenarios.json")
    p.add_argument("--api-key", default=None,
                   help="API key (overrides env var)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL")
    p.add_argument("--model", default=None,
                   help="Model name, e.g. qwen3-vl-flash")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate first N scenarios (for quick testing)")
    p.add_argument(
        "--category",
        action="append",
        default=[],
        help="Only evaluate this category; repeat for multiple categories.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="API request timeout in seconds.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="OpenAI-compatible client retry count.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on a missing image or provider/parse error.",
    )
    p.add_argument("--output", default=None,
                   help="Path to save JSON results")
    return p.parse_args()


def _summary(report: EvalReport) -> dict[str, float]:
    return {
        "object_f1": report.mean_object_f1,
        "task_accuracy": report.mean_task_accuracy,
        "action_rouge_l": report.mean_action_rouge_l,
        "semantic_similarity": report.mean_semantic_similarity,
        "spatial_accuracy": report.mean_spatial_accuracy,
        "composite": report.mean_composite,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).parent

    scenarios = load_scenarios(args.scenarios)
    if args.category:
        selected = set(args.category)
        scenarios = [
            scenario for scenario in scenarios if scenario.category in selected
        ]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")

    engine = VLMEngine(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(
        f"VLM engine ready model={engine.model} base_url={engine.base_url} "
        f"timeout={engine.timeout:.1f}s"
    )

    report = EvalReport()
    predictions_log: list[dict] = []
    failures = 0
    skipped = 0

    for i, scenario in enumerate(scenarios):
        tag = f"[{i + 1}/{len(scenarios)}] {scenario.id}"
        print(f"\n{tag}  instruction: {scenario.instruction}")

        image_path = root / scenario.image_path
        if not image_path.exists():
            message = f"image not found: {image_path}"
            if args.fail_fast:
                raise FileNotFoundError(message)
            print(f"  WARNING: {message}")
            skipped += 1
            predictions_log.append(
                {
                    "scenario_id": scenario.id,
                    "category": scenario.category,
                    "instruction": scenario.instruction,
                    "status": "skipped",
                    "error": message,
                }
            )
            continue

        try:
            t0 = time.time()
            pred = engine.analyze(image_path, scenario.instruction)
            elapsed = time.time() - t0
            if not pred.raw_text.strip():
                raise ValueError("provider returned an empty response")
            print(f"  VLM responded in {elapsed:.1f}s")
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"  ERROR calling VLM: {exc}")
            failures += 1
            predictions_log.append(
                {
                    "scenario_id": scenario.id,
                    "category": scenario.category,
                    "instruction": scenario.instruction,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        result = evaluate_single(scenario.id, pred, scenario.ground_truth)
        report.results.append(result)

        print(f"  obj_f1={result.object_f1:.2f}  task_acc={result.task_accuracy:.0f}"
              f"  rouge={result.action_rouge_l:.2f}  sem_sim={result.semantic_similarity:.2f}"
              f"  spatial={result.spatial_accuracy:.2f}  composite={result.composite:.2f}")

        predictions_log.append({
            "scenario_id": scenario.id,
            "category": scenario.category,
            "instruction": scenario.instruction,
            "status": "ok",
            "latency_s": elapsed,
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

    print(report.summary_table())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{ts}.json"
    scenario_path = Path(args.scenarios)
    write_json(
        out_path,
        {
            "schema_version": "mini-vlo-static-eval/v2",
            "evaluation_kind": "diagnostic_static_schematic",
            "formal": False,
            "generated_at": utc_now_iso(),
            "code_revision": git_revision(root),
            "model": engine.model,
            "base_url": engine.base_url,
            "timeout_s": engine.timeout,
            "max_retries": engine.max_retries,
            "input": {
                "scenarios": str(scenario_path),
                "sha256": file_sha256(scenario_path),
                "categories": args.category,
                "limit": args.limit,
            },
            "prompt_sha256": text_sha256(SYSTEM_PROMPT),
            "requested_scenarios": len(scenarios),
            "num_scenarios": len(report.results),
            "counts": {
                "succeeded": len(report.results),
                "failed": failures,
                "skipped": skipped,
            },
            "summary": _summary(report),
            "per_scenario": predictions_log,
            "limitations": [
                "synthetic single-image benchmark",
                "no temporal, multi-view, motion, or policy-success evaluation",
                "provider failures are excluded from metric means and reported separately",
            ],
        },
    )
    print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
