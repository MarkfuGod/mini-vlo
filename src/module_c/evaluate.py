from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluation.metrics import (
    binary_classification_metrics,
    calibration_metrics,
    coverage_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refinement output.")
    parser.add_argument("--input", required=True, help="Refinement output JSONL")
    parser.add_argument("--output", default="", help="Optional JSON metrics output")
    args = parser.parse_args()

    decisions = Counter()
    with_label = 0
    correct = 0
    total = 0
    confusion = Counter()
    rows = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
            total += 1
            pred = obj["decision"]
            decisions[pred] += 1
            gt = obj.get("aux", {}).get("label")
            if gt is not None:
                with_label += 1
                confusion[(gt, pred)] += 1
                if gt == pred:
                    correct += 1

    print(f"Total samples: {total}")
    print("Decision distribution:")
    for key in ["keep", "drop"]:
        print(f"  {key}: {decisions[key]}")
    extras = sorted(key for key in decisions.keys() if key not in {"keep", "drop"})
    for key in extras:
        print(f"  {key}: {decisions[key]}")

    if with_label > 0:
        accuracy = correct / with_label
        print(f"Labeled samples: {with_label}")
        print(f"Decision accuracy: {accuracy:.4f}")
        print("Confusion (gt -> pred):")
        for (gt, pred), count in confusion.items():
            print(f"  {gt} -> {pred}: {count}")
    else:
        print("No labels found; only distribution is reported.")

    eligible = []
    excluded = Counter()
    for obj in rows:
        reasons = set(obj.get("reason_codes", []))
        semantic = obj.get("aux", {}).get("semantic", {})
        verifier = str(semantic.get("verifier", ""))
        motion = obj.get("aux", {}).get("motion", {})
        provenance = obj.get("aux", {}).get("provenance", {})
        label = obj.get("aux", {}).get("label")
        if label not in {"keep", "drop"}:
            excluded["missing_gold_label"] += 1
            continue
        if verifier.endswith(("_failed", "_fallback")) or any(
            "semantic_verifier_failed" in reason for reason in reasons
        ):
            excluded["semantic_api_failure"] += 1
            continue
        if motion.get("is_dummy") or "dummy_motion_forbidden" in reasons:
            excluded["dummy_motion"] += 1
            continue
        if provenance.get("annotation_status") not in {None, "adjudicated"}:
            excluded["non_adjudicated"] += 1
            continue
        eligible.append(obj)

    payload = {
        "total": len(rows),
        "eligible_formal": len(eligible),
        "excluded": dict(excluded),
        "decision_distribution": dict(decisions),
    }
    if eligible:
        truth = [obj["aux"]["label"] for obj in eligible]
        prediction = [obj["decision"] for obj in eligible]
        scores = [
            min(
                float(obj.get("motion_quality_score") or 0.0),
                float(obj.get("semantic_confidence") or 0.0),
            )
            for obj in eligible
        ]
        correctness = [
            expected == actual for expected, actual in zip(truth, prediction)
        ]
        payload["classification"] = binary_classification_metrics(
            truth,
            prediction,
            positive_label="keep",
            scores=scores,
        )
        payload["calibration"] = calibration_metrics(correctness, scores)
        payload["coverage_accuracy"] = coverage_accuracy(correctness, scores)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

