from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refinement output.")
    parser.add_argument("--input", required=True, help="Refinement output JSONL")
    args = parser.parse_args()

    decisions = Counter()
    with_label = 0
    correct = 0
    total = 0
    confusion = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
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


if __name__ == "__main__":
    main()

