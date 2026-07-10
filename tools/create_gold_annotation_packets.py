#!/usr/bin/env python3
"""Create paired-view annotation packets without pretending they are human gold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.gold import GoldAnnotation, save_gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/libero_goal/processed/manifest.json",
    )
    parser.add_argument("--output-dir", default="data/gold/annotations")
    parser.add_argument("--clip-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for sample in manifest.get("samples", []):
        fps = float(sample["fps"])
        frame_count = int(sample["frame_count"])
        duration = frame_count / fps
        start = 0.0
        clip_index = 0
        while start < duration and len(index) < args.limit:
            end = min(duration, start + args.clip_sec)
            if end - start < 1.0:
                break
            packet_id = f"{sample['sample_id']}_clip_{clip_index:03d}"
            annotation = GoldAnnotation(
                sample_id=packet_id,
                annotation_status="pending_human",
                source_sample_id=str(sample["sample_id"]),
                source_views={
                    str(view): str(path)
                    for view, path in sample["views"].items()
                },
                clip_start_sec=start,
                clip_end_sec=end,
                fps=fps,
                frame_count=max(1, int(round((end - start) * fps))),
                weak_task_title=str(sample.get("ground_truth_instruction", "")),
                notes=(
                    "Candidate annotation packet only. The weak task title must not "
                    "be copied as boundary, micro-action, contact, or keep/drop truth."
                ),
            )
            output_path = output_dir / f"{packet_id}.json"
            save_gold(annotation, output_path)
            index.append(
                {
                    "sample_id": packet_id,
                    "path": str(output_path),
                    "annotation_status": "pending_human",
                }
            )
            clip_index += 1
            start += args.step_sec
        if len(index) >= args.limit:
            break

    index_path = output_dir.parent / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic-motion-gold-index/v1",
                "formal_gold_count": 0,
                "candidate_count": len(index),
                "warning": (
                    "Only adjudicated, independently double-annotated packets count "
                    "as formal gold."
                ),
                "samples": index,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(index)} pending-human packets -> {output_dir}")


if __name__ == "__main__":
    main()
