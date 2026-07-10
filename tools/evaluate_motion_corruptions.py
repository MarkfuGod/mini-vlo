#!/usr/bin/env python3
"""Calibrate motion-quality detection on reproducible synthetic corruptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.corruption import inject_motion_corruption
from src.evaluation.metrics import binary_auroc
from src.module_c.motion_quality import score_motion_quality
from src.module_c.prepare_samples import (
    _normalize_libero_motion,
    _normalize_motion_tracks,
)
from src.module_c.refinement import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument(
        "--config",
        default="configs/module_c_default.yaml",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj = json.loads(Path(args.motion).read_text(encoding="utf-8"))
    tracks = _normalize_libero_motion(obj) or _normalize_motion_tracks(obj)
    if not tracks:
        raise ValueError("Unsupported motion file")
    cfg = load_config(args.config).motion_cfg
    rows = []
    for track_name, (positions, timestamps) in tracks.items():
        for corruption in ("clean", "jitter", "spike", "drop_frame", "time_shift"):
            corrupted = inject_motion_corruption(
                positions,
                timestamps,
                corruption,
                seed=7,
            )
            quality, details, reasons = score_motion_quality(
                corrupted.positions,
                corrupted.timestamps,
                cfg,
            )
            rows.append(
                {
                    "track": track_name,
                    "corruption": corruption,
                    "is_corrupt": corruption != "clean",
                    "quality_score": quality,
                    "anomaly_score": 1.0 - quality,
                    "details": details,
                    "reason_codes": reasons,
                }
            )
    payload = {
        "source": args.motion,
        "corruption_detection_auroc": binary_auroc(
            [row["is_corrupt"] for row in rows],
            [row["anomaly_score"] for row in rows],
        ),
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
