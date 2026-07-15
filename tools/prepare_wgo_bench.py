#!/usr/bin/env python3
"""Download a small, human-annotated WGO-Bench subset via the HF rows API."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime_utils import file_sha256, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="macrodata/WGO-Bench")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--offset",
        type=int,
        default=25,
        help="25 starts the DROID robot-camera portion of WGO-Bench.",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="data/wgo_bench/subset",
    )
    return parser.parse_args()


def _decode_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _fetch_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "offset": args.offset,
            "length": args.limit,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-vlo-wgo-preparer/1"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    rows = payload.get("rows", [])
    if len(rows) != args.limit:
        raise RuntimeError(
            f"Requested {args.limit} rows at offset {args.offset}, got {len(rows)}"
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for entry in _fetch_rows(args):
        row = entry["row"]
        sample_id = str(row["id"])
        encoded = _decode_json_string(row["video"])
        if not isinstance(encoded, str):
            raise ValueError(f"Unexpected video encoding for {sample_id}")
        video_bytes = base64.b64decode(encoded)
        if b"ftyp" not in video_bytes[:64]:
            raise ValueError(f"Decoded payload is not an MP4 for {sample_id}")
        video_path = video_dir / f"{sample_id}.mp4"
        video_path.write_bytes(video_bytes)

        segments_raw = _decode_json_string(row["segments"])
        if not isinstance(segments_raw, list) or not segments_raw:
            raise ValueError(f"No gold segments for {sample_id}")
        segments = [
            {
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "label": str(segment["subtask"]),
            }
            for segment in segments_raw
        ]
        boundaries = [segment["start_sec"] for segment in segments[1:]]
        metadata = _decode_json_string(row.get("metadata", "{}"))
        samples.append(
            {
                "id": sample_id,
                "video": str(video_path.resolve()),
                "instruction": str(row["instruction"]),
                "target_object": "",
                "actions": [segment["label"] for segment in segments],
                "boundaries_sec": boundaries,
                "segments": segments,
                "annotation_status": "benchmark_gold",
                "gold_source": args.dataset,
                "source_row_index": int(entry["row_idx"]),
                "source_metadata": metadata,
                "video_sha256": file_sha256(video_path),
            }
        )
        print(
            f"Prepared {sample_id}: {len(video_bytes) / 1_000_000:.1f} MB, "
            f"{len(segments)} segments"
        )

    manifest = write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "mini-vlo-wgo-bench-subset/v1",
            "generated_at": utc_now_iso(),
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "offset": args.offset,
            "limit": args.limit,
            "license": "CC-BY-NC-SA-4.0",
            "samples": samples,
        },
    )
    print(f"WGO-Bench subset manifest: {manifest}")


if __name__ == "__main__":
    main()
