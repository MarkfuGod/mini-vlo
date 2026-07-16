#!/usr/bin/env python3
"""Download a small, human-annotated WGO-Bench subset via the HF rows API."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
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
        "--batch-size",
        type=int,
        default=5,
        help="Rows fetched per request; keeps full-dataset downloads memory bounded.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checkpointed rows and already-downloaded videos.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for transient Hugging Face rows API failures.",
    )
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


def _fetch_rows(
    args: argparse.Namespace,
    *,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-vlo-wgo-preparer/1"},
    )
    for attempt in range(args.max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
            break
        except (TimeoutError, urllib.error.URLError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or (
                exc.code == 429 or 500 <= exc.code < 600
            )
            if not retryable or attempt >= args.max_retries:
                raise
            delay = min(30.0, 2.0**attempt)
            print(
                f"Transient rows API error at offset {offset}: {exc}; "
                f"retrying in {delay:.0f}s "
                f"({attempt + 1}/{args.max_retries})",
                flush=True,
            )
            time.sleep(delay)
    rows = payload.get("rows", [])
    if len(rows) != length:
        raise RuntimeError(
            f"Requested {length} rows at offset {offset}, got {len(rows)}"
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    samples: list[dict[str, Any]] = []
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("dataset") != args.dataset
            or previous.get("config") != args.config
            or previous.get("split") != args.split
            or int(previous.get("offset", -1)) != args.offset
            or int(previous.get("limit", -1)) != args.limit
        ):
            raise ValueError("Existing manifest does not match requested dataset slice")
        samples = [
            dict(sample)
            for sample in previous.get("samples", [])
            if isinstance(sample, dict)
        ]
        if len(samples) > args.limit:
            raise ValueError("Existing manifest contains more rows than --limit")
        for sample in samples:
            if not Path(str(sample.get("video", ""))).is_file():
                raise FileNotFoundError(
                    f"Checkpointed video is missing: {sample.get('video')}"
                )
        print(f"Resuming with {len(samples)}/{args.limit} checkpointed rows")

    def save_manifest() -> Path:
        return write_json(
            manifest_path,
            {
                "schema_version": "mini-vlo-wgo-bench-subset/v1",
                "generated_at": utc_now_iso(),
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "offset": args.offset,
                "limit": args.limit,
                "prepared": len(samples),
                "complete": len(samples) == args.limit,
                "license": "CC-BY-NC-SA-4.0",
                "samples": samples,
            },
        )

    stop = args.offset + args.limit
    start = args.offset + len(samples)
    for batch_offset in range(start, stop, args.batch_size):
        batch_length = min(args.batch_size, stop - batch_offset)
        for entry in _fetch_rows(
            args,
            offset=batch_offset,
            length=batch_length,
        ):
            row = entry["row"]
            sample_id = str(row["id"])
            encoded = _decode_json_string(row["video"])
            if not isinstance(encoded, str):
                raise ValueError(f"Unexpected video encoding for {sample_id}")
            video_path = video_dir / f"{sample_id}.mp4"
            reused = args.resume and video_path.is_file()
            if reused:
                with video_path.open("rb") as handle:
                    if b"ftyp" not in handle.read(64):
                        raise ValueError(f"Existing payload is not an MP4: {sample_id}")
                video_size = video_path.stat().st_size
            else:
                video_bytes = base64.b64decode(encoded)
                if b"ftyp" not in video_bytes[:64]:
                    raise ValueError(f"Decoded payload is not an MP4 for {sample_id}")
                video_path.write_bytes(video_bytes)
                video_size = len(video_bytes)

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
                f"{'Reused' if reused else 'Prepared'} "
                f"{len(samples)}/{args.limit} {sample_id}: "
                f"{video_size / 1_000_000:.1f} MB, "
                f"{len(segments)} segments",
                flush=True,
            )
        save_manifest()

    manifest = save_manifest()
    print(f"WGO-Bench subset manifest: {manifest}")


if __name__ == "__main__":
    main()
