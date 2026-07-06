from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MotionTuple = tuple[list[list[float]], list[float]]
MotionPlugin = Callable[[dict[str, Any], dict[str, Any], str, str], MotionTuple | None]


@dataclass
class PrepareSamplesOptions:
    text_source: str = "task_instruction"
    sample_level: str = "segment"
    motion_dir: str | Path | None = None
    libero_traj_file: str | Path | None = None
    allow_dummy_motion: bool = False
    allow_missing_motion: bool = False
    motion_plugin: MotionPlugin | None = None


@dataclass
class PrepareSamplesResult:
    written: int
    skipped: int
    samples: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_valid_motion(positions: list[list[float]], timestamps: list[float]) -> bool:
    if len(positions) != len(timestamps):
        return False
    if len(positions) < 4:
        return False
    if any(len(point) != 3 for point in positions):
        return False
    return all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))


def _build_dummy_motion(frame_timestamps: list[float]) -> MotionTuple:
    if len(frame_timestamps) >= 4:
        timestamps = frame_timestamps
    elif len(frame_timestamps) >= 2:
        start = frame_timestamps[0]
        end = frame_timestamps[-1]
        step = (end - start) / 3.0 if end > start else 0.1
        timestamps = [start + i * step for i in range(4)]
    else:
        timestamps = [0.0, 0.1, 0.2, 0.3]
    positions = [[0.02 * i, 0.0, 0.0] for i in range(len(timestamps))]
    return positions, timestamps


def _load_motion_from_file(
    motion_dir: Path,
    video_stem: str,
    segment_id: str,
) -> MotionTuple | None:
    motion_path = motion_dir / f"{video_stem}.json"
    if not motion_path.exists():
        return None

    obj = _read_json(motion_path)
    if "positions" in obj and "timestamps" in obj:
        positions = obj["positions"]
        timestamps = obj["timestamps"]
    elif "segments" in obj and segment_id in obj["segments"]:
        segment = obj["segments"][segment_id]
        positions = segment["positions"]
        timestamps = segment["timestamps"]
    else:
        return None

    if not _is_valid_motion(positions, timestamps):
        return None
    return positions, timestamps


def _load_libero_motion_from_file(
    traj_path: Path,
    segment_obj: dict[str, Any] | None = None,
) -> MotionTuple | None:
    if not traj_path.exists():
        return None

    obj = _read_json(traj_path)
    steps = obj.get("steps", [])
    if not isinstance(steps, list):
        return None

    fps = float(obj.get("fps", 0.0) or 0.0)
    positions: list[list[float]] = []
    timestamps: list[float] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pos = step.get("ee_pos")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        raw_t = step.get("t", idx)
        timestamp = float(raw_t) / fps if fps > 0 else float(raw_t)
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        timestamps.append(timestamp)

    if segment_obj is not None:
        start = segment_obj.get("start_time_sec")
        end = segment_obj.get("end_time_sec")
        if start is not None and end is not None and float(end) > float(start):
            start_time = float(start)
            end_time = float(end)
            filtered = [
                (pos, ts)
                for pos, ts in zip(positions, timestamps)
                if start_time <= ts <= end_time
            ]
            if filtered:
                positions = [pos for pos, _ in filtered]
                timestamps = [ts for _, ts in filtered]

    if not _is_valid_motion(positions, timestamps):
        return None
    return positions, timestamps


def _load_motion_plugin(spec: str) -> MotionPlugin:
    if ":" not in spec:
        raise ValueError("--motion-plugin format must be 'python.module:function_name'")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Motion plugin function not found or not callable: {spec}")
    return func


def _extract_text(segment: dict[str, Any], text_source: str) -> str:
    if text_source == "task_instruction":
        return str(segment.get("task_instruction", "")).strip()
    if text_source == "augmented_first":
        augmented = segment.get("augmented_instructions", [])
        if augmented and isinstance(augmented, list):
            first = augmented[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
    return ""


def _extract_video_text(
    perception_obj: dict[str, Any],
    segments: list[dict[str, Any]],
    text_source: str,
) -> str:
    source_instruction = str(perception_obj.get("source_instruction", "")).strip()
    if source_instruction:
        return source_instruction

    pieces: list[str] = []
    seen = set()
    for segment in sorted(
        segments,
        key=lambda item: float(item.get("start_time_sec", 0.0) or 0.0),
    ):
        text = _extract_text(segment, text_source)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            start = segment.get("start_time_sec")
            end = segment.get("end_time_sec")
            if start is not None and end is not None:
                pieces.append(f"{float(start):.2f}s-{float(end):.2f}s: {text}")
            else:
                pieces.append(text)

    if not pieces:
        return ""
    return "Overall video task sequence:\n" + "\n".join(pieces)


def _extract_segment_texts(
    segments: list[dict[str, Any]],
    text_source: str,
) -> list[str]:
    texts: list[str] = []
    for segment in sorted(
        segments,
        key=lambda item: float(item.get("start_time_sec", 0.0) or 0.0),
    ):
        text = _extract_text(segment, text_source)
        if text:
            texts.append(text)
    return texts


def _frame_timestamps(perception_obj: dict[str, Any]) -> list[float]:
    frame_ts: list[float] = []
    for item in perception_obj.get("frames", []):
        try:
            frame_ts.append(float(item["frame"]["timestamp_sec"]))
        except (KeyError, TypeError, ValueError):
            pass
    return sorted(set(frame_ts))


def _resolve_motion(
    perception_obj: dict[str, Any],
    segment_obj: dict[str, Any],
    video_stem: str,
    segment_id: str,
    frame_ts: list[float],
    options: PrepareSamplesOptions,
) -> MotionTuple | None:
    motion_dir = Path(options.motion_dir) if options.motion_dir else None
    libero_traj_file = (
        Path(options.libero_traj_file) if options.libero_traj_file else None
    )

    motion: MotionTuple | None = None
    if options.motion_plugin is not None:
        motion = options.motion_plugin(
            perception_obj,
            segment_obj,
            video_stem,
            segment_id,
        )
        if motion is not None and not _is_valid_motion(motion[0], motion[1]):
            motion = None
    if motion is None and libero_traj_file is not None:
        motion = _load_libero_motion_from_file(libero_traj_file, segment_obj)
    if motion is None and motion_dir is not None:
        motion = _load_motion_from_file(motion_dir, video_stem, segment_id)
    if motion is None and options.allow_dummy_motion:
        motion = _build_dummy_motion(frame_ts)
    return motion


def convert_perception_objects(
    perception_objects: list[dict[str, Any]],
    options: PrepareSamplesOptions,
) -> PrepareSamplesResult:
    written = 0
    skipped = 0
    samples: list[dict[str, Any]] = []

    for obj in perception_objects:
        video_path = str(obj.get("video_path", "")).strip()
        if not video_path:
            skipped += 1
            continue
        video_stem = Path(video_path).stem

        segments = obj.get("task_segments", [])
        if not isinstance(segments, list) or not segments:
            skipped += 1
            continue

        frame_ts = _frame_timestamps(obj)

        if options.sample_level == "video":
            text = _extract_video_text(obj, segments, options.text_source)
            if not text:
                skipped += 1
                continue

            video_segment = {
                "segment_id": "video",
                "start_time_sec": frame_ts[0] if frame_ts else 0.0,
                "end_time_sec": frame_ts[-1] if frame_ts else 0.0,
                "task_segments": segments,
            }
            motion = _resolve_motion(
                obj,
                video_segment,
                video_stem,
                "video",
                frame_ts,
                options,
            )
            if motion is None and not options.allow_missing_motion:
                skipped += 1
                continue

            sample = {
                "sample_id": f"{video_stem}_video",
                "video_path": video_path,
                "text": text,
                "source_segment_texts": _extract_segment_texts(
                    segments,
                    options.text_source,
                ),
                "source_segments": [
                    str(segment.get("segment_id", "segment")) for segment in segments
                ],
            }
            if motion is not None:
                positions, timestamps = motion
                sample["motion"] = {
                    "positions": positions,
                    "timestamps": timestamps,
                }
                if options.libero_traj_file is not None:
                    sample["motion_source"] = str(options.libero_traj_file)
            samples.append(sample)
            written += 1
            continue

        for segment in segments:
            segment_id = str(segment.get("segment_id", "segment"))
            text = _extract_text(segment, options.text_source)
            if not text:
                skipped += 1
                continue

            motion = _resolve_motion(
                obj,
                segment,
                video_stem,
                segment_id,
                frame_ts,
                options,
            )
            if motion is None and not options.allow_missing_motion:
                skipped += 1
                continue

            sample = {
                "sample_id": f"{video_stem}_{segment_id}",
                "video_path": video_path,
                "text": text,
            }
            if motion is not None:
                positions, timestamps = motion
                sample["motion"] = {
                    "positions": positions,
                    "timestamps": timestamps,
                }
                if options.libero_traj_file is not None:
                    sample["motion_source"] = str(options.libero_traj_file)
            samples.append(sample)
            written += 1

    return PrepareSamplesResult(written=written, skipped=skipped, samples=samples)


def write_samples(
    samples: list[dict[str, Any]],
    output_path: str | Path,
    pretty_output_path: str | Path | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if pretty_output_path is not None:
        pretty_path = Path(pretty_output_path)
        pretty_path.parent.mkdir(parents=True, exist_ok=True)
        with pretty_path.open("w", encoding="utf-8") as pretty_out:
            json.dump(samples, pretty_out, indent=2, ensure_ascii=False)
            pretty_out.write("\n")


def convert_perception_files(
    files: list[str | Path],
    output_path: str | Path,
    options: PrepareSamplesOptions,
    pretty_output_path: str | Path | None = None,
) -> PrepareSamplesResult:
    perception_objects = [_read_json(Path(file_path)) for file_path in files]
    result = convert_perception_objects(perception_objects, options)
    write_samples(result.samples, output_path, pretty_output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mini-VLO perception JSON to Module C samples JSONL."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--perception-dir", help="Directory containing *.json files")
    source.add_argument("--perception-file", help="Single perception JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--pretty-output",
        default="",
        help="Optional pretty JSON path for human-readable inspection",
    )
    parser.add_argument(
        "--text-source",
        choices=["task_instruction", "augmented_first"],
        default="task_instruction",
        help="Which text field to use as sample text",
    )
    parser.add_argument(
        "--sample-level",
        choices=["segment", "video"],
        default="segment",
        help="Write one sample per segment or one aggregated sample per video",
    )
    parser.add_argument(
        "--motion-dir",
        default="",
        help="Optional motion directory. Expect <video_stem>.json.",
    )
    parser.add_argument(
        "--libero-traj-file",
        default="",
        help="Optional LIBERO demo_0_traj.json path.",
    )
    parser.add_argument(
        "--allow-dummy-motion",
        action="store_true",
        help="Allow synthetic placeholder motion when no real motion is found",
    )
    parser.add_argument(
        "--allow-missing-motion",
        action="store_true",
        help="Allow writing samples without motion field",
    )
    parser.add_argument(
        "--motion-plugin",
        default="",
        help="Plugin in 'python.module:function_name' format.",
    )
    args = parser.parse_args()

    if args.perception_file:
        perception_file = Path(args.perception_file)
        if not perception_file.exists():
            raise FileNotFoundError(f"JSON file not found: {perception_file}")
        files = [perception_file]
    else:
        perception_dir = Path(args.perception_dir)
        files = sorted(perception_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in: {perception_dir}")

    options = PrepareSamplesOptions(
        text_source=args.text_source,
        sample_level=args.sample_level,
        motion_dir=args.motion_dir or None,
        libero_traj_file=args.libero_traj_file or None,
        allow_dummy_motion=args.allow_dummy_motion,
        allow_missing_motion=args.allow_missing_motion,
        motion_plugin=_load_motion_plugin(args.motion_plugin)
        if args.motion_plugin
        else None,
    )
    result = convert_perception_files(
        files,
        output_path=args.output,
        pretty_output_path=args.pretty_output or None,
        options=options,
    )

    print(f"Converted samples: {result.written}")
    print(f"Skipped entries: {result.skipped}")
    print(f"Saved to: {args.output}")
    if args.pretty_output:
        print(f"Pretty JSON saved to: {args.pretty_output}")


if __name__ == "__main__":
    main()

