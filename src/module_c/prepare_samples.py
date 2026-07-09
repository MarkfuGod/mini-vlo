from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MotionTuple = tuple[list[list[float]], list[float]]
MotionTracks = dict[str, MotionTuple]
MotionPluginResult = MotionTuple | MotionTracks
MotionPlugin = Callable[
    [dict[str, Any], dict[str, Any], str, str],
    MotionPluginResult | None,
]


@dataclass
class PrepareSamplesOptions:
    text_source: str = "task_instruction"
    sample_level: str = "segment"
    motion_path: str | Path | None = None
    motion_dir: str | Path | None = None
    libero_traj_file: str | Path | None = None
    allow_dummy_motion: bool = True
    allow_missing_motion: bool = False
    motion_plugin: MotionPlugin | None = None
    motion_fps: float = 24.0
    motion_tracks: list[str] | None = None


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


def _normalize_track(
    positions: Any,
    timestamps: Any,
) -> MotionTuple | None:
    if not isinstance(positions, list) or not isinstance(timestamps, list):
        return None
    try:
        normalized_positions = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in positions
            if isinstance(point, list) and len(point) == 3
        ]
        normalized_timestamps = [float(ts) for ts in timestamps]
    except (TypeError, ValueError):
        return None
    if not _is_valid_motion(normalized_positions, normalized_timestamps):
        return None
    return normalized_positions, normalized_timestamps


def _normalize_track_obj(obj: Any) -> MotionTuple | None:
    if not isinstance(obj, dict):
        return None
    return _normalize_track(obj.get("positions"), obj.get("timestamps"))


def _normalize_motion_tracks(obj: Any) -> MotionTracks | None:
    if not isinstance(obj, dict):
        return None

    tracks_obj = obj.get("tracks")
    if isinstance(tracks_obj, dict):
        tracks = {}
        for name, track_obj in tracks_obj.items():
            if isinstance(track_obj, tuple) and len(track_obj) == 2:
                track = _normalize_track(track_obj[0], track_obj[1])
            else:
                track = _normalize_track_obj(track_obj)
            if track is not None:
                tracks[str(name)] = track
        return tracks or None

    single_track = _normalize_track_obj(obj)
    if single_track is not None:
        return {"default": single_track}

    keyed_tracks = {}
    for name, track_obj in obj.items():
        if isinstance(track_obj, tuple) and len(track_obj) == 2:
            track = _normalize_track(track_obj[0], track_obj[1])
        else:
            track = _normalize_track_obj(track_obj)
        if track is not None:
            keyed_tracks[str(name)] = track
    if keyed_tracks:
        return keyed_tracks

    return None


def _normalize_plugin_motion(motion: MotionPluginResult | None) -> MotionTracks | None:
    if motion is None:
        return None
    if isinstance(motion, tuple) and len(motion) == 2:
        track = _normalize_track(motion[0], motion[1])
        return {"default": track} if track is not None else None
    return _normalize_motion_tracks(motion)


def _motion_to_sample_obj(motion: MotionTracks) -> dict[str, Any]:
    return {
        "tracks": {
            name: {
                "positions": positions,
                "timestamps": timestamps,
            }
            for name, (positions, timestamps) in motion.items()
        }
    }


def _filter_tracks_by_segment(
    tracks: MotionTracks,
    segment_obj: dict[str, Any] | None,
) -> MotionTracks | None:
    if segment_obj is None:
        return tracks

    start = segment_obj.get("start_time_sec")
    end = segment_obj.get("end_time_sec")
    if start is None or end is None or float(end) <= float(start):
        return tracks

    start_time = float(start)
    end_time = float(end)
    filtered_tracks: MotionTracks = {}
    for name, (positions, timestamps) in tracks.items():
        filtered = [
            (pos, ts)
            for pos, ts in zip(positions, timestamps)
            if start_time <= ts <= end_time
        ]
        if not filtered:
            continue
        track = _normalize_track(
            [pos for pos, _ in filtered],
            [ts for _, ts in filtered],
        )
        if track is not None:
            filtered_tracks[name] = track
    return filtered_tracks or None


def _build_dummy_motion(frame_timestamps: list[float]) -> MotionTracks:
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
    return {"default": (positions, timestamps)}


def _load_motion_from_file(
    motion_dir: Path,
    video_stem: str,
    segment_id: str,
    motion_fps: float,
    motion_tracks: list[str] | None,
) -> MotionTracks | None:
    candidates = [
        motion_dir / f"{video_stem}.json",
        motion_dir / f"{video_stem}_trajectory.json",
    ]
    motion_path = next((path for path in candidates if path.exists()), None)
    if motion_path is None:
        return None

    obj = _read_json(motion_path)
    if "segments" in obj and segment_id in obj["segments"]:
        obj = obj["segments"][segment_id]

    motion = _normalize_motion_tracks(obj)
    if motion is None:
        motion = _normalize_module_d_motion(obj, motion_fps, motion_tracks)
    return motion


def _normalize_motion_file_obj(
    obj: Any,
    segment_obj: dict[str, Any],
    segment_id: str,
    motion_fps: float,
    motion_tracks: list[str] | None,
) -> MotionTracks | None:
    if not isinstance(obj, dict):
        return None

    motion_obj = obj
    if "segments" in motion_obj and segment_id in motion_obj["segments"]:
        motion_obj = motion_obj["segments"][segment_id]

    motion = _normalize_libero_motion(motion_obj, segment_obj)
    if motion is None:
        motion = _normalize_motion_tracks(motion_obj)
    if motion is None:
        motion = _normalize_module_d_motion(motion_obj, motion_fps, motion_tracks)
    if motion is not None and "segments" not in motion_obj:
        motion = _filter_tracks_by_segment(motion, segment_obj)
    return motion


def _load_motion_from_path(
    motion_path: Path,
    video_stem: str,
    segment_id: str,
    segment_obj: dict[str, Any],
    motion_fps: float,
    motion_tracks: list[str] | None,
) -> MotionTracks | None:
    if not motion_path.exists():
        return None

    if motion_path.is_dir():
        motion = _load_motion_from_file(
            motion_path,
            video_stem,
            segment_id,
            motion_fps,
            motion_tracks,
        )
        return _filter_tracks_by_segment(motion, segment_obj) if motion else None

    if motion_path.is_file():
        return _normalize_motion_file_obj(
            _read_json(motion_path),
            segment_obj,
            segment_id,
            motion_fps,
            motion_tracks,
        )

    return None


def _normalize_module_d_motion(
    obj: Any,
    motion_fps: float,
    motion_tracks: list[str] | None,
) -> MotionTracks | None:
    if not isinstance(obj, dict):
        return None

    fps = float(motion_fps or 0.0)
    if fps <= 0:
        return None

    parsed_frames: list[tuple[int, dict[str, Any]]] = []
    for key, value in obj.items():
        match = re.match(r"frame_(\d+)$", str(key))
        if match and isinstance(value, dict):
            parsed_frames.append((int(match.group(1)), value))
    if not parsed_frames:
        return None

    parsed_frames.sort(key=lambda item: item[0])
    selected = set(motion_tracks) if motion_tracks else None
    available_names = []
    seen_names = set()
    for _, frame_obj in parsed_frames:
        for name, point in frame_obj.items():
            if selected is not None and name not in selected:
                continue
            if isinstance(point, dict) and name not in seen_names:
                seen_names.add(name)
                available_names.append(str(name))

    first_frame = parsed_frames[0][0]
    tracks: MotionTracks = {}
    for name in available_names:
        positions: list[list[float]] = []
        timestamps: list[float] = []
        for frame_id, frame_obj in parsed_frames:
            point = frame_obj.get(name)
            if not isinstance(point, dict):
                continue
            try:
                positions.append(
                    [
                        float(point["x"]),
                        float(point["y"]),
                        float(point["z"]),
                    ]
                )
                timestamps.append((frame_id - first_frame) / fps)
            except (KeyError, TypeError, ValueError):
                continue
        track = _normalize_track(positions, timestamps)
        if track is not None:
            tracks[name] = track

    return tracks or None


def _load_libero_motion_from_file(
    traj_path: Path,
    segment_obj: dict[str, Any] | None = None,
) -> MotionTracks | None:
    if not traj_path.exists():
        return None

    obj = _read_json(traj_path)
    return _normalize_libero_motion(obj, segment_obj)


def _normalize_libero_motion(
    obj: Any,
    segment_obj: dict[str, Any] | None = None,
) -> MotionTracks | None:
    if not isinstance(obj, dict):
        return None

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

    track = _normalize_track(positions, timestamps)
    if track is None:
        return None
    return _filter_tracks_by_segment({"eef": track}, segment_obj)


def _load_motion_plugin(spec: str) -> MotionPlugin:
    if ":" not in spec:
        raise ValueError("--motion-plugin format must be 'python.module:function_name'")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Motion plugin function not found or not callable: {spec}")
    return func


def _parse_motion_tracks(value: str) -> list[str] | None:
    tracks = [item.strip() for item in value.split(",") if item.strip()]
    return tracks or None


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
) -> MotionTracks | None:
    motion_path = Path(options.motion_path) if options.motion_path else None
    motion_dir = Path(options.motion_dir) if options.motion_dir else None
    libero_traj_file = (
        Path(options.libero_traj_file) if options.libero_traj_file else None
    )

    motion: MotionTracks | None = None
    if options.motion_plugin is not None:
        motion = _normalize_plugin_motion(
            options.motion_plugin(
                perception_obj,
                segment_obj,
                video_stem,
                segment_id,
            )
        )
    if motion is None and motion_path is not None:
        motion = _load_motion_from_path(
            motion_path,
            video_stem,
            segment_id,
            segment_obj,
            options.motion_fps,
            options.motion_tracks,
        )
    if motion is None and libero_traj_file is not None:
        motion = _load_libero_motion_from_file(libero_traj_file, segment_obj)
    if motion is None and motion_dir is not None:
        motion = _load_motion_from_file(
            motion_dir,
            video_stem,
            segment_id,
            options.motion_fps,
            options.motion_tracks,
        )
        motion = _filter_tracks_by_segment(motion, segment_obj) if motion else None
    if motion is None and options.allow_dummy_motion and not options.allow_missing_motion:
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
                sample["motion"] = _motion_to_sample_obj(motion)
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
                sample["motion"] = _motion_to_sample_obj(motion)
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
        "--motion-path",
        default="",
        help=(
            "Unified motion JSON file or directory. Files are auto-detected as "
            "LIBERO, Module D, or standard tracks; directories are searched by "
            "video stem."
        ),
    )
    parser.add_argument(
        "--motion-dir",
        default="",
        help="Deprecated alias for a motion directory. Prefer --motion-path.",
    )
    parser.add_argument(
        "--motion-fps",
        type=float,
        default=24.0,
        help="FPS used to timestamp Module D frame_N motion files.",
    )
    parser.add_argument(
        "--motion-tracks",
        default="",
        help="Comma-separated track names to keep, e.g. Root,Hand_R,Hand_L.",
    )
    parser.add_argument(
        "--libero-traj-file",
        default="",
        help="Deprecated alias for a LIBERO demo_0_traj.json path. Prefer --motion-path.",
    )
    parser.add_argument(
        "--allow-dummy-motion",
        action="store_true",
        default=True,
        help="Allow synthetic placeholder motion when no real motion is found (default).",
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
        motion_path=args.motion_path or None,
        motion_dir=args.motion_dir or None,
        libero_traj_file=args.libero_traj_file or None,
        allow_dummy_motion=args.allow_dummy_motion,
        allow_missing_motion=args.allow_missing_motion,
        motion_plugin=_load_motion_plugin(args.motion_plugin)
        if args.motion_plugin
        else None,
        motion_fps=args.motion_fps,
        motion_tracks=_parse_motion_tracks(args.motion_tracks),
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

