"""Construction, loading, and deterministic validation of paired video bundles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.semantic_motion.models import (
    FrameMap,
    MotionReference,
    Provenance,
    SharedTimebase,
    ViewBundle,
    ViewStream,
    VideoFrame,
)


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def probe_video(path: str | Path) -> tuple[float, int, float]:
    """Return FPS, frame count, and duration for a readable video."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video probing requires opencv-python") from exc

    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, frame_count, frame_count / fps


def _trajectory_metadata(path: str | Path) -> tuple[list[str], str, str]:
    trajectory_path = Path(path)
    try:
        obj = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], "meters", "world"
    track_names: list[str] = []
    tracks = obj.get("tracks")
    if isinstance(tracks, dict):
        track_names = [str(name) for name in tracks]
    elif isinstance(obj.get("steps"), list):
        track_names = ["eef"]
    else:
        frames = obj.get("frames", obj)
        if isinstance(frames, dict):
            first_frame = next(
                (
                    value
                    for key, value in frames.items()
                    if str(key).startswith("frame_") and isinstance(value, dict)
                ),
                {},
            )
            track_names = [str(name) for name in first_frame]
    return (
        track_names,
        str(obj.get("spatial_unit", obj.get("units", "meters"))),
        str(obj.get("coordinate_frame", "world")),
    )


def build_view_bundle(
    sample_id: str,
    views: dict[str, str | Path],
    *,
    trajectory_path: str | Path | None = None,
    trajectory_source: str = "unknown",
    dataset: str = "",
    spatial_unit: str = "meters",
) -> ViewBundle:
    """Probe synchronized videos and build a strict shared-timebase contract."""
    streams: dict[str, ViewStream] = {}
    for view_id, path in views.items():
        fps, frame_count, duration = probe_video(path)
        streams[view_id] = ViewStream(
            view_id=view_id,
            video_path=str(path),
            camera_type=(
                "egocentric"
                if view_id.lower() in {"ego", "egocentric", "eye_in_hand"}
                else "fixed"
            ),
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
        )
    if not streams:
        raise ValueError("A ViewBundle requires at least one view")

    reference = next(iter(streams.values()))
    timebase = SharedTimebase(
        fps=reference.fps,
        frame_count=reference.frame_count,
        duration_sec=reference.duration_sec,
        frame_map=FrameMap(frame_count=reference.frame_count),
    )
    track_names, detected_unit, coordinate_frame = (
        _trajectory_metadata(trajectory_path)
        if trajectory_path
        else ([], spatial_unit, "world")
    )
    bundle = ViewBundle(
        sample_id=sample_id,
        timebase=timebase,
        views=streams,
        trajectory=(
            MotionReference(
                path=str(trajectory_path),
                source=trajectory_source,
                spatial_unit=detected_unit or spatial_unit,
                coordinate_frame=coordinate_frame,
                track_names=track_names,
            )
            if trajectory_path
            else None
        ),
        provenance=Provenance(
            dataset=dataset,
            source_id=sample_id,
            generator="src.semantic_motion.view_bundle",
            revision=_git_revision(),
        ),
    )
    errors = validate_view_bundle(bundle)
    if errors:
        bundle.metadata["validation_diagnostics"] = errors
        bundle.metadata["validation_enforced"] = False
    return bundle


def _resolve_path(value: str, manifest_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if len(manifest_path.parents) > 3:
        repository_candidate = manifest_path.parents[3] / path
        if repository_candidate.exists():
            return str(repository_candidate)
    return str((manifest_path.parent / path).resolve())


def load_view_bundle(
    manifest_path: str | Path,
    sample_id: str | None = None,
) -> ViewBundle:
    """Load either a ViewBundle document or a LIBERO-style dataset manifest."""
    path = Path(manifest_path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "timebase" in obj and "views" in obj:
        return ViewBundle.model_validate(obj)

    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Manifest must contain a samples list")
    selected: dict[str, Any] | None = None
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample_id is None or sample.get("sample_id") == sample_id:
            selected = sample
            break
    if selected is None:
        raise KeyError(f"Sample not found in manifest: {sample_id}")

    view_paths = {
        str(view_id): _resolve_path(str(video_path), path)
        for view_id, video_path in selected.get("views", {}).items()
    }
    trajectory_value = selected.get("trajectory")
    return build_view_bundle(
        sample_id=str(selected["sample_id"]),
        views=view_paths,
        trajectory_path=(
            _resolve_path(str(trajectory_value), path) if trajectory_value else None
        ),
        trajectory_source=str(obj.get("dataset", "manifest")),
        dataset=str(obj.get("dataset", "")),
    )


def validate_view_bundle(
    bundle: ViewBundle,
    *,
    require_paired: bool = False,
    fps_tolerance: float = 1e-3,
    duration_tolerance_sec: float = 0.05,
) -> list[str]:
    """Return stable reason codes for contract and synchronization violations."""
    reasons: list[str] = []
    if require_paired and not {"fixed", "ego"}.issubset(bundle.views):
        reasons.append("sync_required_views_missing")
    if not bundle.views:
        reasons.append("sync_views_missing")
        return reasons

    for view_id, stream in bundle.views.items():
        if not Path(stream.video_path).is_file():
            reasons.append(f"sync_view_file_missing:{view_id}")
        if abs(stream.fps - bundle.timebase.fps) > fps_tolerance:
            reasons.append(f"sync_fps_mismatch:{view_id}")
        if stream.frame_count != bundle.timebase.frame_count:
            reasons.append(f"sync_frame_count_mismatch:{view_id}")
        if abs(stream.duration_sec - bundle.timebase.duration_sec) > duration_tolerance_sec:
            reasons.append(f"sync_duration_mismatch:{view_id}")

    frame_map = bundle.timebase.frame_map
    if frame_map.frame_count != bundle.timebase.frame_count:
        reasons.append("sync_frame_map_count_mismatch")
    if (
        frame_map.explicit_source_frames
        and len(frame_map.explicit_source_frames) != frame_map.frame_count
    ):
        reasons.append("sync_explicit_frame_map_count_mismatch")
    if bundle.trajectory and not Path(bundle.trajectory.path).is_file():
        reasons.append("sync_trajectory_file_missing")
    return sorted(set(reasons))


def extract_bundle_frames(
    bundle: ViewBundle,
    frame_indices: list[int],
    output_dir: str | Path,
) -> dict[str, list[VideoFrame]]:
    """Extract identical frame indices from all bundle views."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Frame extraction requires opencv-python") from exc

    output_root = Path(output_dir)
    result: dict[str, list[VideoFrame]] = {}
    for view_id, stream in bundle.views.items():
        cap = cv2.VideoCapture(stream.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {view_id} view: {stream.video_path}")
        frames: list[VideoFrame] = []
        view_dir = output_root / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        try:
            for frame_id, frame_index in enumerate(frame_indices):
                if frame_index < 0 or frame_index >= bundle.timebase.frame_count:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = cap.read()
                if not ok or image is None:
                    continue
                image_path = view_dir / f"frame_{frame_index:08d}.jpg"
                cv2.imwrite(str(image_path), image)
                frames.append(
                    VideoFrame(
                        frame_id=frame_id,
                        frame_index=frame_index,
                        timestamp_sec=frame_index / bundle.timebase.fps,
                        image_path=str(image_path),
                        view_id=view_id,
                    )
                )
        finally:
            cap.release()
        result[view_id] = frames
    return result
