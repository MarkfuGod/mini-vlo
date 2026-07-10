"""Deterministic cross-view, timebase, and motion-alignment quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Sample


@dataclass
class SyncConfig:
    require_paired_views: bool = True
    fps_tolerance: float = 1e-3
    duration_tolerance_sec: float = 0.05
    require_motion_coverage: bool = True
    allowed_spatial_units: tuple[str, ...] = ("meters", "centimeters", "millimeters")


def _probe(path: str) -> tuple[float, int, float] | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        return None
    return fps, count, count / fps


def check_sample_sync(
    sample: Sample,
    cfg: SyncConfig,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate data availability and alignment before semantic judging."""
    reasons: list[str] = []
    views = dict(sample.views)
    if not views and sample.video_path:
        views = {"single": sample.video_path}
    if cfg.require_paired_views and not {"fixed", "ego"}.issubset(views):
        reasons.append("sync_required_views_missing")
    if not views:
        reasons.append("sync_views_missing")

    metadata: dict[str, tuple[float, int, float]] = {}
    for view_id, video_path in views.items():
        if not Path(video_path).is_file():
            reasons.append(f"sync_view_file_missing:{view_id}")
            continue
        probed = _probe(video_path)
        if probed is None:
            reasons.append(f"sync_view_unreadable:{view_id}")
            continue
        metadata[view_id] = probed

    if metadata:
        reference_id = sorted(metadata)[0]
        reference = metadata[reference_id]
        for view_id, values in metadata.items():
            if abs(values[0] - reference[0]) > cfg.fps_tolerance:
                reasons.append(f"sync_fps_mismatch:{view_id}")
            if values[1] != reference[1]:
                reasons.append(f"sync_frame_count_mismatch:{view_id}")
            if abs(values[2] - reference[2]) > cfg.duration_tolerance_sec:
                reasons.append(f"sync_duration_mismatch:{view_id}")
        if sample.fps is not None and abs(reference[0] - sample.fps) > cfg.fps_tolerance:
            reasons.append("sync_declared_fps_mismatch")
        if sample.frame_count is not None and reference[1] != sample.frame_count:
            reasons.append("sync_declared_frame_count_mismatch")

    motion_coverage: dict[str, tuple[float, float]] = {}
    if sample.motion is None:
        reasons.append("motion_missing")
    else:
        if sample.motion.is_dummy:
            reasons.append("dummy_motion_forbidden")
        if sample.motion.spatial_unit not in cfg.allowed_spatial_units:
            reasons.append("motion_unit_unsupported")
        for track_name, track in sample.motion.tracks.items():
            if not track.timestamps:
                reasons.append(f"motion_track_empty:{track_name}")
                continue
            start, end = min(track.timestamps), max(track.timestamps)
            motion_coverage[track_name] = (start, end)
            if (
                cfg.require_motion_coverage
                and sample.segment_start_sec is not None
                and sample.segment_end_sec is not None
                and (
                    start > sample.segment_start_sec + cfg.duration_tolerance_sec
                    or end < sample.segment_end_sec - cfg.duration_tolerance_sec
                )
            ):
                reasons.append(f"motion_segment_not_covered:{track_name}")

    unique_reasons = sorted(set(reasons))
    aux: dict[str, Any] = {
        "valid": not unique_reasons,
        "views": {
            view_id: {
                "fps": values[0],
                "frame_count": values[1],
                "duration_sec": values[2],
            }
            for view_id, values in metadata.items()
        },
        "motion_coverage": motion_coverage,
    }
    return not unique_reasons, aux, unique_reasons
