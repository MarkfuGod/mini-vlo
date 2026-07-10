from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MotionTrack:
    positions: list[list[float]]
    timestamps: list[float]


@dataclass
class MotionData:
    tracks: dict[str, MotionTrack]
    spatial_unit: str = "meters"
    coordinate_frame: str = "world"
    source: str = "unknown"
    is_dummy: bool = False


@dataclass
class Sample:
    sample_id: str
    video_path: str
    text: str
    motion: MotionData | None = None
    label: str | None = None
    views: dict[str, str] = field(default_factory=dict)
    fps: float | None = None
    frame_count: int | None = None
    segment_start_sec: float | None = None
    segment_end_sec: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    sample_id: str
    motion_quality_score: float | None
    semantic_label: str
    semantic_confidence: float | None
    decision: str
    reason_codes: list[str]
    aux: dict[str, Any]

