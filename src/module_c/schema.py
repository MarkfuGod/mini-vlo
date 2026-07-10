from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MotionTrack:
    positions: list[list[float]]
    timestamps: list[float]


@dataclass
class MotionData:
    tracks: dict[str, MotionTrack]


@dataclass
class Sample:
    sample_id: str
    video_path: str
    text: str
    motion: MotionData | None = None
    label: str | None = None


@dataclass
class RefinementResult:
    sample_id: str
    motion_quality_score: float | None
    semantic_label: str
    semantic_confidence: float | None
    decision: str
    reason_codes: list[str]
    aux: dict[str, Any]

