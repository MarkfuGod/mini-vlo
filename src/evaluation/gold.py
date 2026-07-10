"""Versioned human-gold annotation contract and acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class GoldMicroAction(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    verb: str
    object: str = ""
    body_part: str = ""
    contact_state: str = "unknown"
    posture: str = ""
    evidence_frames: dict[str, list[int]] = Field(default_factory=dict)


class GoldSegment(BaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    task_type: str
    domain: str
    target_object: str = ""
    destination: str = ""
    micro_actions: list[GoldMicroAction] = Field(default_factory=list)
    quality_label: Literal["keep", "drop", "uncertain"] = "uncertain"
    corruption_types: list[str] = Field(default_factory=list)


class AnnotatorRecord(BaseModel):
    annotator_id: str
    completed_at: str = ""
    independent: bool = True


class GoldAnnotation(BaseModel):
    schema_version: str = "semantic-motion-gold/v1"
    sample_id: str
    annotation_status: Literal[
        "pending_human",
        "single_annotated",
        "double_annotated",
        "adjudicated",
    ] = "pending_human"
    source_sample_id: str
    source_views: dict[str, str]
    clip_start_sec: float = 0.0
    clip_end_sec: float
    fps: float
    frame_count: int
    weak_task_title: str = ""
    boundaries_sec: list[float] = Field(default_factory=list)
    segments: list[GoldSegment] = Field(default_factory=list)
    annotators: list[AnnotatorRecord] = Field(default_factory=list)
    adjudicator_id: str = ""
    notes: str = ""


def validate_gold(annotation: GoldAnnotation, formal: bool = True) -> list[str]:
    reasons: list[str] = []
    if annotation.clip_end_sec <= annotation.clip_start_sec:
        reasons.append("gold_invalid_clip_interval")
    if annotation.fps <= 0 or annotation.frame_count <= 0:
        reasons.append("gold_invalid_timebase")
    if not {"fixed", "ego"}.issubset(annotation.source_views):
        reasons.append("gold_paired_views_missing")
    if formal and annotation.annotation_status != "adjudicated":
        reasons.append("gold_not_adjudicated")
    if formal and len({item.annotator_id for item in annotation.annotators}) < 2:
        reasons.append("gold_requires_two_annotators")
    previous_end = annotation.clip_start_sec
    for segment in sorted(annotation.segments, key=lambda item: item.start_sec):
        if segment.end_sec <= segment.start_sec:
            reasons.append(f"gold_invalid_segment:{segment.segment_id}")
        if segment.start_sec < previous_end - 1e-6:
            reasons.append(f"gold_overlapping_segment:{segment.segment_id}")
        if (
            segment.start_sec < annotation.clip_start_sec
            or segment.end_sec > annotation.clip_end_sec
        ):
            reasons.append(f"gold_segment_outside_clip:{segment.segment_id}")
        previous_end = max(previous_end, segment.end_sec)
        for action in segment.micro_actions:
            if action.end_sec <= action.start_sec:
                reasons.append(f"gold_invalid_micro:{segment.segment_id}")
    return sorted(set(reasons))


def load_gold(path: str | Path, formal: bool = True) -> GoldAnnotation:
    annotation = GoldAnnotation.model_validate_json(Path(path).read_text(encoding="utf-8"))
    reasons = validate_gold(annotation, formal=formal)
    if reasons:
        raise ValueError("Gold annotation rejected: " + ", ".join(reasons))
    return annotation


def save_gold(annotation: GoldAnnotation, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(annotation.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
