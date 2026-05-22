"""Data models for the Semantic-Motion perception and augmentation streams."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MacroIntent(BaseModel):
    """High-level task intent inferred from a robot scene and instruction."""

    task_type: str = Field(description="Coarse task type, e.g. pick_and_place")
    target_object: str = Field(description="Primary object to interact with")
    destination: Optional[str] = Field(
        None, description="Goal location or object, if the task has one"
    )
    confidence: Optional[float] = Field(
        None, description="Optional model confidence in the macro intent"
    )


class MicroInstruction(BaseModel):
    """Low-level natural-language primitive in an action sequence."""

    step_id: int
    text: str
    verb: str = ""
    object: str = ""
    confidence: Optional[float] = None


class PerceptionAnnotation(BaseModel):
    """Automated semantic labels produced by the perception stream."""

    scenario_id: str
    image_path: str
    source_instruction: str
    objects: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    macro_intent: MacroIntent
    micro_instructions: list[MicroInstruction] = Field(default_factory=list)
    raw_recognition: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AugmentedInstruction(BaseModel):
    """A rewritten instruction derived from a perception annotation."""

    text: str
    strategy: str
    source_step_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticMotionRecord(BaseModel):
    """One complete output record from the first two Semantic-Motion streams."""

    annotation: PerceptionAnnotation
    augmented_instructions: list[AugmentedInstruction] = Field(default_factory=list)


class VideoFrame(BaseModel):
    """A sampled frame used as temporal evidence for video-to-task."""

    frame_id: int
    frame_index: int
    timestamp_sec: float
    image_path: str


class VideoFrameAnnotation(BaseModel):
    """Per-frame perception output for a sampled video frame."""

    frame: VideoFrame
    annotation: PerceptionAnnotation


class TaskSegment(BaseModel):
    """A temporally grounded task segment inferred from video frames."""

    segment_id: str
    start_time_sec: float
    end_time_sec: float
    frame_ids: list[int]
    task_instruction: str
    objects: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    macro_intent: MacroIntent
    micro_instructions: list[MicroInstruction] = Field(default_factory=list)
    augmented_instructions: list[AugmentedInstruction] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoTaskRecord(BaseModel):
    """Full video-to-task output: frame evidence plus task-level segments."""

    video_path: str
    source_instruction: str = ""
    frames: list[VideoFrameAnnotation] = Field(default_factory=list)
    task_segments: list[TaskSegment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
