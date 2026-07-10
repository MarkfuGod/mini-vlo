"""Data models for the Semantic-Motion perception and augmentation streams."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FrameMap(BaseModel):
    """Mapping between encoded video frames and the shared source timeline."""

    video_frame_start: int = 0
    source_frame_start: int = 0
    frame_count: int
    step: int = 1
    explicit_source_frames: list[int] = Field(default_factory=list)


class SharedTimebase(BaseModel):
    """Clock shared by every view and the associated motion trajectory."""

    fps: float
    frame_count: int
    frame_start: int = 0
    duration_sec: float
    time_unit: Literal["seconds"] = "seconds"
    frame_map: FrameMap


class ViewStream(BaseModel):
    """One synchronized camera stream in a multi-view sample."""

    view_id: str
    video_path: str
    camera_name: str = ""
    camera_type: str = ""
    fps: float
    frame_count: int
    duration_sec: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MotionReference(BaseModel):
    """Reference to motion data aligned to the shared timebase."""

    path: str
    source: str = "unknown"
    spatial_unit: str = "meters"
    coordinate_frame: str = "world"
    track_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Lineage required to reproduce or audit one sample."""

    dataset: str = ""
    source_id: str = ""
    generator: str = ""
    revision: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


class ViewBundle(BaseModel):
    """Paired camera views, motion, and provenance on one shared clock."""

    schema_version: str = "semantic-motion-view-bundle/v1"
    sample_id: str
    timebase: SharedTimebase
    views: dict[str, ViewStream]
    trajectory: MotionReference | None = None
    provenance: Provenance = Field(default_factory=Provenance)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def video_paths(self) -> dict[str, str]:
        return {view_id: stream.video_path for view_id, stream in self.views.items()}


class ObservedTrajectoryRef(BaseModel):
    """Motion tracks and interval supporting one observed micro action."""

    track_names: list[str] = Field(default_factory=list)
    start_time_sec: float
    end_time_sec: float
    coordinate_frame: str = "world"


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
    domain: Literal["locomotion", "manipulation", "mixed", "unknown"] = "unknown"


class MicroInstruction(BaseModel):
    """Low-level natural-language primitive in an action sequence."""

    step_id: int
    text: str
    verb: str = ""
    object: str = ""
    confidence: Optional[float] = None
    body_part: str | None = None
    contact_state: str | None = None
    posture: str | None = None
    observed_trajectory: ObservedTrajectoryRef | None = None
    evidence_frame_ids: list[int] = Field(default_factory=list)


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
    view_id: str = ""
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
    view_id: str = ""


class VideoFrameAnnotation(BaseModel):
    """Per-frame perception output for a sampled video frame."""

    frame: VideoFrame
    annotation: PerceptionAnnotation


class MultiViewFrameAnnotation(BaseModel):
    """Per-timestamp evidence and fused semantics from synchronized views."""

    frame_id: int
    timestamp_sec: float
    view_frames: dict[str, VideoFrame] = Field(default_factory=dict)
    view_annotations: dict[str, PerceptionAnnotation] = Field(default_factory=dict)
    fused_annotation: PerceptionAnnotation


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
    start_frame: int | None = None
    end_frame: int | None = None
    evidence_by_view: dict[str, list[int]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoTaskRecord(BaseModel):
    """Full video-to-task output: frame evidence plus task-level segments."""

    schema_version: str = "semantic-motion-video-task/v2"
    video_path: str = ""
    view_bundle: ViewBundle | None = None
    source_instruction: str = ""
    frames: list[VideoFrameAnnotation] = Field(default_factory=list)
    multi_view_frames: list[MultiViewFrameAnnotation] = Field(default_factory=list)
    task_segments: list[TaskSegment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
