"""Video-to-task pipeline for Semantic-Motion.

The pipeline follows the stronger variants of Video2Tasks-style systems:
sample temporal evidence, run frame-level VLM perception, detect task changes,
and aggregate each segment into macro intents plus executable micro steps.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.scenario import GroundTruth, Prediction, Scenario
from src.semantic_motion.augmentation import AugmentationStream, InstructionRewriter
from src.semantic_motion.models import (
    AugmentedInstruction,
    MacroIntent,
    MicroInstruction,
    PerceptionAnnotation,
    TaskSegment,
    VideoFrame,
    VideoFrameAnnotation,
    VideoTaskRecord,
)
from src.semantic_motion.perception import PerceptionStream
from src.semantic_motion.recognition import RecognitionModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = _norm(item)
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _majority(items: Iterable[str]) -> str:
    values = [_norm(item) for item in items if item]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _split_action(action: str) -> tuple[str, str]:
    parts = action.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _tokens(items: Iterable[str]) -> set[str]:
    blob = " ".join(items).lower()
    return set(re.findall(r"[a-z0-9]+", blob))


GENERIC_TARGET_TOKENS = {
    "object",
    "target",
    "unknown",
    "destination",
    "location",
    "place",
    "placement",
}


def _content_tokens(text: str | None) -> set[str]:
    return _tokens([text or ""]) - GENERIC_TARGET_TOKENS


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    max_frames: int = 12,
) -> list[VideoFrame]:
    """Uniformly sample frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video sampling requires opencv-python. Install requirements.txt "
            "or pass --frame-dir with pre-extracted frames."
        ) from exc

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count from video: {video_path}")

    max_frames = max(1, min(max_frames, frame_count))
    if max_frames == 1:
        indices = [frame_count // 2]
    else:
        indices = [
            round(i * (frame_count - 1) / (max_frames - 1))
            for i in range(max_frames)
        ]

    frames: list[VideoFrame] = []
    for frame_id, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        image_path = output_dir / f"frame_{frame_id:04d}_{frame_index:06d}.jpg"
        cv2.imwrite(str(image_path), frame)
        timestamp = frame_index / fps if fps > 0 else float(frame_id)
        frames.append(
            VideoFrame(
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp_sec=timestamp,
                image_path=str(image_path),
            )
        )

    cap.release()
    return frames


def load_frame_directory(
    frame_dir: str | Path,
    fps: float = 1.0,
) -> list[VideoFrame]:
    """Load pre-extracted frames as a video-like temporal sequence."""
    frame_dir = Path(frame_dir)
    paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    frames: list[VideoFrame] = []
    for frame_id, path in enumerate(paths):
        timestamp = frame_id / fps if fps > 0 else float(frame_id)
        frames.append(
            VideoFrame(
                frame_id=frame_id,
                frame_index=frame_id,
                timestamp_sec=timestamp,
                image_path=str(path),
            )
        )
    return frames


def _frame_scenario(
    frame: VideoFrame,
    instruction: str,
) -> Scenario:
    return Scenario(
        id=f"video_frame_{frame.frame_id:04d}",
        category="video",
        image_path=frame.image_path,
        instruction=instruction,
        ground_truth=GroundTruth(
            objects=[],
            spatial_relations=[],
            task_type="",
            action_sequence=[],
            target_object="",
            destination=None,
        ),
    )


def _frame_instruction(source_instruction: str, frame: VideoFrame) -> str:
    if source_instruction:
        return (
            f"{source_instruction}\n"
            f"Analyze this video frame at {frame.timestamp_sec:.2f}s and infer "
            "the current manipulation subtask and primitive action state."
        )
    return (
        f"Analyze this video frame at {frame.timestamp_sec:.2f}s. Infer the "
        "robot manipulation task, target object, destination, and primitive "
        "action sequence from the visible scene."
    )


class TemporalTaskAggregator:
    """Detect task boundaries and merge frame annotations into task segments."""

    def __init__(
        self,
        action_similarity_threshold: float = 0.30,
        min_segment_frames: int = 1,
    ):
        self.action_similarity_threshold = action_similarity_threshold
        self.min_segment_frames = min_segment_frames

    def segment(
        self,
        frame_annotations: list[VideoFrameAnnotation],
        augmentation: AugmentationStream,
        num_variants: int,
    ) -> list[TaskSegment]:
        if not frame_annotations:
            return []

        groups: list[list[VideoFrameAnnotation]] = []
        current: list[VideoFrameAnnotation] = [frame_annotations[0]]

        for frame_annotation in frame_annotations[1:]:
            previous = current[-1]
            if self._is_boundary(previous.annotation, frame_annotation.annotation):
                if len(current) < self.min_segment_frames and groups:
                    groups[-1].extend(current)
                else:
                    groups.append(current)
                current = [frame_annotation]
            else:
                current.append(frame_annotation)

        if len(current) < self.min_segment_frames and groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

        return [
            self._aggregate_group(idx, group, augmentation, num_variants)
            for idx, group in enumerate(groups)
        ]

    def _is_boundary(
        self,
        previous: PerceptionAnnotation,
        current: PerceptionAnnotation,
    ) -> bool:
        previous_intent = previous.macro_intent
        current_intent = current.macro_intent

        if (
            previous_intent.task_type
            and current_intent.task_type
            and _norm(previous_intent.task_type) != _norm(current_intent.task_type)
        ):
            return True

        if (
            previous_intent.target_object
            and current_intent.target_object
            and not self._targets_match(
                previous_intent.target_object,
                current_intent.target_object,
            )
        ):
            return True

        return False

    def _targets_match(self, previous: str, current: str) -> bool:
        previous_tokens = _content_tokens(previous)
        current_tokens = _content_tokens(current)
        if not previous_tokens or not current_tokens:
            return True
        return _jaccard(previous_tokens, current_tokens) >= self.action_similarity_threshold

    def _aggregate_group(
        self,
        idx: int,
        group: list[VideoFrameAnnotation],
        augmentation: AugmentationStream,
        num_variants: int,
    ) -> TaskSegment:
        annotations = [item.annotation for item in group]
        frames = [item.frame for item in group]
        task_type = _majority(a.macro_intent.task_type for a in annotations)
        target = _majority(a.macro_intent.target_object for a in annotations)
        destination = _majority(a.macro_intent.destination or "" for a in annotations) or None

        objects = _dedupe(obj for annotation in annotations for obj in annotation.objects)
        spatial_relations = _dedupe(
            rel for annotation in annotations for rel in annotation.spatial_relations
        )
        action_texts = _dedupe(
            step.text
            for annotation in annotations
            for step in annotation.micro_instructions
        )
        confidence_values = [
            annotation.macro_intent.confidence
            for annotation in annotations
            if annotation.macro_intent.confidence is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        micro_steps: list[MicroInstruction] = []
        for step_id, action in enumerate(action_texts, start=1):
            verb, obj = _split_action(action)
            micro_steps.append(
                MicroInstruction(
                    step_id=step_id,
                    text=action,
                    verb=verb,
                    object=obj,
                    confidence=confidence,
                )
            )

        macro_intent = MacroIntent(
            task_type=task_type,
            target_object=target,
            destination=destination,
            confidence=confidence,
        )
        task_instruction = self._build_instruction(
            macro_intent,
            micro_steps,
        )
        segment_annotation = PerceptionAnnotation(
            scenario_id=f"video_segment_{idx:03d}",
            image_path=frames[0].image_path,
            source_instruction=task_instruction,
            objects=objects,
            spatial_relations=spatial_relations,
            macro_intent=macro_intent,
            micro_instructions=micro_steps,
            raw_recognition="\n".join(a.raw_recognition for a in annotations),
            metadata={
                "source": "video_temporal_aggregation",
                "frame_ids": [frame.frame_id for frame in frames],
            },
        )
        augmented = augmentation.augment(
            segment_annotation,
            num_variants=num_variants,
        )

        return TaskSegment(
            segment_id=f"segment_{idx:03d}",
            start_time_sec=frames[0].timestamp_sec,
            end_time_sec=frames[-1].timestamp_sec,
            frame_ids=[frame.frame_id for frame in frames],
            task_instruction=task_instruction,
            objects=objects,
            spatial_relations=spatial_relations,
            macro_intent=macro_intent,
            micro_instructions=micro_steps,
            augmented_instructions=augmented,
            confidence=confidence,
            metadata={"num_frames": len(frames)},
        )

    def _build_instruction(
        self,
        intent: MacroIntent,
        micro_steps: list[MicroInstruction],
    ) -> str:
        target = intent.target_object or "the target object"
        if intent.destination:
            return f"{intent.task_type.replace('_', ' ')} {target} to {intent.destination}"
        if intent.task_type and target:
            return f"{intent.task_type.replace('_', ' ')} {target}"
        if micro_steps:
            return micro_steps[-1].text
        return "infer manipulation task"


class VideoTaskPipeline:
    """End-to-end video-to-task stream using an existing recognition model."""

    def __init__(
        self,
        recognizer: RecognitionModel,
        rewriter: InstructionRewriter | None = None,
        aggregator: TemporalTaskAggregator | None = None,
    ):
        self.perception = PerceptionStream(recognizer)
        self.augmentation = AugmentationStream(rewriter)
        self.aggregator = aggregator or TemporalTaskAggregator()

    def run_frames(
        self,
        frames: list[VideoFrame],
        video_path: str | Path,
        source_instruction: str = "",
        num_variants: int = 3,
    ) -> VideoTaskRecord:
        frame_annotations: list[VideoFrameAnnotation] = []
        for frame in frames:
            instruction = _frame_instruction(source_instruction, frame)
            scenario = _frame_scenario(frame, instruction)
            prediction = self.perception.recognizer.analyze(
                frame.image_path,
                instruction,
            )
            annotation = self.perception.from_prediction(
                scenario,
                frame.image_path,
                prediction,
            )
            frame_annotations.append(
                VideoFrameAnnotation(frame=frame, annotation=annotation)
            )

        segments = self.aggregator.segment(
            frame_annotations,
            augmentation=self.augmentation,
            num_variants=num_variants,
        )
        return VideoTaskRecord(
            video_path=str(video_path),
            source_instruction=source_instruction,
            frames=frame_annotations,
            task_segments=segments,
            metadata={
                "num_sampled_frames": len(frames),
                "num_task_segments": len(segments),
                "method": (
                    "keyframe_perception_temporal_boundary_detection_"
                    "segment_instruction_augmentation"
                ),
            },
        )

    def run_video(
        self,
        video_path: str | Path,
        work_dir: str | Path,
        source_instruction: str = "",
        max_frames: int = 12,
        num_variants: int = 3,
    ) -> VideoTaskRecord:
        frames = sample_video_frames(
            video_path,
            output_dir=Path(work_dir) / "frames",
            max_frames=max_frames,
        )
        return self.run_frames(
            frames,
            video_path=video_path,
            source_instruction=source_instruction,
            num_variants=num_variants,
        )

    def run_frame_dir(
        self,
        frame_dir: str | Path,
        source_instruction: str = "",
        fps: float = 1.0,
        num_variants: int = 3,
    ) -> VideoTaskRecord:
        frames = load_frame_directory(frame_dir, fps=fps)
        return self.run_frames(
            frames,
            video_path=frame_dir,
            source_instruction=source_instruction,
            num_variants=num_variants,
        )
