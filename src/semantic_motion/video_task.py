"""Video-to-task pipeline for Semantic-Motion.

The pipeline follows the stronger variants of Video2Tasks-style systems:
sample temporal evidence, run frame-level VLM perception, detect task changes,
and aggregate each segment into macro intents plus executable micro steps.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.scenario import GroundTruth, Prediction, Scenario
from src.semantic_motion.augmentation import AugmentationStream, InstructionRewriter
from src.semantic_motion.models import (
    AugmentedInstruction,
    MacroIntent,
    MicroInstruction,
    MultiViewFrameAnnotation,
    ObservedTrajectoryRef,
    PerceptionAnnotation,
    TaskSegment,
    ViewBundle,
    VideoFrame,
    VideoFrameAnnotation,
    VideoTaskRecord,
)
from src.semantic_motion.perception import PerceptionStream
from src.semantic_motion.recognition import RecognitionModel
from src.semantic_motion.view_bundle import (
    build_view_bundle,
    extract_bundle_frames,
    validate_view_bundle,
)
from src.semantic_motion.windowing import (
    CutVote,
    aggregate_cut_votes,
    build_micro_windows,
    build_temporal_windows,
    intervals_from_cuts,
    transition_votes,
)


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


def _fuse_predictions(predictions: list[Prediction]) -> Prediction:
    """Late-fuse per-view predictions when a recognizer lacks multi-image support."""
    if not predictions:
        return Prediction()
    action_sequence = _dedupe(
        action for prediction in predictions for action in prediction.action_sequence
    )
    action_details: list[dict[str, Any]] = []
    seen_details: set[str] = set()
    for prediction in predictions:
        for detail in prediction.action_details:
            key = _norm(str(detail.get("text", "")))
            if key and key not in seen_details:
                seen_details.add(key)
                action_details.append(detail)
    confidence_values = [
        float(prediction.confidence)
        for prediction in predictions
        if prediction.confidence is not None
    ]
    return Prediction(
        objects=_dedupe(obj for prediction in predictions for obj in prediction.objects),
        spatial_relations=_dedupe(
            relation
            for prediction in predictions
            for relation in prediction.spatial_relations
        ),
        task_type=_majority(prediction.task_type for prediction in predictions),
        action_sequence=action_sequence,
        target_object=_majority(
            prediction.target_object for prediction in predictions
        ),
        destination=(
            _majority(prediction.destination or "" for prediction in predictions) or None
        ),
        domain=_majority(prediction.domain for prediction in predictions) or "unknown",
        instruction=_majority(
            prediction.instruction for prediction in predictions
        ),
        transitions=sorted(
            {
                int(value)
                for prediction in predictions
                for value in prediction.transitions
                if isinstance(value, (int, float))
            }
        ),
        confidence=(
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
        action_details=action_details,
        raw_text="\n".join(prediction.raw_text for prediction in predictions),
    )


def _ordered_view_paths(
    frames_by_view: dict[str, list[VideoFrame]],
    view_ids: list[str],
) -> list[str]:
    """Interleave synchronized views by timestamp for early-fusion VLM input."""
    by_index = {
        view_id: {frame.frame_index: frame for frame in frames_by_view.get(view_id, [])}
        for view_id in view_ids
    }
    indices = sorted(
        {
            frame_index
            for view_frames in by_index.values()
            for frame_index in view_frames
        }
    )
    return [
        by_index[view_id][frame_index].image_path
        for frame_index in indices
        for view_id in view_ids
        if frame_index in by_index[view_id]
    ]


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
        domain = _majority(a.macro_intent.domain for a in annotations) or "unknown"

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
            domain=domain,
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
            start_frame=frames[0].frame_index,
            end_frame=frames[-1].frame_index + 1,
            evidence_by_view={
                frames[0].view_id or "single": [
                    frame.frame_index for frame in frames
                ]
            },
            metadata={"num_frames": len(frames), "legacy_single_view": True},
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

    def _recognize_many(
        self,
        image_paths: list[str],
        instruction: str,
    ) -> Prediction:
        analyze_many = getattr(self.perception.recognizer, "analyze_many", None)
        if callable(analyze_many):
            return analyze_many(image_paths, instruction)
        return _fuse_predictions(
            [
                self.perception.recognizer.analyze(image_path, instruction)
                for image_path in image_paths
            ]
        )

    @staticmethod
    def _selected_views(bundle: ViewBundle, view_mode: str) -> list[str]:
        mode = view_mode.lower()
        if mode == "fused":
            selected = [
                view_id for view_id in ("fixed", "ego") if view_id in bundle.views
            ]
            return selected or list(bundle.views)
        if mode not in bundle.views:
            return list(bundle.views)[:1]
        return [mode]

    def _window_prediction(
        self,
        bundle: ViewBundle,
        frame_indices: list[int],
        output_dir: Path,
        view_ids: list[str],
        source_instruction: str,
        *,
        level: str,
    ) -> tuple[Prediction, dict[str, list[VideoFrame]]]:
        frames_by_view = extract_bundle_frames(bundle, frame_indices, output_dir)
        image_paths = _ordered_view_paths(frames_by_view, view_ids)
        if not image_paths:
            raise RuntimeError(f"No frames extracted for {bundle.sample_id}")
        paired_note = (
            "At each timestamp, images are ordered fixed then ego. Transition "
            "indices refer to timestamps, not individual images."
            if len(view_ids) > 1
            else "Transition indices refer to the ordered image timestamps."
        )
        instruction = (
            f"{source_instruction}\n" if source_instruction else ""
        ) + (
            f"Analyze this {level} temporal window with {len(frame_indices)} "
            f"timestamps. {paired_note} Report only observed loco-manipulation "
            "facts, body parts, contact states, posture, primitive order, and "
            "atomic task transitions."
        )
        return self._recognize_many(image_paths, instruction), frames_by_view

    def _annotation_for_prediction(
        self,
        prediction: Prediction,
        frames_by_view: dict[str, list[VideoFrame]],
        instruction: str,
        scenario_id: str,
        view_mode: str,
    ) -> PerceptionAnnotation:
        first_frame = next(
            (
                frame
                for frames in frames_by_view.values()
                for frame in frames
            ),
            VideoFrame(
                frame_id=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="",
            ),
        )
        scenario = _frame_scenario(first_frame, instruction)
        scenario.id = scenario_id
        annotation = self.perception.from_prediction(
            scenario,
            first_frame.image_path,
            prediction,
        )
        annotation.view_id = view_mode
        annotation.metadata["view_mode"] = view_mode
        annotation.metadata["evidence_paths"] = [
            frame.image_path
            for frames in frames_by_view.values()
            for frame in frames
        ]
        return annotation

    def _build_windowed_segment(
        self,
        *,
        segment_index: int,
        start_frame: int,
        end_frame: int,
        start_sec: float,
        end_sec: float,
        bundle: ViewBundle,
        view_ids: list[str],
        view_mode: str,
        source_instruction: str,
        work_dir: Path,
        macro_predictions: list[Prediction],
        num_variants: int,
        micro_window_sec: float,
        micro_step_sec: float,
        micro_frames: int,
    ) -> TaskSegment:
        micro_windows = build_micro_windows(
            start_frame,
            end_frame,
            fps=bundle.timebase.fps,
            window_sec=micro_window_sec,
            step_sec=micro_step_sec,
            frames_per_window=micro_frames,
        )
        annotated_windows: list[
            tuple[Any, Prediction, PerceptionAnnotation, dict[str, list[VideoFrame]]]
        ] = []
        for micro_window in micro_windows:
            prediction, frames_by_view = self._window_prediction(
                bundle,
                micro_window.frame_indices,
                work_dir
                / "micro"
                / f"segment_{segment_index:03d}"
                / f"window_{micro_window.window_id:03d}",
                view_ids,
                source_instruction,
                level="1–3 second micro-action",
            )
            annotation = self._annotation_for_prediction(
                prediction,
                frames_by_view,
                source_instruction,
                f"{bundle.sample_id}_segment_{segment_index:03d}_"
                f"micro_{micro_window.window_id:03d}",
                view_mode,
            )
            annotated_windows.append(
                (micro_window, prediction, annotation, frames_by_view)
            )

        predictions = [item[1] for item in annotated_windows] or macro_predictions
        annotations = [item[2] for item in annotated_windows]
        task_type = _majority(prediction.task_type for prediction in predictions)
        target = _majority(prediction.target_object for prediction in predictions)
        destination = (
            _majority(prediction.destination or "" for prediction in predictions)
            or None
        )
        domain = _majority(prediction.domain for prediction in predictions) or "unknown"
        confidence_values = [
            float(prediction.confidence)
            for prediction in predictions
            if prediction.confidence is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        macro_intent = MacroIntent(
            task_type=task_type,
            target_object=target,
            destination=destination,
            confidence=confidence,
            domain=domain
            if domain in {"locomotion", "manipulation", "mixed", "unknown"}
            else "unknown",
        )

        micro_steps: list[MicroInstruction] = []
        seen_steps: set[str] = set()
        trajectory_tracks = (
            bundle.trajectory.track_names if bundle.trajectory else []
        )
        canonical_windows = annotated_windows[:1]
        for micro_window, _, annotation, _ in canonical_windows:
            for step in annotation.micro_instructions:
                key = _norm(step.text)
                if not key or key in seen_steps:
                    continue
                seen_steps.add(key)
                evidence = sorted(set(micro_window.frame_indices))
                micro_steps.append(
                    step.model_copy(
                        update={
                            "step_id": len(micro_steps) + 1,
                            "evidence_frame_ids": evidence,
                            "observed_trajectory": (
                                ObservedTrajectoryRef(
                                    track_names=trajectory_tracks,
                                    start_time_sec=micro_window.start_frame
                                    / bundle.timebase.fps,
                                    end_time_sec=(micro_window.end_frame + 1)
                                    / bundle.timebase.fps,
                                    coordinate_frame=(
                                        bundle.trajectory.coordinate_frame
                                        if bundle.trajectory
                                        else "world"
                                    ),
                                )
                                if bundle.trajectory
                                else None
                            ),
                        }
                    )
                )

        objects = _dedupe(
            obj for annotation in annotations for obj in annotation.objects
        )
        spatial_relations = _dedupe(
            relation
            for annotation in annotations
            for relation in annotation.spatial_relations
        )
        instruction_candidates = [
            prediction.instruction.strip()
            for prediction in predictions
            if prediction.instruction.strip()
        ]
        task_instruction = _majority(instruction_candidates)
        if not task_instruction:
            task_instruction = self.aggregator._build_instruction(
                macro_intent,
                micro_steps,
            )
        segment_annotation = PerceptionAnnotation(
            scenario_id=f"{bundle.sample_id}_segment_{segment_index:03d}",
            image_path=(
                annotated_windows[0][2].image_path if annotated_windows else ""
            ),
            source_instruction=task_instruction,
            objects=objects,
            spatial_relations=spatial_relations,
            macro_intent=macro_intent,
            micro_instructions=micro_steps,
            raw_recognition="\n".join(
                prediction.raw_text for prediction in predictions
            ),
            view_id=view_mode,
            metadata={
                "source": "overlapping_macro_dense_micro",
                "bundle_id": bundle.sample_id,
            },
        )
        augmented = self.augmentation.augment(
            segment_annotation,
            num_variants=num_variants,
        )
        evidence_by_view = {
            view_id: sorted(
                {
                    frame.frame_index
                    for _, _, _, frames_by_view in annotated_windows
                    for frame in frames_by_view.get(view_id, [])
                }
            )
            for view_id in view_ids
        }
        frame_ids = sorted(
            {frame for values in evidence_by_view.values() for frame in values}
        )
        return TaskSegment(
            segment_id=f"segment_{segment_index:03d}",
            start_time_sec=start_sec,
            end_time_sec=end_sec,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_ids=frame_ids,
            task_instruction=task_instruction,
            objects=objects,
            spatial_relations=spatial_relations,
            macro_intent=macro_intent,
            micro_instructions=micro_steps,
            augmented_instructions=augmented,
            confidence=confidence,
            evidence_by_view=evidence_by_view,
            metadata={
                "micro_window_count": len(micro_windows),
                "view_mode": view_mode,
                "trajectory_linked": bool(bundle.trajectory),
                "augmentation_audit": getattr(
                    self.augmentation.rewriter,
                    "last_audit",
                    [],
                ),
                "alternative_micro_observations": [
                    [step.text for step in annotation.micro_instructions]
                    for _, _, annotation, _ in annotated_windows[1:]
                ],
            },
        )

    def run_view_bundle(
        self,
        bundle: ViewBundle,
        work_dir: str | Path,
        source_instruction: str = "",
        *,
        view_mode: str = "fused",
        num_variants: int = 3,
        macro_window_sec: float = 16.0,
        macro_step_sec: float = 8.0,
        macro_frames: int = 16,
        micro_window_sec: float = 2.0,
        micro_step_sec: float = 1.0,
        micro_frames: int = 4,
    ) -> VideoTaskRecord:
        """Run synchronized fixed/ego perception on overlapping temporal windows."""
        validation_reasons = validate_view_bundle(
            bundle,
            require_paired=view_mode == "fused",
        )
        view_ids = self._selected_views(bundle, view_mode)
        if not view_ids:
            raise ValueError("ViewBundle contains no readable view entries")
        work_path = Path(work_dir)
        macro_windows = build_temporal_windows(
            bundle.timebase.fps,
            bundle.timebase.frame_count,
            window_sec=macro_window_sec,
            step_sec=macro_step_sec,
            frames_per_window=macro_frames,
        )
        votes: list[CutVote] = []
        macro_records: list[
            tuple[Any, Prediction, PerceptionAnnotation, dict[str, list[VideoFrame]]]
        ] = []
        multi_view_frames: list[MultiViewFrameAnnotation] = []
        for window in macro_windows:
            prediction, frames_by_view = self._window_prediction(
                bundle,
                window.frame_indices,
                work_path / "macro" / f"window_{window.window_id:03d}",
                view_ids,
                source_instruction,
                level="10+ second macro-intent",
            )
            annotation = self._annotation_for_prediction(
                prediction,
                frames_by_view,
                source_instruction,
                f"{bundle.sample_id}_macro_{window.window_id:03d}",
                view_mode,
            )
            macro_records.append((window, prediction, annotation, frames_by_view))
            votes.extend(transition_votes(window, prediction.transitions))
            for frame_index in window.frame_indices:
                view_frames = {
                    view_id: frame
                    for view_id in view_ids
                    for frame in frames_by_view.get(view_id, [])
                    if frame.frame_index == frame_index
                }
                if view_frames:
                    multi_view_frames.append(
                        MultiViewFrameAnnotation(
                            frame_id=len(multi_view_frames),
                            timestamp_sec=frame_index / bundle.timebase.fps,
                            view_frames=view_frames,
                            view_annotations={},
                            fused_annotation=annotation,
                        )
                    )

        cuts = aggregate_cut_votes(
            votes,
            fps=bundle.timebase.fps,
            frame_count=bundle.timebase.frame_count,
        )
        intervals = intervals_from_cuts(
            cuts,
            frame_count=bundle.timebase.frame_count,
            fps=bundle.timebase.fps,
        )
        segments: list[TaskSegment] = []
        for idx, (start_frame, end_frame, start_sec, end_sec) in enumerate(intervals):
            overlapping_predictions = [
                prediction
                for window, prediction, _, _ in macro_records
                if window.end_frame >= start_frame and window.start_frame < end_frame
            ]
            segments.append(
                self._build_windowed_segment(
                    segment_index=idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    bundle=bundle,
                    view_ids=view_ids,
                    view_mode=view_mode,
                    source_instruction=source_instruction,
                    work_dir=work_path,
                    macro_predictions=overlapping_predictions,
                    num_variants=num_variants,
                    micro_window_sec=micro_window_sec,
                    micro_step_sec=micro_step_sec,
                    micro_frames=micro_frames,
                )
            )

        primary_path = bundle.views[view_ids[0]].video_path
        unique_multi_frames = {
            (item.timestamp_sec, tuple(sorted(item.view_frames))): item
            for item in multi_view_frames
        }
        return VideoTaskRecord(
            video_path=primary_path,
            view_bundle=bundle,
            source_instruction=source_instruction,
            multi_view_frames=list(unique_multi_frames.values()),
            task_segments=segments,
            metadata={
                "method": "paired_overlapping_macro_dense_micro_v1",
                "view_mode": view_mode,
                "macro_window_sec": macro_window_sec,
                "macro_step_sec": macro_step_sec,
                "macro_frames": macro_frames,
                "micro_window_sec": micro_window_sec,
                "micro_step_sec": micro_step_sec,
                "micro_frames": micro_frames,
                "cut_frames": cuts,
                "macro_window_count": len(macro_windows),
                "provenance": bundle.provenance.model_dump(),
                "validation_diagnostics": validation_reasons,
                "validation_enforced": False,
                "generation_model": getattr(
                    self.perception.recognizer,
                    "model",
                    type(self.perception.recognizer).__name__,
                ),
                "rewriter": type(self.augmentation.rewriter).__name__,
                "rewrite_prompt_version": getattr(
                    self.augmentation.rewriter,
                    "PROMPT_VERSION",
                    "",
                ),
            },
        )

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
        bundle = build_view_bundle(
            sample_id=Path(video_path).stem,
            views={"fixed": video_path},
        )
        return self.run_view_bundle(
            bundle,
            work_dir=work_dir,
            source_instruction=source_instruction,
            num_variants=num_variants,
            view_mode="fixed",
            macro_frames=max_frames,
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
