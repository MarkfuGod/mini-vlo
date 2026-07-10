from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from .motion_quality import (
    MotionQualityConfig,
    normalize_positions_to_meters,
    score_motion_tracks,
)
from .schema import MotionData, MotionTrack, RefinementResult, Sample
from .semantic_consistency import (
    SemanticConfig,
    build_verifier,
    verify_multiview_semantic_consistency,
    verify_semantic_consistency,
)
from .sync_checks import SyncConfig, check_sample_sync


@dataclass
class RefinementConfig:
    motion_min_score: float
    motion_cfg: MotionQualityConfig
    semantic_cfg: SemanticConfig
    semantic_min_confidence: float = 0.7
    allow_mock_keep: bool = False
    sync_cfg: SyncConfig = field(default_factory=SyncConfig)


def load_config(path: str | Path) -> RefinementConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    thresholds = raw["thresholds"]
    semantic_raw = dict(raw["semantic"])
    semantic_raw.pop("low_confidence_penalty", None)
    semantic_min_confidence = float(
        thresholds.get("semantic_min_confidence", 0.7)
    )
    return RefinementConfig(
        motion_min_score=float(thresholds["motion_min"]),
        motion_cfg=MotionQualityConfig(**raw["motion_quality"]),
        semantic_cfg=SemanticConfig(**semantic_raw),
        semantic_min_confidence=semantic_min_confidence,
        allow_mock_keep=bool(raw.get("policy", {}).get("allow_mock_keep", False)),
        sync_cfg=SyncConfig(**raw.get("sync", {})),
    )


def _load_motion_data(motion_obj: object) -> MotionData | None:
    if not isinstance(motion_obj, dict):
        return None

    tracks_obj = motion_obj.get("tracks")
    if isinstance(tracks_obj, dict):
        tracks = {}
        for name, track_obj in tracks_obj.items():
            if not isinstance(track_obj, dict):
                continue
            positions = track_obj.get("positions")
            timestamps = track_obj.get("timestamps")
            if positions is not None and timestamps is not None:
                tracks[str(name)] = MotionTrack(
                    positions=positions,
                    timestamps=timestamps,
                )
        return (
            MotionData(
                tracks=tracks,
                spatial_unit=str(motion_obj.get("spatial_unit", "meters")),
                coordinate_frame=str(motion_obj.get("coordinate_frame", "world")),
                source=str(motion_obj.get("source", "unknown")),
                is_dummy=bool(motion_obj.get("is_dummy", False)),
            )
            if tracks
            else None
        )

    positions = motion_obj.get("positions")
    timestamps = motion_obj.get("timestamps")
    if positions is not None and timestamps is not None:
        return MotionData(
            tracks={
                "default": MotionTrack(
                    positions=positions,
                    timestamps=timestamps,
                )
            },
            spatial_unit=str(motion_obj.get("spatial_unit", "meters")),
            coordinate_frame=str(motion_obj.get("coordinate_frame", "world")),
            source=str(motion_obj.get("source", "unknown")),
            is_dummy=bool(motion_obj.get("is_dummy", False)),
        )
    return None


def load_samples(jsonl_path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            motion = _load_motion_data(obj.get("motion"))
            samples.append(
                Sample(
                    sample_id=obj["sample_id"],
                    video_path=obj["video_path"],
                    text=obj["text"],
                    motion=motion,
                    label=obj.get("label"),
                    views={
                        str(key): str(value)
                        for key, value in obj.get("views", {}).items()
                    },
                    fps=(
                        float(obj["fps"]) if obj.get("fps") is not None else None
                    ),
                    frame_count=(
                        int(obj["frame_count"])
                        if obj.get("frame_count") is not None
                        else None
                    ),
                    segment_start_sec=(
                        float(obj["segment_start_sec"])
                        if obj.get("segment_start_sec") is not None
                        else None
                    ),
                    segment_end_sec=(
                        float(obj["segment_end_sec"])
                        if obj.get("segment_end_sec") is not None
                        else None
                    ),
                    provenance=dict(obj.get("provenance", {})),
                )
            )
    return samples


def refine_samples(
    samples: Iterable[Sample],
    cfg: RefinementConfig,
) -> list[RefinementResult]:
    verifier = build_verifier(cfg.semantic_cfg)
    results: list[RefinementResult] = []
    for sample in samples:
        sync_valid, sync_aux, sync_reasons = check_sample_sync(sample, cfg.sync_cfg)
        if sample.motion is None:
            motion_score = None
            motion_aux = {"valid": 0.0, "missing": 1.0}
            motion_reasons = ["motion_missing"]
        else:
            try:
                normalized_tracks = {
                    name: (
                        normalize_positions_to_meters(
                            track.positions,
                            sample.motion.spatial_unit,
                        ),
                        track.timestamps,
                    )
                    for name, track in sample.motion.tracks.items()
                }
                motion_score, motion_aux, motion_reasons = score_motion_tracks(
                    tracks=normalized_tracks,
                    cfg=cfg.motion_cfg,
                )
                motion_aux["input_spatial_unit"] = sample.motion.spatial_unit
                motion_aux["normalized_spatial_unit"] = "meters"
            except ValueError:
                motion_score = 0.0
                motion_aux = {"valid": 0.0}
                motion_reasons = ["motion_unit_unsupported"]

        semantic_text = sample.text
        if (
            sample.segment_start_sec is not None
            and sample.segment_end_sec is not None
        ):
            semantic_text = (
                f"Evaluate only video interval {sample.segment_start_sec:.3f}s-"
                f"{sample.segment_end_sec:.3f}s. Candidate text: {sample.text}"
            )
        if sample.views:
            semantic_aux, semantic_reasons = verify_multiview_semantic_consistency(
                video_paths=sample.views,
                text=semantic_text,
                verifier=verifier,
            )
        else:
            semantic_aux, semantic_reasons = verify_semantic_consistency(
                video_path=sample.video_path,
                text=semantic_text,
                verifier=verifier,
            )

        is_motion_low = motion_score is None or motion_score < cfg.motion_min_score
        semantic_label = str(semantic_aux.get("label", "uncertain"))
        try:
            semantic_confidence = float(semantic_aux.get("confidence"))
        except (TypeError, ValueError):
            semantic_confidence = None
        is_semantic_mismatch = semantic_label != "consistent"
        is_semantic_low_confidence = (
            semantic_confidence is None
            or semantic_confidence < cfg.semantic_min_confidence
        )
        is_mock_forbidden = (
            not cfg.allow_mock_keep
            and (
                str(semantic_aux.get("verifier", "")) == "mock"
                or any(
                    str(item.get("verifier", "")) == "mock"
                    for item in (
                        semantic_aux.get("per_view", {}).values()
                        if isinstance(semantic_aux.get("per_view"), dict)
                        else []
                    )
                )
            )
        )
        decision = "keep"

        threshold_reasons: list[str] = []
        if is_motion_low:
            threshold_reasons.append("low_motion_score")
        if is_semantic_mismatch:
            threshold_reasons.append("semantic_not_consistent")
        if is_semantic_low_confidence:
            threshold_reasons.append("semantic_confidence_below_threshold")
        if is_mock_forbidden:
            threshold_reasons.append("mock_verifier_forbidden")
        threshold_reasons.append("quality_gates_disabled")

        results.append(
            RefinementResult(
                sample_id=sample.sample_id,
                motion_quality_score=motion_score,
                semantic_label=semantic_label,
                semantic_confidence=semantic_confidence,
                decision=decision,
                reason_codes=sorted(
                    set(
                        sync_reasons
                        + motion_reasons
                        + semantic_reasons
                        + threshold_reasons
                    )
                ),
                aux={
                    "sync": sync_aux,
                    "motion": motion_aux,
                    "semantic": semantic_aux,
                    "text": sample.text,
                    "label": sample.label,
                    "views": sample.views,
                    "provenance": sample.provenance,
                },
            )
        )
    return results


def result_to_dict(result: RefinementResult) -> dict:
    return {
        "sample_id": result.sample_id,
        "motion_quality_score": (
            None
            if result.motion_quality_score is None
            else float(result.motion_quality_score)
        ),
        "semantic_label": result.semantic_label,
        "semantic_confidence": result.semantic_confidence,
        "decision": result.decision,
        "reason_codes": result.reason_codes,
        "aux": result.aux,
    }


def save_results(
    results: Iterable[RefinementResult],
    output_path: str | Path,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")


def save_results_pretty(
    results: Iterable[RefinementResult],
    output_path: str | Path,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [result_to_dict(result) for result in results],
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

