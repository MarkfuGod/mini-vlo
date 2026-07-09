from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .motion_quality import MotionQualityConfig, score_motion_tracks
from .schema import MotionData, MotionTrack, RefinementResult, Sample
from .semantic_consistency import (
    SemanticConfig,
    build_verifier,
    score_semantic_consistency,
)


@dataclass
class RefinementConfig:
    motion_min_score: float
    semantic_min_score: float
    motion_cfg: MotionQualityConfig
    semantic_cfg: SemanticConfig


def load_config(path: str | Path) -> RefinementConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RefinementConfig(
        motion_min_score=float(raw["thresholds"]["motion_min"]),
        semantic_min_score=float(raw["thresholds"]["semantic_min"]),
        motion_cfg=MotionQualityConfig(**raw["motion_quality"]),
        semantic_cfg=SemanticConfig(**raw["semantic"]),
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
        return MotionData(tracks=tracks) if tracks else None

    positions = motion_obj.get("positions")
    timestamps = motion_obj.get("timestamps")
    if positions is not None and timestamps is not None:
        return MotionData(
            tracks={
                "default": MotionTrack(
                    positions=positions,
                    timestamps=timestamps,
                )
            }
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
        if sample.motion is None:
            motion_score = None
            motion_aux = {"valid": 0.0, "missing": 1.0}
            motion_reasons = ["motion_missing"]
        else:
            motion_score, motion_aux, motion_reasons = score_motion_tracks(
                tracks={
                    name: (track.positions, track.timestamps)
                    for name, track in sample.motion.tracks.items()
                },
                cfg=cfg.motion_cfg,
            )

        semantic_score, semantic_aux, semantic_reasons = score_semantic_consistency(
            video_path=sample.video_path,
            text=sample.text,
            verifier=verifier,
            cfg=cfg.semantic_cfg,
        )

        is_motion_low = (
            motion_score is not None and motion_score < cfg.motion_min_score
        )
        is_semantic_low = semantic_score < cfg.semantic_min_score
        if motion_score is None:
            final_score = semantic_score
            decision = "drop" if is_semantic_low else "keep"
        else:
            final_score = min(motion_score, semantic_score)
            decision = "drop" if (is_motion_low or is_semantic_low) else "keep"

        threshold_reasons: list[str] = []
        if is_motion_low:
            threshold_reasons.append("low_motion_score")
        if is_semantic_low:
            threshold_reasons.append("low_semantic_score")
        if motion_score is None:
            threshold_reasons.append("semantic_only_mode")

        results.append(
            RefinementResult(
                sample_id=sample.sample_id,
                motion_quality_score=motion_score,
                semantic_consistency_score=semantic_score,
                final_score=final_score,
                decision=decision,
                reason_codes=sorted(
                    set(motion_reasons + semantic_reasons + threshold_reasons)
                ),
                aux={
                    "motion": motion_aux,
                    "semantic": semantic_aux,
                    "text": sample.text,
                    "label": sample.label,
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
        "semantic_consistency_score": result.semantic_consistency_score,
        "final_score": (
            float(result.final_score) if math.isfinite(result.final_score) else 0.0
        ),
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

