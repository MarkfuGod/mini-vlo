from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionQualityConfig:
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    max_jitter_ratio: float
    aggregation: str = "min"
    max_interval_cv: float = 0.25
    max_time_gap_ratio: float = 2.5


UNIT_TO_METERS = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "centimeters": 0.01,
    "centimeter": 0.01,
    "cm": 0.01,
    "millimeters": 0.001,
    "millimeter": 0.001,
    "mm": 0.001,
}


def normalize_positions_to_meters(
    positions: list[list[float]],
    spatial_unit: str,
) -> list[list[float]]:
    scale = UNIT_TO_METERS.get(spatial_unit.lower())
    if scale is None:
        raise ValueError(f"Unsupported motion spatial unit: {spatial_unit}")
    return (np.asarray(positions, dtype=np.float64) * scale).tolist()


def _safe_diff(arr: np.ndarray, dt: np.ndarray) -> np.ndarray:
    dt = np.clip(dt, 1e-6, None)
    return np.diff(arr, axis=0) / dt[:, None]


def score_motion_quality(
    positions: list[list[float]],
    timestamps: list[float],
    cfg: MotionQualityConfig,
) -> tuple[float, dict[str, float], list[str]]:
    reasons: list[str] = []
    p = np.asarray(positions, dtype=np.float32)
    t = np.asarray(timestamps, dtype=np.float32)

    if len(p) < 4 or len(t) < 4:
        return 0.0, {"valid": 0.0}, ["too_short"]
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(t)):
        return 0.0, {"valid": 0.0}, ["nan_or_inf"]
    if np.any(np.diff(t) <= 0):
        return 0.0, {"valid": 0.0}, ["non_increasing_timestamps"]

    dt = np.diff(t)
    v = _safe_diff(p, dt)
    a = _safe_diff(v, dt[1:])
    j = _safe_diff(a, dt[2:])

    v_norm = np.linalg.norm(v, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    j_norm = np.linalg.norm(j, axis=1)

    velocity_ratio = float(np.mean(v_norm > cfg.max_velocity))
    acceleration_ratio = float(np.mean(a_norm > cfg.max_acceleration))
    jerk_ratio = float(np.mean(j_norm > cfg.max_jerk))

    nonzero = v_norm > 1e-8
    unit_velocity = np.zeros_like(v)
    unit_velocity[nonzero] = v[nonzero] / v_norm[nonzero, None]
    direction_cosine = np.sum(unit_velocity[:-1] * unit_velocity[1:], axis=1)
    valid_direction_pairs = nonzero[:-1] & nonzero[1:]
    jitter_ratio = (
        float(np.mean(direction_cosine[valid_direction_pairs] < 0.0))
        if np.any(valid_direction_pairs)
        else 0.0
    )
    median_dt = float(np.median(dt))
    interval_cv = float(np.std(dt) / max(float(np.mean(dt)), 1e-6))
    time_gap_ratio = float(np.max(dt) / max(median_dt, 1e-6))

    if jitter_ratio > cfg.max_jitter_ratio:
        reasons.append("high_jitter")
    if velocity_ratio > 0.2:
        reasons.append("high_velocity_spikes")
    if acceleration_ratio > 0.2:
        reasons.append("high_acceleration_spikes")
    if jerk_ratio > 0.2:
        reasons.append("high_jerk_spikes")
    if interval_cv > cfg.max_interval_cv:
        reasons.append("irregular_frame_intervals")
    if time_gap_ratio > cfg.max_time_gap_ratio:
        reasons.append("drop_frame_or_time_shift")

    penalties = np.array(
        [
            min(1.0, velocity_ratio / 0.2),
            min(1.0, acceleration_ratio / 0.2),
            min(1.0, jerk_ratio / 0.2),
            min(1.0, jitter_ratio / max(cfg.max_jitter_ratio, 1e-6)),
            min(1.0, interval_cv / max(cfg.max_interval_cv, 1e-6)),
            min(
                1.0,
                max(0.0, time_gap_ratio - 1.0)
                / max(cfg.max_time_gap_ratio - 1.0, 1e-6),
            ),
        ],
        dtype=np.float32,
    )
    score = float(np.clip(1.0 - np.mean(penalties), 0.0, 1.0))
    details = {
        "velocity_ratio": velocity_ratio,
        "acceleration_ratio": acceleration_ratio,
        "jerk_ratio": jerk_ratio,
        "jitter_ratio": jitter_ratio,
        "interval_cv": interval_cv,
        "time_gap_ratio": time_gap_ratio,
    }
    return score, details, reasons


def score_motion_tracks(
    tracks: dict[str, tuple[list[list[float]], list[float]]],
    cfg: MotionQualityConfig,
) -> tuple[float, dict, list[str]]:
    track_results: dict[str, dict] = {}
    track_scores: list[float] = []
    reasons: list[str] = []

    for name, (positions, timestamps) in tracks.items():
        score, details, track_reasons = score_motion_quality(
            positions=positions,
            timestamps=timestamps,
            cfg=cfg,
        )
        track_scores.append(score)
        track_results[name] = {
            "score": score,
            **details,
            "reason_codes": track_reasons,
        }
        reasons.extend(f"{name}:{reason}" for reason in track_reasons)

    if not track_scores:
        return 0.0, {"valid": 0.0, "tracks": {}}, ["motion_tracks_missing"]

    aggregation = cfg.aggregation
    if aggregation == "mean":
        score = float(np.mean(track_scores))
    elif aggregation == "min":
        score = float(np.min(track_scores))
    else:
        aggregation = "min"
        score = float(np.min(track_scores))
        reasons.append("unsupported_motion_aggregation")

    aux = {
        "aggregation": aggregation,
        "track_count": float(len(track_scores)),
        "tracks": track_results,
    }
    return score, aux, reasons

