"""Controlled motion corruptions for quality-filter calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptedMotion:
    positions: list[list[float]]
    timestamps: list[float]
    corruption_type: str
    severity: float


def inject_motion_corruption(
    positions: list[list[float]],
    timestamps: list[float],
    corruption_type: str,
    *,
    severity: float = 1.0,
    seed: int = 0,
) -> CorruptedMotion:
    """Inject a reproducible anomaly while retaining a parseable trajectory."""
    points = np.asarray(positions, dtype=np.float64).copy()
    times = np.asarray(timestamps, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    kind = corruption_type.lower()
    scale = max(float(np.std(points)), 1e-3)

    if kind == "jitter":
        alternating = np.where(np.arange(len(points)) % 2 == 0, 1.0, -1.0)
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        points += alternating[:, None] * direction * scale * severity
    elif kind == "spike":
        index = max(1, min(len(points) - 2, len(points) // 2))
        points[index] += np.array([1.0, -0.7, 0.5]) * scale * 10.0 * severity
    elif kind == "drop_frame":
        if len(times) >= 5:
            index = len(times) // 2
            gap = np.median(np.diff(times)) * max(3.0, 3.0 * severity)
            times[index:] += gap
    elif kind == "time_shift":
        if len(times) >= 5:
            index = len(times) // 2
            times[index:] += max(0.05, severity)
    elif kind == "nan":
        points[len(points) // 2, 0] = np.nan
    elif kind == "clean":
        pass
    else:
        raise ValueError(f"Unsupported corruption type: {corruption_type}")

    return CorruptedMotion(
        positions=points.tolist(),
        timestamps=times.tolist(),
        corruption_type=kind,
        severity=severity,
    )
