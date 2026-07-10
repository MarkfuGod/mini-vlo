"""Overlapping temporal windows and weighted task-boundary aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    window_id: int
    start_frame: int
    end_frame: int
    frame_indices: list[int]


@dataclass(frozen=True)
class CutVote:
    frame_index: int
    weight: float
    window_id: int


def build_temporal_windows(
    fps: float,
    frame_count: int,
    *,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16,
) -> list[TemporalWindow]:
    """Match Video2Tasks' overlapping-window policy without its server layer."""
    if fps <= 0 or frame_count <= 0:
        return []
    window_frames = max(1, int(round(window_sec * fps)))
    step_frames = max(1, int(round(step_sec * fps)))
    sample_count = max(1, frames_per_window)
    windows: list[TemporalWindow] = []
    start = 0
    while start < frame_count:
        end = min(frame_count - 1, start + window_frames - 1)
        if windows and end - start < window_frames // 2:
            break
        indices = (
            np.linspace(start, end, num=sample_count).astype(int).clip(0, frame_count - 1)
        )
        windows.append(
            TemporalWindow(
                window_id=len(windows),
                start_frame=start,
                end_frame=end,
                frame_indices=indices.tolist(),
            )
        )
        start += step_frames
    return windows


def transition_votes(
    window: TemporalWindow,
    transition_indices: list[int],
) -> list[CutVote]:
    """Map local transition indices to global frames with Hanning weights."""
    sample_count = len(window.frame_indices)
    if sample_count == 0:
        return []
    weights = np.hanning(sample_count + 2)[1:-1]
    votes: list[CutVote] = []
    for transition in transition_indices:
        try:
            local_index = int(transition)
        except (TypeError, ValueError):
            continue
        if 0 <= local_index < sample_count:
            votes.append(
                CutVote(
                    frame_index=window.frame_indices[local_index],
                    weight=float(weights[local_index]),
                    window_id=window.window_id,
                )
            )
    return votes


def aggregate_cut_votes(
    votes: list[CutVote],
    *,
    fps: float,
    frame_count: int,
    cluster_gap_sec: float = 2.5,
    min_segment_sec: float = 0.8,
) -> list[int]:
    """Cluster weighted votes and return valid interior segment boundaries."""
    if frame_count <= 1:
        return []
    if not votes:
        return []
    cluster_gap = max(1, int(round(cluster_gap_sec * fps)))
    sorted_votes = sorted(votes, key=lambda vote: vote.frame_index)
    clusters: list[list[CutVote]] = [[sorted_votes[0]]]
    for vote in sorted_votes[1:]:
        if vote.frame_index - clusters[-1][-1].frame_index < cluster_gap:
            clusters[-1].append(vote)
        else:
            clusters.append([vote])

    candidates: list[int] = []
    for cluster in clusters:
        frames = np.asarray([vote.frame_index for vote in cluster], dtype=np.float64)
        weights = np.asarray([vote.weight for vote in cluster], dtype=np.float64)
        if float(weights.sum()) > 1e-9:
            candidate = int(round(float(np.average(frames, weights=weights))))
        else:
            candidate = int(round(float(frames.mean())))
        if 0 < candidate < frame_count:
            candidates.append(candidate)

    min_frames = max(1, int(round(min_segment_sec * fps)))
    accepted: list[int] = []
    previous = 0
    for candidate in sorted(set(candidates)):
        if candidate - previous < min_frames:
            continue
        if frame_count - candidate < min_frames:
            continue
        accepted.append(candidate)
        previous = candidate
    return accepted


def intervals_from_cuts(
    cuts: list[int],
    *,
    frame_count: int,
    fps: float,
) -> list[tuple[int, int, float, float]]:
    """Convert cut frames into half-open frame and second intervals."""
    if frame_count <= 0 or fps <= 0:
        return []
    boundaries = [0, *sorted(cut for cut in set(cuts) if 0 < cut < frame_count), frame_count]
    return [
        (start, end, start / fps, end / fps)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]


def build_micro_windows(
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    frames_per_window: int = 4,
) -> list[TemporalWindow]:
    """Create dense 1–3 second windows inside one macro task segment."""
    frame_count = max(0, end_frame - start_frame)
    local_windows = build_temporal_windows(
        fps,
        frame_count,
        window_sec=window_sec,
        step_sec=step_sec,
        frames_per_window=frames_per_window,
    )
    return [
        TemporalWindow(
            window_id=window.window_id,
            start_frame=start_frame + window.start_frame,
            end_frame=start_frame + window.end_frame,
            frame_indices=[start_frame + index for index in window.frame_indices],
        )
        for window in local_windows
    ]
