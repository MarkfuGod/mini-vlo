"""Adapter around the pinned Video2Tasks source vendored in this repository."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable


UPSTREAM_REPOSITORY = "https://github.com/ly-geming/video2tasks"
UPSTREAM_REVISION = "8d405a120a37df4e5a4869b61ba3d9edb7b4dfe3"
WINDOWING_SHA256 = "85b50d6cce12b0f56b2c5a2f0550591a17d1fb0677c591fb5ce06ff986095418"
PROMPT_SHA256 = "9e0d5fd5ebce3d75a62ac8ec470b157f12c66575f1fe95f1e58582fbeaaeb761"


def _assert_source_hash(module: Any, expected: str, component: str) -> None:
    source_file = inspect.getsourcefile(module)
    if not source_file:
        raise RuntimeError(f"Cannot verify Video2Tasks {component} source")
    actual = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Video2Tasks {component} hash mismatch: expected {expected}, got {actual}. "
            f"Install the pinned revision {UPSTREAM_REVISION}."
        )


def _windowing_module():
    try:
        module = importlib.import_module("video2tasks.server.windowing")
        _assert_source_hash(module, WINDOWING_SHA256, "windowing")
        return module
    except ImportError as exc:
        raise RuntimeError(
            "The vendored Video2Tasks package is missing or cannot be imported. "
            "Restore video2tasks/ from upstream revision "
            f"{UPSTREAM_REVISION}; install requirements-video2tasks.txt only "
            "when using the original server/worker CLIs."
        ) from exc


def upstream_prompt(n_images: int) -> str:
    """Use the prompt shipped by the same pinned upstream installation."""
    _windowing_module()
    prompt_module = importlib.import_module("video2tasks.prompt")
    _assert_source_hash(prompt_module, PROMPT_SHA256, "prompt")
    return str(prompt_module.prompt_switch_detection(n_images))


def run_upstream_video2tasks(
    video_path: str | Path,
    infer_window: Callable[[list[int], int], dict[str, Any]],
    *,
    sample_id: str,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16,
) -> dict[str, Any]:
    """Call upstream build_windows and build_segments_via_cuts unchanged."""
    windowing = _windowing_module()
    fps, frame_count = windowing.read_video_info(str(video_path))
    windows = windowing.build_windows(
        fps,
        frame_count,
        window_sec=window_sec,
        step_sec=step_sec,
        frames_per_window=frames_per_window,
    )
    by_window = {}
    for window in windows:
        by_window[window.window_id] = {
            "vlm_json": infer_window(window.frame_ids, window.window_id)
        }
    result = windowing.build_segments_via_cuts(
        sample_id,
        windows,
        by_window,
        fps,
        frame_count,
        frames_per_window=frames_per_window,
    )
    return {
        **result,
        "fps": fps,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "window_sec": window_sec,
        "step_sec": step_sec,
        "frames_per_window": frames_per_window,
        "window_count": len(windows),
    }
