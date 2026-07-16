#!/usr/bin/env python3
"""Run the vendored Video2Tasks pipeline directly with a DashScope VLM."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import OpenAI

from src.runtime_utils import (
    file_sha256,
    git_revision,
    text_sha256,
    utc_now_iso,
    write_json,
)
from src.vlm_engine import _parse_json_response
from video2tasks.prompt import prompt_switch_detection
from video2tasks.server.windowing import (
    FrameExtractor,
    build_segments_via_cuts,
    build_windows,
    read_video_info,
)


ROOT = Path(__file__).parent
UPSTREAM_REPOSITORY = "https://github.com/ly-geming/video2tasks"
UPSTREAM_REVISION = "8d405a120a37df4e5a4869b61ba3d9edb7b4dfe3"

load_dotenv(ROOT / ".env")


class WindowClient(Protocol):
    def infer(
        self,
        images_b64: list[str],
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, int]]: ...


class DashScopeWindowClient:
    """OpenAI-compatible adapter for the upstream Video2Tasks prompt."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout: float,
        max_retries: int,
        max_tokens: int,
    ) -> None:
        resolved_key = api_key or os.getenv(
            "DASHSCOPE_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        )
        if not resolved_key:
            raise RuntimeError(
                "Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key"
            )
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        self.model = model or os.getenv("VLM_MODEL", "qwen3-vl-flash")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.max_tokens = int(max_tokens)
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def infer(
        self,
        images_b64: list[str],
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, int]]:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
            for encoded in images_b64
        ]
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return (
            _parse_json_response(raw),
            raw,
            {
                "requests": 1,
                "input_tokens": int(
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "input_tokens", 0)
                    or 0
                ),
                "output_tokens": int(
                    getattr(usage, "completion_tokens", 0)
                    or getattr(usage, "output_tokens", 0)
                    or 0
                ),
            },
        )


def normalize_window_output(
    parsed: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[dict[str, Any], list[str]]:
    """Make provider JSON safe for the unchanged upstream aggregator."""
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("VLM response did not contain a JSON object")

    raw_transitions = parsed.get("transitions", [])
    raw_instructions = parsed.get("instructions", [])
    if not isinstance(raw_transitions, list):
        raise ValueError("'transitions' must be a list")
    if not isinstance(raw_instructions, list):
        raise ValueError("'instructions' must be a list")

    transitions: list[int] = []
    for value in raw_transitions:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < frame_count:
            transitions.append(index)
    transitions = sorted(set(transitions))

    instructions = [
        str(value).strip()
        for value in raw_instructions
        if str(value).strip()
    ]
    required = len(transitions) + 1
    if len(instructions) < required:
        raise ValueError(
            f"VLM returned {len(transitions)} transitions but only "
            f"{len(instructions)} instructions"
        )

    warnings: list[str] = []
    if len(instructions) > required:
        overflow = instructions[required - 1 :]
        instructions = [
            *instructions[: required - 1],
            "; ".join(overflow),
        ]
        warnings.append(
            f"merged {len(overflow) - 1} extra instruction(s) into the final segment"
        )

    return {
        **parsed,
        "transitions": transitions,
        "instructions": instructions,
    }, warnings


def _add_usage(total: dict[str, int], current: dict[str, int]) -> None:
    for key in ("requests", "input_tokens", "output_tokens"):
        total[key] += int(current.get(key, 0))


def _select_views(views: dict[str, Path], view_mode: str) -> list[str]:
    if not views:
        raise ValueError("At least one video view is required")
    if view_mode == "fused":
        selected = [name for name in ("fixed", "ego") if name in views]
        return selected or sorted(views)
    if view_mode not in views:
        raise ValueError(
            f"Requested view '{view_mode}' is unavailable; "
            f"available views: {sorted(views)}"
        )
    return [view_mode]


def _multiview_prompt(timestamp_count: int, view_ids: list[str]) -> str:
    prompt = prompt_switch_detection(timestamp_count)
    if len(view_ids) == 1:
        return (
            prompt
            + f"\n\nThe images come from the `{view_ids[0]}` camera. "
            "Transition indices refer to ordered timestamps."
        )
    order = " then ".join(view_ids)
    return (
        prompt
        + "\n\n### Multi-view input\n"
        + f"There are {timestamp_count} synchronized timestamps. At each "
        + f"timestamp, images are ordered {order}. Transition indices refer "
        + "to timestamps 0 through "
        + f"{timestamp_count - 1}, never to individual image positions. "
        + "Use the fixed view for global geometry and the ego view for "
        + "grasp/contact evidence. Do not count a camera change as a task switch."
    )


def run_views(
    view_paths: dict[str, str | Path],
    *,
    sample_id: str,
    client: WindowClient,
    view_mode: str = "fused",
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16,
    target_width: int = 720,
    target_height: int = 480,
    png_compression: int = 3,
    window_retries: int = 1,
) -> dict[str, Any]:
    """Run upstream Video2Tasks with fixed, ego, or synchronized fused input."""
    views = {str(name): Path(path) for name, path in view_paths.items()}
    selected_view_ids = _select_views(views, view_mode)
    selected_views = {name: views[name] for name in selected_view_ids}
    for name, video in selected_views.items():
        if not video.is_file():
            raise FileNotFoundError(f"Video view not found ({name}): {video}")

    view_info = {
        name: read_video_info(str(video))
        for name, video in selected_views.items()
    }
    fps, frame_count = view_info[selected_view_ids[0]]
    for name in selected_view_ids[1:]:
        current_fps, current_frames = view_info[name]
        if abs(current_fps - fps) > 1e-3:
            raise ValueError(
                f"Synchronized views have different FPS: "
                f"{selected_view_ids[0]}={fps}, {name}={current_fps}"
            )
        if current_frames != frame_count:
            raise ValueError(
                f"Synchronized views have different frame counts: "
                f"{selected_view_ids[0]}={frame_count}, {name}={current_frames}"
            )

    windows = build_windows(
        fps,
        frame_count,
        window_sec=window_sec,
        step_sec=step_sec,
        frames_per_window=frames_per_window,
    )
    if not windows:
        raise ValueError(f"Video views contain no readable frames: {selected_views}")

    by_window: dict[int, dict[str, Any]] = {}
    window_outputs: list[dict[str, Any]] = []
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    started = time.monotonic()

    with ExitStack() as stack:
        extractors = {
            name: stack.enter_context(FrameExtractor(str(selected_views[name])))
            for name in selected_view_ids
        }
        for window in windows:
            images_by_view = {
                name: extractors[name].get_many_b64(
                    window.frame_ids,
                    target_w=target_width,
                    target_h=target_height,
                    compression=png_compression,
                )
                for name in selected_view_ids
            }
            for name, images in images_by_view.items():
                if len(images) != len(window.frame_ids) or any(
                    not image for image in images
                ):
                    raise RuntimeError(
                        f"Failed to extract all {name} frames for "
                        f"window {window.window_id}"
                    )
            images = [
                images_by_view[name][timestamp_index]
                for timestamp_index in range(len(window.frame_ids))
                for name in selected_view_ids
            ]

            prompt = _multiview_prompt(len(window.frame_ids), selected_view_ids)
            error = ""
            parsed: dict[str, Any] = {}
            raw = ""
            warnings: list[str] = []
            for attempt in range(max(0, window_retries) + 1):
                try:
                    candidate, raw, current_usage = client.infer(images, prompt)
                    _add_usage(usage, current_usage)
                    parsed, warnings = normalize_window_output(
                        candidate,
                        frame_count=len(window.frame_ids),
                    )
                    error = ""
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt >= max(0, window_retries):
                        raise RuntimeError(
                            f"Window {window.window_id} failed after "
                            f"{attempt + 1} attempt(s): {error}"
                        ) from exc

            by_window[window.window_id] = {"vlm_json": parsed}
            window_outputs.append(
                {
                    "window_id": window.window_id,
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "frame_ids": window.frame_ids,
                    "view_ids": selected_view_ids,
                    "images_per_timestamp": len(selected_view_ids),
                    "vlm_json": parsed,
                    "warnings": warnings,
                    "raw_response": raw,
                    "error": error,
                }
            )
            print(
                f"  window {window.window_id + 1}/{len(windows)} "
                f"cuts={parsed.get('transitions', [])}",
                flush=True,
            )

    prediction = build_segments_via_cuts(
        sample_id,
        windows,
        by_window,
        fps,
        frame_count,
        frames_per_window=frames_per_window,
    )
    return {
        "sample_id": sample_id,
        "view_mode": view_mode,
        "views": {
            name: str(video.resolve())
            for name, video in selected_views.items()
        },
        "view_sha256": {
            name: file_sha256(video)
            for name, video in selected_views.items()
        },
        "fps": fps,
        "frame_count": frame_count,
        "window_count": len(windows),
        "latency_s": time.monotonic() - started,
        "usage": usage,
        "prediction": prediction,
        "window_outputs": window_outputs,
        "error": "",
    }


def run_video(
    video_path: str | Path,
    *,
    sample_id: str,
    client: WindowClient,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16,
    target_width: int = 720,
    target_height: int = 480,
    png_compression: int = 3,
    window_retries: int = 1,
) -> dict[str, Any]:
    """Backward-compatible single-video wrapper."""
    return run_views(
        {"fixed": video_path},
        sample_id=sample_id,
        client=client,
        view_mode="fixed",
        window_sec=window_sec,
        step_sec=step_sec,
        frames_per_window=frames_per_window,
        target_width=target_width,
        target_height=target_height,
        png_compression=png_compression,
        window_retries=window_retries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the vendored Video2Tasks pipeline directly with DashScope; "
            "no server, worker, GPU, or external video2tasks install required."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Single MP4 input")
    source.add_argument(
        "--manifest",
        help=(
            "Single ViewBundle or {'samples': [...]} with video or "
            "views.fixed/views.ego fields"
        ),
    )
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--view-mode",
        choices=["fixed", "ego", "fused"],
        default="fused",
        help="Use one camera or timestamp-aligned fixed+ego evidence.",
    )
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", "qwen3-vl-flash"))
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL"))
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("VLM_TIMEOUT", "300")),
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--window-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--window-sec", type=float, default=16.0)
    parser.add_argument("--step-sec", type=float, default=8.0)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--target-width", type=int, default=384)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output",
        default="",
        help="Aggregate JSON output path",
    )
    return parser.parse_args()


def _resolve_manifest_video(value: str, manifest: Path) -> str:
    video = Path(value)
    if video.is_absolute():
        return str(video)
    candidate = manifest.parent / video
    return str(candidate if candidate.exists() else ROOT / video)


def load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path | None]:
    if args.video:
        video = Path(args.video)
        return [{"id": video.stem, "views": {"fixed": str(video)}}], None

    manifest = Path(args.manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and "timebase" in payload
        and "views" in payload
    ):
        rows = [payload]
    else:
        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list or contain a 'samples' list")

    wanted = set(args.sample_id)
    samples: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id_value = row.get("sample_id", row.get("id"))
        if sample_id_value is None:
            continue
        sample_id = str(sample_id_value)
        if wanted and sample_id not in wanted:
            continue
        raw_views = row.get("views")
        views: dict[str, str] = {}
        if isinstance(raw_views, dict):
            for name, value in raw_views.items():
                if isinstance(value, dict):
                    value = value.get("video_path")
                if value:
                    views[str(name)] = _resolve_manifest_video(
                        str(value),
                        manifest,
                    )
        elif row.get("video"):
            views["fixed"] = _resolve_manifest_video(
                str(row["video"]),
                manifest,
            )
        if views:
            samples.append({"id": sample_id, "views": views})

    if wanted:
        missing = sorted(wanted - {sample["id"] for sample in samples})
        if missing:
            raise ValueError(f"Unknown --sample-id value(s): {missing}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No samples selected")
    return samples, manifest


def _result_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", ""))


def main() -> None:
    args = parse_args()
    if args.frames_per_window < 1:
        raise ValueError("--frames-per-window must be >= 1")
    if args.window_sec <= 0 or args.step_sec <= 0:
        raise ValueError("--window-sec and --step-sec must be > 0")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")

    samples, manifest = load_samples(args)
    if args.output:
        output = Path(args.output)
    elif args.video:
        output = ROOT / "results" / f"video2tasks_{samples[0]['id']}.json"
    else:
        output = ROOT / "results" / "video2tasks_manifest.json"

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        existing = {
            _result_key(row): row
            for row in previous.get("results", [])
            if isinstance(row, dict) and _result_key(row)
        }
    results = list(existing.values())

    client = DashScopeWindowClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )

    def save() -> None:
        ordered = {sample["id"]: index for index, sample in enumerate(samples)}
        results.sort(key=lambda row: ordered.get(_result_key(row), len(ordered)))
        succeeded = sum(not row.get("error") for row in results)
        write_json(
            output,
            {
                "schema_version": "vendored-video2tasks-run/v2",
                "formal": False,
                "generated_at": utc_now_iso(),
                "code_revision": git_revision(ROOT),
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_revision": UPSTREAM_REVISION,
                "vendored_source": "video2tasks/",
                "model": client.model,
                "base_url": client.base_url,
                "prompt_sha256": text_sha256(
                    prompt_switch_detection(args.frames_per_window)
                ),
                "input": {
                    "manifest": str(manifest) if manifest else None,
                    "manifest_sha256": file_sha256(manifest) if manifest else None,
                    "requested_samples": len(samples),
                    "view_mode": args.view_mode,
                },
                "windowing": {
                    "window_sec": args.window_sec,
                    "step_sec": args.step_sec,
                    "frames_per_window": args.frames_per_window,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "png_compression": args.png_compression,
                },
                "summary": {
                    "completed": len(results),
                    "succeeded": succeeded,
                    "failed": len(results) - succeeded,
                },
                "limitations": [
                    "generic VLM; not a robot-domain world model",
                    "output quality depends on provider vision grounding",
                ],
                "results": results,
            },
        )

    for index, sample in enumerate(samples, start=1):
        previous = existing.get(sample["id"])
        if previous and not previous.get("error"):
            print(f"[{index}/{len(samples)}] {sample['id']} already complete")
            continue
        print(f"[{index}/{len(samples)}] {sample['id']}", flush=True)
        try:
            row = run_views(
                sample["views"],
                sample_id=sample["id"],
                client=client,
                view_mode=args.view_mode,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                frames_per_window=args.frames_per_window,
                target_width=args.target_width,
                target_height=args.target_height,
                png_compression=args.png_compression,
                window_retries=args.window_retries,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            row = {
                "sample_id": sample["id"],
                "view_mode": args.view_mode,
                "views": {
                    name: str(Path(path).resolve())
                    for name, path in sample["views"].items()
                },
                "prediction": {},
                "window_outputs": [],
                "usage": {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "latency_s": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results = [
            current for current in results if _result_key(current) != sample["id"]
        ]
        results.append(row)
        save()
        print(
            f"  segments={len(row.get('prediction', {}).get('segments', []))} "
            f"error={row.get('error', '') or 'none'}",
            flush=True,
        )

    save()
    print(f"Saved Video2Tasks output to {output}")


if __name__ == "__main__":
    main()
