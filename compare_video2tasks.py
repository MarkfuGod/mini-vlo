#!/usr/bin/env python3
"""Run an honest prompt ablation inside pinned Video2Tasks aggregation.

This script intentionally does not call the full Semantic-Motion
``VideoTaskPipeline``. It compares the upstream Video2Tasks prompt with a blind
Semantic-Motion prompt while holding the model, frame windows, and aggregation
constant. The output labels that limitation explicitly.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import cv2
from openai import OpenAI

from src.baselines import (
    UPSTREAM_REVISION,
    run_upstream_video2tasks,
    upstream_prompt,
)
from src.evaluation.metrics import (
    boundary_metrics,
    labeled_segment_metrics,
    match_temporal_segments,
    normalized_edit_score,
    paired_bootstrap_ci,
    segmental_metrics,
    slot_f1,
    temporal_iou,
    token_f1,
)
from src.runtime_utils import (
    file_sha256,
    git_revision,
    text_sha256,
    utc_now_iso,
    write_json,
)
from src.vlm_engine import _parse_json_response
from run_video2tasks import normalize_window_output


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "results" / "video2tasks_prompt_ablation.json"

SEMANTIC_MOTION_PROMPT = """You are a video-to-task semantic engine for robot manipulation demonstrations.
Analyze the ordered video frames without using filenames or a closed task list.

Return ONLY valid JSON:
{
  "transitions": [],
  "instructions": ["one evidence-grounded label per temporal segment"],
  "instruction": "one concise whole-window task label",
  "task_type": "concise open-vocabulary task type",
  "objects": ["visible task-relevant objects"],
  "target_object": "primary object being manipulated",
  "destination": "goal object/location, or null",
  "action_sequence": ["ordered primitive manipulation steps"],
  "confidence": 0.0
}

Prefer concrete objects over generic labels. Use the full temporal sequence:
what changes from early frames to late frames matters more than one frame.

Temporal output rules:
- `transitions` MUST contain integer image indices only, never text.
- If there are K transitions, `instructions` MUST contain exactly K+1 labels.
- If the window contains one continuous task, return `transitions: []` and
  exactly one instruction in `instructions`.
- Do not split approach, minor pose adjustment, or retreat unless a completed
  manipulation event changes the object/world state.
"""

SEMANTIC_JUDGE_PROMPT = """You evaluate robot subtask labels.
For each indexed pair, decide whether the predicted label describes the same
completed manipulation event as the gold label. Synonyms are valid. Reject
wrong action, object, source/destination/direction, or hallucinated events.
Also extract concise action, object, and destination slots from each label.

Return ONLY JSON:
{"judgments":[{"index":0,"semantic_match":true,
"predicted_slots":{"action":"","object":"","destination":""},
"gold_slots":{"action":"","object":"","destination":""}}]}
"""


# Backward-compatible diagnostic sample set. It is author-authored rather than
# independently adjudicated and has no temporal gold; outputs using it are
# always marked non-formal.
DIAGNOSTIC_SAMPLES: list[dict[str, Any]] = [
    {
        "id": "close_drawer",
        "video": "demos/video2tasks_compare/close_drawer.mp4",
        "instruction": "Close the drawer.",
        "target_object": "drawer",
        "actions": ["reach drawer handle", "push drawer inward", "close drawer"],
    },
    {
        "id": "get_napkin",
        "video": "demos/video2tasks_compare/get_napkin.mp4",
        "instruction": "Get the napkin.",
        "target_object": "napkin",
        "actions": ["reach napkin", "grasp napkin", "lift napkin"],
    },
    {
        "id": "hang_towel",
        "video": "demos/video2tasks_compare/hang_towel.mp4",
        "instruction": "Hang the towel.",
        "target_object": "towel",
        "actions": ["grasp towel", "move towel to rack", "hang towel"],
    },
    {
        "id": "measure_apple",
        "video": "demos/video2tasks_compare/measure_apple.mp4",
        "instruction": "Measure the apple.",
        "target_object": "apple",
        "actions": [
            "bring measuring tool to apple",
            "align tool with apple",
            "measure apple",
        ],
    },
    {
        "id": "open_bottle",
        "video": "demos/video2tasks_compare/open_bottle.mp4",
        "instruction": "Open the bottle.",
        "target_object": "bottle",
        "actions": ["grasp bottle", "twist cap", "open bottle"],
    },
    {
        "id": "sweep_trash",
        "video": "demos/video2tasks_compare/sweep_trash.mp4",
        "instruction": "Sweep the trash into the dustpan.",
        "target_object": "trash",
        "actions": [
            "move broom to trash",
            "sweep trash",
            "collect trash in dustpan",
        ],
    },
    {
        "id": "take_out_toaster",
        "video": "demos/video2tasks_compare/take_out_toaster.mp4",
        "instruction": "Take out the toaster and put it on the wooden plate.",
        "target_object": "toaster",
        "actions": [
            "grasp toaster",
            "lift toaster",
            "move toaster to wooden plate",
            "place toaster",
        ],
    },
    {
        "id": "turn_on_lamp",
        "video": "demos/video2tasks_compare/turn_on_lamp.mp4",
        "instruction": "Turn on the lamp.",
        "target_object": "lamp",
        "actions": ["reach lamp switch", "press switch", "turn on lamp"],
    },
    {
        "id": "unplug_charger",
        "video": "demos/video2tasks_compare/unplug_charger.mp4",
        "instruction": "Unplug the charger.",
        "target_object": "charger",
        "actions": [
            "grasp charger",
            "pull charger from outlet",
            "unplug charger",
        ],
    },
    {
        "id": "sort_trash_to_tray",
        "video": "demos/video2tasks_compare/sort_trash_to_tray.mp4",
        "instruction": "Sort the trash to the tray.",
        "target_object": "trash",
        "actions": [
            "pick up trash",
            "move trash to tray",
            "place trash in tray",
        ],
    },
]


def build_semantic_motion_prompt(
    sample: dict[str, Any] | None = None,
    *,
    n_images: int | None = None,
) -> str:
    """Return a blind prompt; sample metadata is deliberately ignored."""
    del sample
    if n_images is None:
        return SEMANTIC_MOTION_PROMPT
    return (
        SEMANTIC_MOTION_PROMPT
        + f"\nThe input contains {n_images} images indexed 0 through "
        f"{max(0, n_images - 1)}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare upstream and Semantic-Motion prompts under pinned "
            "Video2Tasks windowing/aggregation. This is not a full-system test."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["prompt-ablation", "full-upstream"],
        default="prompt-ablation",
        help="'full-upstream' is a deprecated alias for prompt-ablation.",
    )
    parser.add_argument(
        "--samples",
        default="",
        help=(
            "JSON list or {'samples': [...]} manifest. Without it, the legacy "
            "10-case author-authored diagnostic set is used."
        ),
    )
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", "qwen3-vl-flash"))
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional text judge for semantic label, slot, and labeled e2e metrics.",
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=(
            float(os.environ["VLM_INPUT_COST_PER_MILLION"])
            if os.getenv("VLM_INPUT_COST_PER_MILLION")
            else None
        ),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=(
            float(os.environ["VLM_OUTPUT_COST_PER_MILLION"])
            if os.getenv("VLM_OUTPUT_COST_PER_MILLION")
            else None
        ),
    )
    parser.add_argument("--window-sec", type=float, default=16.0)
    parser.add_argument("--step-sec", type=float, default=8.0)
    parser.add_argument("--frames-per-window", type=int, default=16)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Deprecated alias for --frames-per-window.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--sample-order",
        choices=["manifest", "duration-asc"],
        default="manifest",
        help="Optionally run short episodes first without changing evaluation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Refresh derived metrics in an existing --output without API calls.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _resolve_path(value: str, manifest_path: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    return ROOT / path


def load_samples(path_value: str) -> tuple[list[dict[str, Any]], Path | None]:
    if not path_value:
        return (
            [
                {
                    **sample,
                    "annotation_status": "author_diagnostic",
                }
                for sample in DIAGNOSTIC_SAMPLES
            ],
            None,
        )
    path = Path(path_value)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("samples", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("--samples must contain a JSON list or a 'samples' list")
    samples = [dict(row) for row in rows if isinstance(row, dict)]
    required = {"id", "video", "instruction"}
    for index, sample in enumerate(samples):
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"Sample {index} is missing fields: {missing}")
        sample["video"] = str(_resolve_path(str(sample["video"]), path))
    return samples, path


def sample_duration_sec(sample: dict[str, Any]) -> float:
    segments = sample.get("segments")
    if isinstance(segments, list):
        end_times = [
            float(segment["end_sec"])
            for segment in segments
            if isinstance(segment, dict) and "end_sec" in segment
        ]
        if end_times:
            return max(end_times)
    metadata = sample.get("source_metadata")
    if isinstance(metadata, dict) and metadata.get("duration_sec") is not None:
        return float(metadata["duration_sec"])
    return float("inf")


def extract_frame_indices(
    video_path: str | Path,
    output_root: str | Path,
    frame_indices: list[int],
    window_id: int,
) -> list[Path]:
    output_dir = Path(output_root) / f"window_{window_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    paths: list[Path] = []
    try:
        for local_index, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Cannot decode frame {frame_index} from {video_path}"
                )
            path = output_dir / f"{local_index:02d}_{frame_index:08d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot write extracted frame: {path}")
            paths.append(path)
    finally:
        capture.release()
    return paths


def _image_content(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def call_vlm(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    frame_paths: list[Path],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    content = [_image_content(path) for path in frame_paths]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    if not parsed:
        raise ValueError("VLM response did not contain a valid JSON object")
    usage = getattr(response, "usage", None)
    return parsed, {
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
    }


def _run_method(
    *,
    method: str,
    sample: dict[str, Any],
    client: OpenAI,
    model: str,
    frame_root: Path,
    window_sec: float,
    step_sec: float,
    frames_per_window: int,
    max_tokens: int,
) -> dict[str, Any]:
    call_count = 0
    input_tokens = 0
    output_tokens = 0
    window_outputs: list[dict[str, Any]] = []

    def infer(frame_indices: list[int], window_id: int) -> dict[str, Any]:
        nonlocal call_count, input_tokens, output_tokens
        frame_paths = extract_frame_indices(
            sample["video"],
            frame_root / str(sample["id"]) / method,
            frame_indices,
            window_id,
        )
        prompt = (
            upstream_prompt(len(frame_paths))
            if method == "baseline"
            else build_semantic_motion_prompt(n_images=len(frame_paths))
        )
        call_count += 1
        parsed, usage = call_vlm(
            client,
            model=model,
            prompt=prompt,
            frame_paths=frame_paths,
            max_tokens=max_tokens,
        )
        parsed, normalization_warnings = normalize_window_output(
            parsed,
            frame_count=len(frame_indices),
        )
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        if method == "semantic_motion":
            if not parsed.get("instructions") and parsed.get("instruction"):
                parsed["instructions"] = [parsed["instruction"]]
            parsed.setdefault("transitions", [])
        transitions = parsed.get("transitions", [])
        instructions = parsed.get("instructions", [])
        if not isinstance(transitions, list) or any(
            not isinstance(value, int) for value in transitions
        ):
            raise ValueError(
                f"{method} returned non-integer transition indices"
            )
        if not isinstance(instructions, list) or any(
            not isinstance(value, str) for value in instructions
        ):
            raise ValueError(f"{method} returned invalid instructions")
        if instructions and len(instructions) != len(transitions) + 1:
            raise ValueError(
                f"{method} returned {len(transitions)} transitions but "
                f"{len(instructions)} instructions"
            )
        window_outputs.append(
            {
                "window_id": window_id,
                "frame_indices": frame_indices,
                "parsed": parsed,
                "normalization_warnings": normalization_warnings,
            }
        )
        return parsed

    started = time.monotonic()
    output = run_upstream_video2tasks(
        sample["video"],
        infer,
        sample_id=str(sample["id"]),
        window_sec=window_sec,
        step_sec=step_sec,
        frames_per_window=frames_per_window,
    )
    return {
        "method": (
            "pinned Video2Tasks prompt + aggregation"
            if method == "baseline"
            else "blind Semantic-Motion prompt + Video2Tasks aggregation"
        ),
        "prediction": output,
        "latency_s": time.monotonic() - started,
        "api_calls": call_count,
        "usage": {
            "requests": call_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "window_outputs": window_outputs,
    }


def _prediction_text(output: dict[str, Any]) -> str:
    return " ; ".join(
        str(segment.get("instruction", "")).strip()
        for segment in output.get("segments", [])
        if str(segment.get("instruction", "")).strip()
    )


def score_output(
    sample: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, float | None]:
    prediction = _prediction_text(output)
    predicted_actions = [
        str(segment.get("instruction", ""))
        for segment in output.get("segments", [])
    ]
    gold_actions = [str(value) for value in sample.get("actions", [])]
    return {
        "instruction_token_f1": token_f1(
            prediction,
            str(sample.get("instruction", "")),
        ),
        "target_mention_f1": (
            token_f1(
                prediction,
                str(sample.get("target_object", "")),
            )
            if str(sample.get("target_object", "")).strip()
            else None
        ),
        "action_coverage_f1": token_f1(
            prediction,
            gold_actions,
        ),
        "action_edit": normalized_edit_score(predicted_actions, gold_actions),
    }


def _boundary_scores(
    sample: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, dict[str, float]] | None:
    if "boundaries_sec" not in sample:
        return None
    fps = float(output.get("fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError("Prediction has no valid FPS for boundary scoring")
    predicted = [
        float(segment["start_frame"]) / fps
        for segment in output.get("segments", [])[1:]
    ]
    gold = [float(value) for value in sample.get("boundaries_sec", [])]
    return {
        "0.5s": boundary_metrics(predicted, gold, tolerance_sec=0.5),
        "1.0s": boundary_metrics(predicted, gold, tolerance_sec=1.0),
    }


def _segmental_scores(
    sample: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, dict[str, float]] | None:
    gold = sample.get("segments")
    if not isinstance(gold, list):
        return None
    fps = float(output.get("fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError("Prediction has no valid FPS for segment scoring")
    predicted = [
        {
            "start_sec": float(segment["start_frame"]) / fps,
            "end_sec": float(segment["end_frame"]) / fps,
            "label": "",
        }
        for segment in output.get("segments", [])
    ]
    temporal_gold = [
        {
            "start_sec": float(item["start_sec"]),
            "end_sec": float(item["end_sec"]),
            "label": "",
        }
        for item in gold
    ]
    return {
        str(threshold): segmental_metrics(
            predicted,
            temporal_gold,
            iou_threshold=threshold,
        )
        for threshold in (0.5, 0.75)
    }


def _temporal_segments(output: dict[str, Any]) -> list[dict[str, Any]]:
    fps = float(output.get("fps", 0.0) or 0.0)
    if fps <= 0:
        return []
    return [
        {
            "start_sec": float(segment["start_frame"]) / fps,
            "end_sec": float(segment["end_frame"]) / fps,
            "label": str(segment.get("instruction", "")),
        }
        for segment in output.get("segments", [])
    ]


def _gold_segments(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": float(segment["start_sec"]),
            "end_sec": float(segment["end_sec"]),
            "label": str(
                segment.get("label", segment.get("subtask", ""))
            ),
        }
        for segment in sample.get("segments", [])
    ]


def _usage_with_cost(
    usage: dict[str, int],
    *,
    input_price: float | None,
    output_price: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(usage)
    if input_price is None or output_price is None:
        result["estimated_cost_usd"] = None
    else:
        result["estimated_cost_usd"] = (
            usage["input_tokens"] * input_price
            + usage["output_tokens"] * output_price
        ) / 1_000_000
    return result


def _judge_label_pairs(
    client: OpenAI,
    *,
    model: str,
    pairs: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, int]]:
    if not pairs:
        return {}, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEMANTIC_JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    "Judge these indexed label pairs:\n"
                    + json.dumps(pairs, ensure_ascii=False)
                ),
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    judgments = parsed.get("judgments", [])
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for item in judgments if isinstance(judgments, list) else []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < len(pairs)):
            continue
        pair = pairs[index]
        by_pair[(pair["pred_index"], pair["gold_index"])] = item
    usage = getattr(response, "usage", None)
    return by_pair, {
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
    }


def _semantic_segment_scores(
    sample: dict[str, Any],
    output: dict[str, Any],
    *,
    client: OpenAI,
    judge_model: str | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    if not judge_model or not isinstance(sample.get("segments"), list):
        return None, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    predicted = _temporal_segments(output)
    gold = _gold_segments(sample)
    pairs = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, truth in enumerate(gold):
            iou = temporal_iou(
                (pred["start_sec"], pred["end_sec"]),
                (truth["start_sec"], truth["end_sec"]),
            )
            if iou >= 0.5:
                pairs.append(
                    {
                        "index": len(pairs),
                        "pred_index": pred_index,
                        "gold_index": gold_index,
                        "predicted_label": pred["label"],
                        "gold_label": truth["label"],
                        "temporal_iou": iou,
                    }
                )
    judgments, usage = _judge_label_pairs(
        client,
        model=judge_model,
        pairs=pairs,
        max_tokens=max_tokens,
    )

    def label_match(pred_index: int, gold_index: int) -> bool:
        return bool(
            judgments.get((pred_index, gold_index), {}).get(
                "semantic_match",
                False,
            )
        )

    aligned = match_temporal_segments(predicted, gold, iou_threshold=0.5)
    aligned_judgments = [
        judgments.get((pred_index, gold_index), {})
        for pred_index, gold_index, _ in aligned
    ]
    correct = sum(bool(item.get("semantic_match")) for item in aligned_judgments)
    predicted_slots = [
        dict(item.get("predicted_slots", {}))
        for item in aligned_judgments
        if isinstance(item.get("predicted_slots"), dict)
    ]
    gold_slots = [
        dict(item.get("gold_slots", {}))
        for item in aligned_judgments
        if isinstance(item.get("gold_slots"), dict)
    ]
    mapped_actions = []
    for pred_index, pred in enumerate(predicted):
        candidates = [
            (gold_index, temporal_iou(
                (pred["start_sec"], pred["end_sec"]),
                (truth["start_sec"], truth["end_sec"]),
            ))
            for gold_index, truth in enumerate(gold)
            if label_match(pred_index, gold_index)
        ]
        if candidates:
            gold_index, _ = max(candidates, key=lambda item: item[1])
            mapped_actions.append(gold[gold_index]["label"])
        else:
            mapped_actions.append(pred["label"])
    return {
        "semantic_label_accuracy": (
            correct / len(aligned_judgments) if aligned_judgments else 0.0
        ),
        "semantic_label_temporal_coverage": (
            len(aligned_judgments) / len(gold) if gold else 1.0
        ),
        "slot_f1": slot_f1(predicted_slots, gold_slots),
        "labeled_end_to_end_segment_f1": {
            str(threshold): labeled_segment_metrics(
                predicted,
                gold,
                label_match,
                iou_threshold=threshold,
            )
            for threshold in (0.5, 0.75)
        },
        "semantic_action_edit": normalized_edit_score(
            mapped_actions,
            [truth["label"] for truth in gold],
        ),
        "judge_pair_count": len(pairs),
    }, usage


def run_sample(
    *,
    sample: dict[str, Any],
    repeat: int,
    client: OpenAI,
    args: argparse.Namespace,
    frame_root: Path,
    baseline_first: bool,
) -> dict[str, Any]:
    methods = ["baseline", "semantic_motion"]
    if not baseline_first:
        methods.reverse()
    completed: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for method in methods:
        try:
            completed[method] = _run_method(
                method=method,
                sample=sample,
                client=client,
                model=args.model,
                frame_root=frame_root / f"repeat_{repeat:02d}",
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                frames_per_window=args.frames_per_window,
                max_tokens=args.max_tokens,
            )
            errors[method] = ""
        except Exception as exc:
            if args.fail_fast:
                raise
            completed[method] = {
                "method": method,
                "prediction": {},
                "latency_s": 0.0,
                "api_calls": 0,
                "usage": {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            }
            errors[method] = f"{type(exc).__name__}: {exc}"

    result: dict[str, Any] = {
        "sample": sample,
        "repeat": repeat,
        "method_order": methods,
        "input_budget": {
            "window_sec": args.window_sec,
            "step_sec": args.step_sec,
            "frames_per_window": args.frames_per_window,
            "max_tokens_per_call": args.max_tokens,
            "same_model": True,
            "temperature": 0.0,
        },
    }
    for method in ("baseline", "semantic_motion"):
        method_result = completed[method]
        output = method_result["prediction"]
        method_result["scores"] = score_output(sample, output)
        method_result["boundary"] = (
            _boundary_scores(sample, output) if output else None
        )
        method_result["segmental"] = (
            _segmental_scores(sample, output) if output else None
        )
        semantic_evaluation_error = ""
        try:
            semantic_evaluation, judge_usage = (
                _semantic_segment_scores(
                    sample,
                    output,
                    client=client,
                    judge_model=args.judge_model,
                    max_tokens=args.max_tokens,
                )
                if output
                else (
                    None,
                    {"requests": 0, "input_tokens": 0, "output_tokens": 0},
                )
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            semantic_evaluation = None
            judge_usage = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            semantic_evaluation_error = f"{type(exc).__name__}: {exc}"
        method_result["semantic_evaluation"] = semantic_evaluation
        method_result["semantic_evaluation_error"] = semantic_evaluation_error
        method_result["usage"] = _usage_with_cost(
            method_result["usage"],
            input_price=args.input_cost_per_million,
            output_price=args.output_cost_per_million,
        )
        method_result["judge_usage"] = _usage_with_cost(
            judge_usage,
            input_price=args.input_cost_per_million,
            output_price=args.output_cost_per_million,
        )
        method_result["error"] = errors[method]
        result[method] = method_result
    result["delta_instruction_token_f1"] = (
        result["semantic_motion"]["scores"]["instruction_token_f1"]
        - result["baseline"]["scores"]["instruction_token_f1"]
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in results
        if not row["baseline"]["error"] and not row["semantic_motion"]["error"]
    ]
    metric_keys = [
        "instruction_token_f1",
        "target_mention_f1",
        "action_coverage_f1",
        "action_edit",
    ]
    summary: dict[str, Any] = {
        "num_runs": len(results),
        "valid_runs": len(valid),
        "errored_runs": len(results) - len(valid),
    }
    for method in ("baseline", "semantic_motion"):
        summary[method] = {}
        for key in metric_keys:
            values = [
                float(row[method]["scores"][key])
                for row in valid
                if row[method]["scores"].get(key) is not None
            ]
            summary[method][key] = (
                sum(values) / len(values) if values else None
            )
    temporal_run_count = 0
    for method in ("baseline", "semantic_motion"):
        boundary_rows = [
            row[method]["boundary"]
            for row in valid
            if isinstance(row[method].get("boundary"), dict)
        ]
        temporal_run_count = max(temporal_run_count, len(boundary_rows))
        for tolerance in ("0.5s", "1.0s"):
            rows_at_tolerance = [
                row[tolerance]
                for row in boundary_rows
                if tolerance in row
            ]
            summary[method][f"boundary_f1_{tolerance}"] = (
                sum(float(row["f1"]) for row in rows_at_tolerance)
                / len(rows_at_tolerance)
                if rows_at_tolerance
                else None
            )
        for threshold in ("0.5", "0.75"):
            segment_rows = [
                row[method]["segmental"][threshold]
                for row in valid
                if isinstance(row[method].get("segmental"), dict)
                and threshold in row[method]["segmental"]
            ]
            summary[method][f"segment_f1_iou_{threshold}"] = (
                sum(float(row["f1"]) for row in segment_rows) / len(segment_rows)
                if segment_rows
                else None
            )
        semantic_rows = [
            row[method]["semantic_evaluation"]
            for row in valid
            if isinstance(row[method].get("semantic_evaluation"), dict)
        ]
        semantic_scalars = {
            "semantic_label_accuracy": ("semantic_label_accuracy",),
            "semantic_label_temporal_coverage": (
                "semantic_label_temporal_coverage",
            ),
            "slot_macro_f1": ("slot_f1", "macro_f1"),
            "labeled_e2e_f1_iou_0.5": (
                "labeled_end_to_end_segment_f1",
                "0.5",
                "f1",
            ),
            "labeled_e2e_f1_iou_0.75": (
                "labeled_end_to_end_segment_f1",
                "0.75",
                "f1",
            ),
            "semantic_action_edit": ("semantic_action_edit",),
        }
        for name, path in semantic_scalars.items():
            values = []
            for row in semantic_rows:
                value: Any = row
                for key in path:
                    value = value[key]
                values.append(float(value))
            summary[method][name] = (
                sum(values) / len(values) if values else None
            )
        summary[method]["mean_latency_s"] = (
            sum(float(row[method]["latency_s"]) for row in valid) / len(valid)
            if valid
            else None
        )
        costs = [
            float(row[method]["usage"]["estimated_cost_usd"])
            + float(row[method]["judge_usage"]["estimated_cost_usd"])
            for row in valid
            if row[method]["usage"].get("estimated_cost_usd") is not None
            and row[method]["judge_usage"].get("estimated_cost_usd") is not None
        ]
        summary[method]["mean_estimated_cost_usd"] = (
            sum(costs) / len(costs) if costs else None
        )
    summary["temporal_gold_run_count"] = temporal_run_count

    paired_metrics = [
        *metric_keys,
        "boundary_f1_0.5s",
        "boundary_f1_1.0s",
        "segment_f1_iou_0.5",
        "segment_f1_iou_0.75",
        "semantic_label_accuracy",
        "slot_macro_f1",
        "labeled_e2e_f1_iou_0.5",
        "labeled_e2e_f1_iou_0.75",
        "semantic_action_edit",
    ]
    summary["paired_bootstrap_95ci"] = {}
    for name in paired_metrics:
        deltas = []
        for row in valid:
            if name in row["baseline"]["scores"]:
                baseline_value = row["baseline"]["scores"][name]
                semantic_value = row["semantic_motion"]["scores"][name]
            else:
                baseline_value = summary["baseline"].get(name)
                semantic_value = summary["semantic_motion"].get(name)
                if name.startswith("boundary_f1_"):
                    tolerance = name.removeprefix("boundary_f1_")
                    baseline_row = row["baseline"].get("boundary") or {}
                    semantic_row = row["semantic_motion"].get("boundary") or {}
                    baseline_value = (
                        baseline_row.get(tolerance, {}).get("f1")
                    )
                    semantic_value = (
                        semantic_row.get(tolerance, {}).get("f1")
                    )
                elif name.startswith("segment_f1_iou_"):
                    threshold = name.removeprefix("segment_f1_iou_")
                    baseline_row = row["baseline"].get("segmental") or {}
                    semantic_row = row["semantic_motion"].get("segmental") or {}
                    baseline_value = baseline_row.get(threshold, {}).get("f1")
                    semantic_value = semantic_row.get(threshold, {}).get("f1")
                else:
                    semantic_paths = {
                        "semantic_label_accuracy": (
                            "semantic_label_accuracy",
                        ),
                        "slot_macro_f1": ("slot_f1", "macro_f1"),
                        "labeled_e2e_f1_iou_0.5": (
                            "labeled_end_to_end_segment_f1",
                            "0.5",
                            "f1",
                        ),
                        "labeled_e2e_f1_iou_0.75": (
                            "labeled_end_to_end_segment_f1",
                            "0.75",
                            "f1",
                        ),
                        "semantic_action_edit": ("semantic_action_edit",),
                    }
                    path = semantic_paths.get(name)
                    baseline_eval = row["baseline"].get(
                        "semantic_evaluation"
                    )
                    semantic_eval = row["semantic_motion"].get(
                        "semantic_evaluation"
                    )
                    if path and baseline_eval and semantic_eval:
                        baseline_value = baseline_eval
                        semantic_value = semantic_eval
                        for key in path:
                            baseline_value = baseline_value[key]
                            semantic_value = semantic_value[key]
                    else:
                        baseline_value = semantic_value = None
            if baseline_value is not None and semantic_value is not None:
                deltas.append(float(semantic_value) - float(baseline_value))
        summary["paired_bootstrap_95ci"][name] = paired_bootstrap_ci(deltas)

    instruction_deltas = [
        float(row["delta_instruction_token_f1"]) for row in valid
    ]
    summary["wins"] = sum(value > 0 for value in instruction_deltas)
    summary["ties"] = sum(abs(value) < 1e-12 for value in instruction_deltas)
    summary["losses"] = sum(value < 0 for value in instruction_deltas)
    return summary


def _result_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["sample"]["id"]), int(row.get("repeat", 0))


def rescore_output(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    for row in results:
        for method in ("baseline", "semantic_motion"):
            evaluation = row.get(method, {}).get("semantic_evaluation")
            if not isinstance(evaluation, dict):
                continue
            slots = evaluation.get("slot_f1")
            if isinstance(slots, dict) and int(slots.get("pair_count", 0)) == 0:
                slots["per_slot"] = {
                    "action": 0.0,
                    "object": 0.0,
                    "destination": 0.0,
                }
                slots["macro_f1"] = 0.0
    payload["summary"] = summarize(results)
    samples = [row.get("sample", {}) for row in results]
    temporal_gold = (
        int(payload.get("input", {}).get("repeats", 0)) >= 3
        and bool(samples)
        and all(
            sample.get("annotation_status")
            in {"adjudicated", "benchmark_gold"}
            and "boundaries_sec" in sample
            and isinstance(sample.get("segments"), list)
            for sample in samples
        )
    )
    independent_judge = bool(payload.get("judge_model")) and (
        payload.get("judge_model") != payload.get("model")
    )
    payload["formal_temporal_metrics"] = temporal_gold
    payload["formal_semantic_metrics"] = temporal_gold and independent_judge
    payload["semantic_judge_independent"] = independent_judge
    payload["formal"] = temporal_gold and independent_judge
    payload["rescored_at"] = utc_now_iso()
    write_json(path, payload)
    print(f"Rescored comparison output: {path}")


def main() -> None:
    args = parse_args()
    if args.rescore_only:
        rescore_output(Path(args.output))
        return
    if args.mode == "full-upstream":
        warnings.warn(
            "--mode full-upstream is now named prompt-ablation because the "
            "Semantic-Motion arm is a prompt, not the full Module A pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.frames is not None:
        warnings.warn(
            "--frames is deprecated; use --frames-per-window.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.frames_per_window = args.frames
    if not args.api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.frames_per_window < 1:
        raise ValueError("--frames-per-window must be >= 1")

    samples, sample_manifest = load_samples(args.samples)
    if args.sample_order == "duration-asc":
        samples.sort(key=lambda sample: (sample_duration_sec(sample), str(sample["id"])))
    output = Path(args.output)
    previous: dict[tuple[str, int], dict[str, Any]] = {}
    if args.resume and output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        previous = {
            _result_key(row): row
            for row in old.get("results", [])
            if isinstance(row, dict) and row.get("sample")
        }

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=max(0, args.max_retries),
    )
    results = list(previous.values())

    def save() -> None:
        results.sort(key=_result_key)
        temporal_gold = bool(samples) and args.repeats >= 3 and all(
            sample.get("annotation_status")
            in {"adjudicated", "benchmark_gold"}
            and "boundaries_sec" in sample
            and isinstance(sample.get("segments"), list)
            for sample in samples
        )
        independent_judge = bool(args.judge_model) and (
            args.judge_model != args.model
        )
        write_json(
            output,
            {
                "schema_version": "semantic-motion-video2tasks-ablation/v3",
                "comparison_mode": "prompt-ablation-pinned-aggregation",
                "formal": temporal_gold and independent_judge,
                "formal_temporal_metrics": temporal_gold,
                "formal_semantic_metrics": (
                    temporal_gold and independent_judge
                ),
                "semantic_judge_independent": independent_judge,
                "claim_scope": (
                    "prompt ablation only; does not evaluate the full "
                    "Semantic-Motion VideoTaskPipeline or multi-view fusion"
                ),
                "generated_at": utc_now_iso(),
                "code_revision": git_revision(ROOT),
                "model": args.model,
                "judge_model": args.judge_model,
                "base_url": args.base_url,
                "request_timeout_s": args.timeout,
                "max_retries": args.max_retries,
                "upstream_revision": UPSTREAM_REVISION,
                "filename_or_task_list_prior": False,
                "prompt_hashes": {
                    "semantic_motion": text_sha256(SEMANTIC_MOTION_PROMPT),
                    "semantic_judge": text_sha256(SEMANTIC_JUDGE_PROMPT),
                },
                "pricing": {
                    "input_usd_per_million_tokens": (
                        args.input_cost_per_million
                    ),
                    "output_usd_per_million_tokens": (
                        args.output_cost_per_million
                    ),
                },
                "input": {
                    "sample_manifest": (
                        str(sample_manifest) if sample_manifest is not None else None
                    ),
                    "sample_manifest_sha256": (
                        file_sha256(sample_manifest)
                        if sample_manifest is not None
                        else None
                    ),
                    "sample_source": (
                        "external_manifest"
                        if sample_manifest is not None
                        else "author_diagnostic_inline"
                    ),
                    "num_samples": len(samples),
                    "repeats": args.repeats,
                    "sample_order": args.sample_order,
                },
                "windowing": {
                    "window_sec": args.window_sec,
                    "step_sec": args.step_sec,
                    "frames_per_window": args.frames_per_window,
                },
                "summary": summarize(results),
                "results": results,
            },
        )

    with tempfile.TemporaryDirectory(prefix="video2tasks_ablation_") as temporary:
        frame_root = Path(temporary)
        for sample_index, sample in enumerate(samples):
            video_path = _resolve_path(str(sample["video"]), sample_manifest)
            sample["video"] = str(video_path)
            for repeat in range(args.repeats):
                key = str(sample["id"]), repeat
                previous_row = previous.get(key)
                if (
                    previous_row
                    and not previous_row["baseline"].get("error")
                    and not previous_row["semantic_motion"].get("error")
                ):
                    print(f"{sample['id']} repeat={repeat} already complete")
                    continue
                row = run_sample(
                    sample=sample,
                    repeat=repeat,
                    client=client,
                    args=args,
                    frame_root=frame_root,
                    baseline_first=(sample_index + repeat) % 2 == 0,
                )
                results = [
                    existing for existing in results if _result_key(existing) != key
                ]
                results.append(row)
                save()
                print(
                    f"{sample['id']} repeat={repeat} "
                    f"baseline={row['baseline']['scores']['instruction_token_f1']:.3f} "
                    f"semantic={row['semantic_motion']['scores']['instruction_token_f1']:.3f} "
                    f"delta={row['delta_instruction_token_f1']:+.3f} "
                    f"errors={bool(row['baseline']['error'] or row['semantic_motion']['error'])}",
                    flush=True,
                )
    save()
    print(json.dumps(summarize(results), indent=2))
    print(f"Saved prompt ablation to {output}")


if __name__ == "__main__":
    main()
