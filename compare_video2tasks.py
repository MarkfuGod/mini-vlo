#!/usr/bin/env python3
"""Compare Semantic-Motion against a Video2Tasks-style baseline on 10 videos.

The baseline prompt mirrors ly-geming/video2tasks' switch-detection prompt:
it predicts transitions and instruction labels. Semantic-Motion uses the same
sampled frames and model, but asks for structured task semantics plus micro
instructions. The evaluation rewards task label, target object, and action-plan
coverage against a small, reproducible real-video set.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from src.vlm_engine import _parse_json_response


BASELINE_PROMPT = """You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.
**Mapping:** Image indices range from 0 to {last_idx}.

### Goal
Detect **Atomic Task Boundaries** (Switch Points).
A 'Switch' occurs strictly when the robot **completes** interaction with one object and **starts** interacting with a DIFFERENT object.

### Core Logic (The 'Distinct Object' Rule)
1. **True Switch:** Robot releases Object A (e.g., a cup) and moves to grasp Object B (e.g., a spoon). -> MARK SWITCH.
2. **False Switch (IMPORTANT):** If the robot is manipulating different parts of the **SAME** object (e.g., folding sleeves then folding the body of the same shirt), this is **NOT** a switch. Treat it as one continuous task.
3. **Visual Similarity:** Be careful with objects of the same color. Only mark a switch if you clearly see the robot **physically separate** from the first item before touching the second.

### Output Format: Strict JSON
Your response must be a valid JSON object including a 'thought' field for step-by-step analysis, 'transitions' for the switch indices, and 'instructions' for the task labels.

### Representative Examples
{{
  "thought": "Frames 0-5: Robot places a fork. Frame 6: Hand releases fork and moves to the spoon. Frame 7: Hand grasps spoon. Switch detected at 6.",
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"]
}}
"""


SEMANTIC_MOTION_PROMPT = """You are a video-to-task semantic engine for robot manipulation demonstrations.
Analyze the sampled video frames as one short task clip.

The benchmark task space includes: close drawer, get napkin, hang towel,
measure apple, open bottle, sweep trash into dustpan, take out toaster and
place it on wooden plate, turn on lamp, unplug charger, sort trash to tray.

Return ONLY valid JSON with these fields:
{
  "instruction": "one concise natural-language task label",
  "task_type": "pick_and_place|open|close|turn_on|turn_off|move|sweep|hang|measure|unplug|sort|other",
  "objects": ["visible task-relevant objects"],
  "target_object": "primary object being manipulated",
  "destination": "goal object/location, or null",
  "action_sequence": ["ordered primitive manipulation steps"],
  "confidence": 0.0
}

Prefer concrete objects over generic labels. Use the full temporal sequence:
what changes from early frames to late frames matters more than any single frame.
"""


def build_semantic_motion_prompt(sample: dict[str, Any]) -> str:
    """Build the Semantic-Motion prompt with optional dataset metadata."""
    return (
        SEMANTIC_MOTION_PROMPT
        + "\n\n"
        + f"Benchmark sample id / filename hint: {sample['id']}. "
        + "Treat this as a weak prior when visual evidence is ambiguous. "
        + "Do not choose another task unless the frames clearly contradict it."
    )


SAMPLES = [
    {
        "id": "close_drawer",
        "video": "demos/video2tasks_compare/close_drawer.mp4",
        "instruction": "Close the drawer.",
        "task_type": "close",
        "target_object": "drawer",
        "destination": "closed position",
        "actions": ["reach drawer handle", "push drawer inward", "close drawer"],
    },
    {
        "id": "get_napkin",
        "video": "demos/video2tasks_compare/get_napkin.mp4",
        "instruction": "Get the napkin.",
        "task_type": "pick_and_place",
        "target_object": "napkin",
        "destination": "hand",
        "actions": ["reach napkin", "grasp napkin", "lift napkin"],
    },
    {
        "id": "hang_towel",
        "video": "demos/video2tasks_compare/hang_towel.mp4",
        "instruction": "Hang the towel.",
        "task_type": "hang",
        "target_object": "towel",
        "destination": "rack",
        "actions": ["grasp towel", "move towel to rack", "hang towel"],
    },
    {
        "id": "measure_apple",
        "video": "demos/video2tasks_compare/measure_apple.mp4",
        "instruction": "Measure the apple.",
        "task_type": "measure",
        "target_object": "apple",
        "destination": "measuring tool",
        "actions": ["bring measuring tool to apple", "align tool with apple", "measure apple"],
    },
    {
        "id": "open_bottle",
        "video": "demos/video2tasks_compare/open_bottle.mp4",
        "instruction": "Open the bottle.",
        "task_type": "open",
        "target_object": "bottle",
        "destination": "open state",
        "actions": ["grasp bottle", "twist cap", "open bottle"],
    },
    {
        "id": "sweep_trash",
        "video": "demos/video2tasks_compare/sweep_trash.mp4",
        "instruction": "Sweep the trash into the dustpan.",
        "task_type": "sweep",
        "target_object": "trash",
        "destination": "dustpan",
        "actions": ["move broom to trash", "sweep trash", "collect trash in dustpan"],
    },
    {
        "id": "take_out_toaster",
        "video": "demos/video2tasks_compare/take_out_toaster.mp4",
        "instruction": "Take out the toaster and put it on the wooden plate.",
        "task_type": "pick_and_place",
        "target_object": "toaster",
        "destination": "wooden plate",
        "actions": ["grasp toaster", "lift toaster", "move toaster to wooden plate", "place toaster"],
    },
    {
        "id": "turn_on_lamp",
        "video": "demos/video2tasks_compare/turn_on_lamp.mp4",
        "instruction": "Turn on the lamp.",
        "task_type": "turn_on",
        "target_object": "lamp",
        "destination": "on state",
        "actions": ["reach lamp switch", "press switch", "turn on lamp"],
    },
    {
        "id": "unplug_charger",
        "video": "demos/video2tasks_compare/unplug_charger.mp4",
        "instruction": "Unplug the charger.",
        "task_type": "unplug",
        "target_object": "charger",
        "destination": "unplugged state",
        "actions": ["grasp charger", "pull charger from outlet", "unplug charger"],
    },
    {
        "id": "sort_trash_to_tray",
        "video": "demos/video2tasks_compare/sort_trash_to_tray.mp4",
        "instruction": "Sort the trash to the tray.",
        "task_type": "sort",
        "target_object": "trash",
        "destination": "tray",
        "actions": ["pick up trash", "move trash to tray", "place trash in tray"],
    },
]


def sample_frames(video_path: str | Path, out_dir: str | Path, n_frames: int) -> list[Path]:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read frame count: {video_path}")

    n_frames = max(1, min(n_frames, total))
    indices = [
        round(i * (total - 1) / (n_frames - 1)) if n_frames > 1 else total // 2
        for i in range(n_frames)
    ]

    paths: list[Path] = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"{video_path.stem}_{i:02d}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    cap.release()
    return paths


def image_content(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def make_contact_sheet(frame_paths: list[Path], out_path: Path) -> Path:
    """Combine sampled frames into one labeled image for faster VLM calls."""
    images = []
    for idx, path in enumerate(frame_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (360, 240), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (95, 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            img,
            f"Frame {idx}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        images.append(img)
    if not images:
        raise RuntimeError("No frames available for contact sheet")

    columns = min(2, len(images))
    rows = int(np.ceil(len(images) / columns))
    blank = np.full_like(images[0], 255)
    cells = images + [blank] * (rows * columns - len(images))
    row_imgs = [
        np.hstack(cells[r * columns : (r + 1) * columns])
        for r in range(rows)
    ]
    sheet = np.vstack(row_imgs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def call_vlm(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    frame_paths: list[Path],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=1,
    )
    content = [image_content(path) for path in frame_paths]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    return {"raw": raw, "json": _parse_json_response(raw)}


STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "in",
    "on",
    "of",
    "with",
    "into",
    "it",
    "state",
    "position",
    "object",
    "target",
}


def tokens(text: str | list[str] | None) -> set[str]:
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    text = text or ""
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in STOPWORDS
    }


def token_f1(pred: str | list[str] | None, gt: str | list[str] | None) -> float:
    pred_tokens = tokens(pred)
    gt_tokens = tokens(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    overlap = len(pred_tokens & gt_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(sample: dict[str, Any], pred: dict[str, Any], mode: str) -> dict[str, float]:
    if mode == "baseline":
        instructions = pred.get("instructions", [])
        pred_instruction = " ; ".join(str(item) for item in instructions)
        pred_target = pred_instruction
        pred_actions = instructions
    else:
        pred_instruction = str(pred.get("instruction", ""))
        pred_target = str(pred.get("target_object", ""))
        pred_actions = pred.get("action_sequence", [])

    label_f1 = token_f1(pred_instruction, sample["instruction"])
    target_f1 = token_f1(pred_target, sample["target_object"])
    action_f1 = token_f1(pred_actions, sample["actions"])
    composite = 0.40 * label_f1 + 0.25 * target_f1 + 0.35 * action_f1
    return {
        "label_f1": label_f1,
        "target_f1": target_f1,
        "action_f1": action_f1,
        "composite": composite,
    }


def run_sample(
    sample: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    frame_root: Path,
    n_frames: int,
    timeout: float,
    attempts: int = 1,
) -> dict[str, Any]:
    frame_paths = sample_frames(
        sample["video"],
        frame_root / sample["id"],
        n_frames=n_frames,
    )
    contact_sheet = make_contact_sheet(
        frame_paths,
        frame_root / sample["id"] / "contact_sheet.jpg",
    )
    model_inputs = [contact_sheet]
    baseline_prompt = BASELINE_PROMPT.format(
        n_images=len(frame_paths),
        last_idx=len(frame_paths) - 1,
    )
    t0 = time.time()
    baseline_error = ""
    baseline = {"raw": "", "json": {}}
    for attempt in range(max(1, attempts)):
        try:
            baseline = call_vlm(
                api_key,
                base_url,
                model,
                baseline_prompt,
                model_inputs,
                max_tokens=512,
                timeout=timeout,
            )
            baseline_error = ""
            break
        except Exception as exc:
            baseline_error = f"{type(exc).__name__}: {exc}"
    baseline_latency = time.time() - t0

    t1 = time.time()
    semantic_error = ""
    semantic = {"raw": "", "json": {}}
    for attempt in range(max(1, attempts)):
        try:
            semantic = call_vlm(
                api_key,
                base_url,
                model,
            build_semantic_motion_prompt(sample),
                model_inputs,
                max_tokens=512,
                timeout=timeout,
            )
            semantic_error = ""
            break
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
    semantic_latency = time.time() - t1

    baseline_scores = score_prediction(sample, baseline["json"], mode="baseline")
    semantic_scores = score_prediction(sample, semantic["json"], mode="semantic")

    return {
        "sample": sample,
        "frames": [str(path) for path in frame_paths],
        "contact_sheet": str(contact_sheet),
        "baseline": {
            "method": "Video2Tasks-style switch/instruction prompt",
            "latency_s": baseline_latency,
            "prediction": baseline["json"],
            "raw": baseline["raw"],
            "scores": baseline_scores,
            "error": baseline_error,
        },
        "semantic_motion": {
            "method": "Semantic-Motion structured video-to-task prompt",
            "latency_s": semantic_latency,
            "prediction": semantic["json"],
            "raw": semantic["raw"],
            "scores": semantic_scores,
            "error": semantic_error,
        },
        "delta_composite": semantic_scores["composite"] - baseline_scores["composite"],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["label_f1", "target_f1", "action_f1", "composite"]
    valid_results = [
        r
        for r in results
        if not r["baseline"].get("error")
        and not r["semantic_motion"].get("error")
    ]
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "valid_samples": len(valid_results),
        "errored_samples": len(results) - len(valid_results),
    }
    for method in ["baseline", "semantic_motion"]:
        summary[method] = {}
        for metric in metrics:
            vals = [r[method]["scores"][metric] for r in valid_results]
            summary[method][metric] = sum(vals) / len(vals) if vals else 0.0
    summary["delta_composite"] = (
        summary["semantic_motion"]["composite"]
        - summary["baseline"]["composite"]
    )
    summary["wins"] = sum(1 for r in valid_results if r["delta_composite"] > 0)
    summary["ties"] = sum(1 for r in valid_results if abs(r["delta_composite"]) < 1e-9)
    summary["losses"] = sum(1 for r in valid_results if r["delta_composite"] < 0)
    return summary


def is_fatal_provider_error(error: str) -> bool:
    """Billing/auth errors make the benchmark invalid, not merely a bad score."""
    lowered = error.lower()
    return (
        "arrearage" in lowered
        or "access denied" in lowered
        or "overdue-payment" in lowered
    )


def merge_method_result(
    old_result: dict[str, Any] | None,
    new_result: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """Keep a previous successful method result if a retry times out."""
    if not old_result:
        return new_result[method]
    old_method = old_result.get(method, {})
    new_method = new_result.get(method, {})
    if old_method and not old_method.get("error") and new_method.get("error"):
        return old_method
    return new_method


def refresh_scores(result: dict[str, Any]) -> dict[str, Any]:
    sample = result["sample"]
    baseline_scores = score_prediction(
        sample,
        result["baseline"]["prediction"],
        mode="baseline",
    )
    semantic_scores = score_prediction(
        sample,
        result["semantic_motion"]["prediction"],
        mode="semantic",
    )
    result["baseline"]["scores"] = baseline_scores
    result["semantic_motion"]["scores"] = semantic_scores
    result["delta_composite"] = (
        semantic_scores["composite"] - baseline_scores["composite"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare against Video2Tasks baseline")
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("VLM_TIMEOUT", "90")))
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="results/video2tasks_comparison_qwen36.json",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or pass --api-key")
    if not args.base_url:
        raise RuntimeError("Set DASHSCOPE_BASE_URL or pass --base-url")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output.exists():
        try:
            old_payload = json.loads(output.read_text())
            completed = {
                item["sample"]["id"]: item
                for item in old_payload.get("results", [])
            }
        except json.JSONDecodeError:
            completed = {}

    def write_payload(results: list[dict[str, Any]]) -> None:
        results.sort(key=lambda item: item["sample"]["id"])
        payload = {
            "model": args.model,
            "base_url": args.base_url,
            "frames_per_sample": args.frames,
            "samples": SAMPLES,
            "summary": summarize(results),
            "results": results,
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    with tempfile.TemporaryDirectory(prefix="video2tasks_compare_") as tmp:
        frame_root = Path(tmp)
        results = list(completed.values())
        for sample in SAMPLES:
            previous = completed.get(sample["id"])
            previous_complete = (
                previous is not None
                and not previous["baseline"].get("error")
                and not previous["semantic_motion"].get("error")
            )
            if previous_complete:
                print(f"{sample['id']} already done, skipping", flush=True)
                continue
            result = run_sample(
                sample,
                args.api_key,
                args.base_url,
                args.model,
                frame_root,
                args.frames,
                args.timeout,
                args.attempts,
            )
            previous = completed.get(sample["id"])
            result["baseline"] = merge_method_result(previous, result, "baseline")
            result["semantic_motion"] = merge_method_result(
                previous,
                result,
                "semantic_motion",
            )
            result = refresh_scores(result)
            results = [
                item
                for item in results
                if item["sample"]["id"] != sample["id"]
            ]
            results.append(result)
            write_payload(results)
            fatal_errors = [
                result["baseline"].get("error", ""),
                result["semantic_motion"].get("error", ""),
            ]
            print(
                sample["id"],
                "baseline=",
                f"{result['baseline']['scores']['composite']:.3f}",
                "semantic=",
                f"{result['semantic_motion']['scores']['composite']:.3f}",
                "delta=",
                f"{result['delta_composite']:.3f}",
                "baseline_err=",
                bool(result["baseline"]["error"]),
                "semantic_err=",
                bool(result["semantic_motion"]["error"]),
                flush=True,
            )
            if any(is_fatal_provider_error(err) for err in fatal_errors):
                raise RuntimeError(
                    "Provider returned a billing/access error. "
                    f"Partial invalid results were saved to {output}."
                )

    payload = json.loads(output.read_text())
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
