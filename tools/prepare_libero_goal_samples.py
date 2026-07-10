#!/usr/bin/env python3
"""Convert selected LIBERO Goal HDF5 demos into paired videos and trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


VIEW_DATASETS = {
    "fixed": "agentview_rgb",
    "ego": "eye_in_hand_rgb",
}


def _numeric_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[-1]), name
    except ValueError:
        return 10**9, name


def _task_metadata(data_group: h5py.Group, source_path: Path) -> tuple[str, str, int]:
    bddl_path = str(data_group.attrs.get("bddl_file_name", source_path.stem))
    task_id = Path(bddl_path).stem
    instruction = task_id.replace("_", " ").strip()

    fps = 20
    raw_env_args = data_group.attrs.get("env_args")
    if raw_env_args:
        try:
            env_args = json.loads(str(raw_env_args))
            fps = int(env_args.get("env_kwargs", {}).get("control_freq", fps))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return task_id, instruction, fps


def _write_video(frames_rgb: np.ndarray, output_path: Path, fps: int) -> None:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames_rgb.shape}")

    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")

    try:
        for frame_rgb in frames_rgb:
            frame = np.asarray(frame_rgb, dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_trajectory(
    demo: h5py.Group,
    output_path: Path,
    *,
    fps: int,
    task_id: str,
    instruction: str,
    source_hdf5: Path,
    demo_id: str,
) -> None:
    obs = demo["obs"]
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
    ee_ori = np.asarray(obs["ee_ori"], dtype=np.float64)
    gripper = np.asarray(obs["gripper_states"], dtype=np.float64)
    frame_count = min(len(ee_pos), len(ee_ori), len(gripper))

    payload: dict[str, Any] = {
        "task_id": task_id,
        "ground_truth_instruction": instruction,
        "ground_truth_source": "official LIBERO BDDL filename",
        "source_hdf5": str(source_hdf5),
        "demo_id": demo_id,
        "fps": fps,
        "steps": [],
    }
    for frame_index in range(frame_count):
        payload["steps"].append(
            {
                "t": frame_index,
                "timestamp_sec": frame_index / fps,
                "ee_pos": ee_pos[frame_index].tolist(),
                "ee_ori": ee_ori[frame_index].tolist(),
                "gripper_states": gripper[frame_index].tolist(),
            }
        )

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def convert_file(
    source_path: Path,
    output_root: Path,
    demos_per_task: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with h5py.File(source_path, "r") as h5_file:
        data = h5_file["data"]
        task_id, instruction, fps = _task_metadata(data, source_path)
        task_dir = output_root / source_path.stem
        task_dir.mkdir(parents=True, exist_ok=True)

        demo_ids = sorted(data.keys(), key=_numeric_demo_key)[:demos_per_task]
        for demo_id in demo_ids:
            demo = data[demo_id]
            obs = demo["obs"]
            outputs = {
                "fixed": task_dir / f"{demo_id}_agentview_rgb.mp4",
                "ego": task_dir / f"{demo_id}_eye_in_hand_rgb.mp4",
            }
            for view_name, dataset_name in VIEW_DATASETS.items():
                if overwrite or not outputs[view_name].exists():
                    _write_video(np.asarray(obs[dataset_name]), outputs[view_name], fps)

            trajectory_path = task_dir / f"{demo_id}_traj.json"
            if overwrite or not trajectory_path.exists():
                _write_trajectory(
                    demo,
                    trajectory_path,
                    fps=fps,
                    task_id=task_id,
                    instruction=instruction,
                    source_hdf5=source_path,
                    demo_id=demo_id,
                )

            truth_path = task_dir / f"{demo_id}_ground_truth.json"
            truth = {
                "task_id": task_id,
                "instruction": instruction,
                "source": "official LIBERO BDDL filename",
                "scope": "weak task-level label only",
                "limitations": (
                    "This title is not frame-level boundary, micro-action, contact-state, "
                    "or keep/drop ground truth."
                ),
            }
            truth_path.write_text(
                json.dumps(truth, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            records.append(
                {
                    "sample_id": f"{task_id}_{demo_id}",
                    "task_id": task_id,
                    "ground_truth_instruction": instruction,
                    "ground_truth_source": truth["source"],
                    "ground_truth_scope": truth["scope"],
                    "fps": fps,
                    "frame_count": int(len(obs[VIEW_DATASETS["fixed"]])),
                    "views": {
                        name: str(path) for name, path in outputs.items()
                    },
                    "trajectory": str(trajectory_path),
                    "truth": str(truth_path),
                    "source_hdf5": str(source_path),
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired-view LIBERO Goal samples from HDF5 files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/libero_goal/raw",
        help="Directory containing selected LIBERO Goal .hdf5 files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/libero_goal/processed",
        help="Output directory for videos, trajectories, and manifest",
    )
    parser.add_argument(
        "--demos-per-task",
        type=int,
        default=1,
        help="Number of demos to export from each task file",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source_path in files:
        records.extend(
            convert_file(
                source_path,
                output_dir,
                demos_per_task=max(1, args.demos_per_task),
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "dataset": "LIBERO Goal",
        "sample_count": len(records),
        "ground_truth_policy": (
            "Use the official BDDL/file title as weak task-level instruction truth only."
        ),
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} paired-view sample(s): {manifest_path}")


if __name__ == "__main__":
    main()
