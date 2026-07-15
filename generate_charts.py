#!/usr/bin/env python3
"""Generate charts from current Mini-VLO result schemas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.runtime_utils import utc_now_iso, write_json


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PNG charts for static eval, Video2Tasks comparison, "
            "Semantic-Motion gold eval, or Module C eval JSON."
        )
    )
    parser.add_argument(
        "--input",
        default="",
        help="Result JSON. Defaults to the newest supported JSON in results/.",
    )
    parser.add_argument(
        "--kind",
        choices=[
            "auto",
            "static",
            "comparison",
            "semantic-motion",
            "libero-title",
            "refinement",
        ],
        default="auto",
    )
    parser.add_argument("--output-dir", default=str(ASSETS_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _supported_kind(data: dict[str, Any]) -> str | None:
    schema = str(data.get("schema_version", ""))
    if schema.startswith("semantic-motion-libero-title-eval/"):
        return "libero-title"
    if schema.startswith("mini-vlo-static-eval/") or (
        "per_scenario" in data and "summary" in data
    ):
        return "static"
    if "comparison_mode" in data or (
        isinstance(data.get("summary"), dict)
        and "baseline" in data["summary"]
        and "semantic_motion" in data["summary"]
    ):
        return "comparison"
    if "boundary" in data and "macro" in data and "micro" in data:
        return "semantic-motion"
    if "decision_distribution" in data:
        return "refinement"
    return None


def _load_input(path_value: str) -> tuple[Path, dict[str, Any]]:
    if path_value:
        path = Path(path_value)
        return path, json.loads(path.read_text(encoding="utf-8"))
    candidates = sorted(
        RESULTS_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if _supported_kind(data):
            return path, data
    raise FileNotFoundError("No supported JSON result found in results/.")


def _save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _static_charts(
    data: dict[str, Any],
    prefix: Path,
    dpi: int,
) -> list[Path]:
    summary = data["summary"]
    metric_keys = [
        "object_f1",
        "task_accuracy",
        "action_rouge_l",
        "semantic_similarity",
        "spatial_accuracy",
    ]
    labels = [
        "Object F1",
        "Task accuracy",
        "Action ROUGE-L",
        "Semantic similarity",
        "Spatial accuracy",
    ]
    values = [float(summary.get(key, 0.0)) for key in metric_keys]
    model = str(data.get("model", "unknown model"))
    outputs: list[Path] = []

    angles = [index / len(labels) * 2 * math.pi for index in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    closed_angles = angles + [angles[0]]
    closed_values = values + [values[0]]
    ax.plot(closed_angles, closed_values, "o-", linewidth=2.2, color="#2563eb")
    ax.fill(closed_angles, closed_values, alpha=0.16, color="#2563eb")
    ax.set_title(f"Static diagnostic metrics\n{model}", pad=24)
    output = prefix.with_name(prefix.name + "_static_overall.png")
    _save(fig, output, dpi)
    outputs.append(output)

    rows = [
        row
        for row in data.get("per_scenario", [])
        if row.get("status", "ok") == "ok" and isinstance(row.get("metrics"), dict)
    ]
    categories: dict[str, list[float]] = {}
    for row in rows:
        category = str(
            row.get("category")
            or str(row.get("scenario_id", "other")).split("_")[0]
        )
        categories.setdefault(category, []).append(
            float(row["metrics"].get("composite", 0.0))
        )
    if categories:
        names = sorted(categories)
        means = [_mean(categories[name]) for name in names]
        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 4.8))
        bars = ax.bar(names, means, color="#2563eb")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean composite score (0–1)")
        ax.set_xlabel("Scenario category")
        ax.set_title("Static benchmark score by category")
        output = prefix.with_name(prefix.name + "_static_categories.png")
        _save(fig, output, dpi)
        outputs.append(output)

        matrix = np.zeros((len(names), len(metric_keys)))
        for row_index, name in enumerate(names):
            members = [
                row
                for row in rows
                if str(
                    row.get("category")
                    or str(row.get("scenario_id", "other")).split("_")[0]
                )
                == name
            ]
            for column_index, key in enumerate(metric_keys):
                matrix[row_index, column_index] = _mean(
                    [float(row["metrics"].get(key, 0.0)) for row in members]
                )
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(names) * 0.65)))
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
        ax.set_yticks(range(len(names)), labels=names)
        for row_index in range(len(names)):
            for column_index in range(len(labels)):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.65 else "black",
                )
        fig.colorbar(image, ax=ax, label="Mean score (0–1)")
        ax.set_title("Static benchmark metric breakdown")
        output = prefix.with_name(prefix.name + "_static_heatmap.png")
        _save(fig, output, dpi)
        outputs.append(output)
    return outputs


def _comparison_charts(
    data: dict[str, Any],
    prefix: Path,
    dpi: int,
) -> list[Path]:
    summary = data.get("summary", {})
    baseline = summary.get("baseline", {})
    semantic = summary.get("semantic_motion", {})
    preferred = [
        "instruction_token_f1",
        "label_f1",
        "target_mention_f1",
        "target_f1",
        "action_coverage_f1",
        "action_f1",
        "action_edit",
        "boundary_f1_0.5s",
        "boundary_f1_1.0s",
        "segment_f1_iou_0.5",
        "segment_f1_iou_0.75",
        "semantic_label_accuracy",
        "slot_macro_f1",
        "labeled_e2e_f1_iou_0.5",
        "labeled_e2e_f1_iou_0.75",
        "semantic_action_edit",
        "composite",
    ]
    keys = [
        key
        for key in preferred
        if isinstance(baseline.get(key), (int, float))
        and isinstance(semantic.get(key), (int, float))
    ]
    if not keys:
        raise ValueError("Comparison summary contains no shared numeric metrics.")
    labels = [key.replace("_", " ") for key in keys]
    positions = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, len(keys) * 1.45), 5))
    ax.bar(
        positions - width / 2,
        [baseline[key] for key in keys],
        width,
        label="Video2Tasks",
        color="#64748b",
    )
    ax.bar(
        positions + width / 2,
        [semantic[key] for key in keys],
        width,
        label="Semantic-Motion prompt",
        color="#2563eb",
    )
    ax.set_xticks(positions, labels=labels, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score (0–1)")
    ax.set_xlabel("Metric")
    ax.set_title("Video2Tasks prompt-ablation metrics")
    ax.legend()
    outputs = [prefix.with_name(prefix.name + "_comparison_metrics.png")]
    _save(fig, outputs[-1], dpi)

    result_rows = data.get("results", [])
    ids: list[str] = []
    deltas: list[float] = []
    for row in result_rows:
        sample = row.get("sample", {})
        ids.append(str(sample.get("id", row.get("sample_id", len(ids)))))
        if isinstance(row.get("delta_instruction_token_f1"), (int, float)):
            deltas.append(float(row["delta_instruction_token_f1"]))
        else:
            deltas.append(float(row.get("delta_composite", 0.0)))
    if ids:
        fig, ax = plt.subplots(figsize=(10, max(4.5, len(ids) * 0.38)))
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in deltas]
        ax.barh(ids, deltas, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Semantic-Motion minus Video2Tasks score")
        ax.set_ylabel("Episode")
        ax.set_title("Per-episode paired score delta")
        outputs.append(prefix.with_name(prefix.name + "_comparison_deltas.png"))
        _save(fig, outputs[-1], dpi)
    return outputs


def _semantic_motion_chart(
    data: dict[str, Any],
    prefix: Path,
    dpi: int,
) -> list[Path]:
    boundary = data.get("boundary", {})
    labels: list[str] = []
    values: list[float] = []
    if isinstance(boundary.get("f1"), (int, float)):
        labels.append("Boundary F1")
        values.append(float(boundary["f1"]))
    else:
        for tolerance, row in sorted(boundary.items()):
            if isinstance(row, dict):
                labels.append(f"Boundary F1 ±{tolerance}s")
                values.append(float(row.get("f1", 0.0)))
    for threshold, row in sorted(data.get("segmental_f1_at_iou", {}).items()):
        labels.append(f"Segment F1@{threshold}")
        values.append(float(row.get("f1", 0.0)))
    for group in ("macro", "micro"):
        for key, value in data.get(group, {}).items():
            if isinstance(value, (int, float)):
                labels.append(f"{group}: {key}".replace("_", " "))
                values.append(float(value))
    for key in (
        "semantic_label_accuracy",
        "semantic_label_temporal_coverage",
    ):
        if isinstance(data.get(key), (int, float)):
            labels.append(key.replace("_", " "))
            values.append(float(data[key]))
    slot = data.get("slot_f1", {})
    if isinstance(slot.get("macro_f1"), (int, float)):
        labels.append("slot macro F1")
        values.append(float(slot["macro_f1"]))
    for threshold, row in sorted(
        data.get("labeled_end_to_end_segment_f1", {}).items()
    ):
        labels.append(f"Labeled e2e F1@{threshold}")
        values.append(float(row.get("f1", 0.0)))
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(labels) * 0.42)))
    bars = ax.barh(labels, values, color="#2563eb")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (0–1)")
    ax.set_ylabel("Gold-evaluation metric")
    ax.set_title("Semantic-Motion gold evaluation")
    output = prefix.with_name(prefix.name + "_semantic_motion.png")
    _save(fig, output, dpi)
    return [output]


def _libero_title_charts(
    data: dict[str, Any],
    prefix: Path,
    dpi: int,
) -> list[Path]:
    by_view = data.get("summary", {}).get("by_view", {})
    views = [view for view in ("fixed", "ego", "fused") if view in by_view]
    metric_keys = [
        "title_instruction_token_f1",
        "semantic_label_accuracy",
        "slot_macro_f1",
        "boundary_f1_0.5s",
        "segment_f1_iou_0.5",
        "labeled_e2e_f1_iou_0.5",
        "action_edit",
    ]
    metric_keys = [
        key
        for key in metric_keys
        if any(isinstance(by_view[view].get(key), (int, float)) for view in views)
    ]
    if not views or not metric_keys:
        raise ValueError("LIBERO title result has no plottable view metrics.")
    positions = np.arange(len(metric_keys))
    width = 0.8 / len(views)
    colors = ["#64748b", "#f59e0b", "#2563eb"]
    fig, ax = plt.subplots(figsize=(max(10, len(metric_keys) * 1.45), 5.2))
    for index, view in enumerate(views):
        offset = (index - (len(views) - 1) / 2) * width
        ax.bar(
            positions + offset,
            [float(by_view[view].get(key, 0.0)) for key in metric_keys],
            width,
            label=view,
            color=colors[index],
        )
    ax.set_xticks(
        positions,
        labels=[key.replace("_", " ") for key in metric_keys],
        rotation=22,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score (0–1)")
    ax.set_xlabel("Weak title-as-single-segment metric")
    ax.set_title("LIBERO Goal repeated view ablation")
    ax.legend()
    outputs = [prefix.with_name(prefix.name + "_libero_title_metrics.png")]
    _save(fig, outputs[-1], dpi)

    latencies = [float(by_view[view].get("mean_latency_s", 0.0)) for view in views]
    costs = [by_view[view].get("mean_estimated_cost_usd") for view in views]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    latency_bars = axes[0].bar(views, latencies, color=colors[: len(views)])
    axes[0].bar_label(latency_bars, fmt="%.1f", padding=3)
    axes[0].set_ylabel("Mean latency (seconds/run)")
    axes[0].set_xlabel("View mode")
    axes[0].set_title("Inference latency")
    numeric_costs = [
        float(value) if isinstance(value, (int, float)) else 0.0
        for value in costs
    ]
    cost_bars = axes[1].bar(views, numeric_costs, color=colors[: len(views)])
    axes[1].bar_label(cost_bars, fmt="$%.5f", padding=3)
    axes[1].set_ylabel("Estimated API cost (USD/run)")
    axes[1].set_xlabel("View mode")
    axes[1].set_title("Token-estimated cost")
    outputs.append(prefix.with_name(prefix.name + "_libero_title_runtime.png"))
    _save(fig, outputs[-1], dpi)
    return outputs


def _refinement_chart(
    data: dict[str, Any],
    prefix: Path,
    dpi: int,
) -> list[Path]:
    distribution = data.get("decision_distribution", {})
    labels = list(distribution)
    values = [int(distribution[label]) for label in labels]
    if not labels:
        raise ValueError("Refinement result has no decision distribution.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bars = axes[0].bar(labels, values, color="#64748b")
    axes[0].bar_label(bars, padding=3)
    axes[0].set_ylabel("Sample count")
    axes[0].set_xlabel("Decision")
    axes[0].set_title("Refinement decisions")
    classification = data.get("classification", {})
    metric_keys = [
        key
        for key in ("accuracy", "precision", "recall", "f1", "false_keep_rate", "auroc")
        if isinstance(classification.get(key), (int, float))
        and math.isfinite(float(classification[key]))
    ]
    metric_values = [float(classification[key]) for key in metric_keys]
    metric_bars = axes[1].barh(
        [key.replace("_", " ") for key in metric_keys],
        metric_values,
        color="#2563eb",
    )
    axes[1].bar_label(metric_bars, fmt="%.2f", padding=3)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Score (0–1)")
    axes[1].set_title("Formal labeled subset")
    output = prefix.with_name(prefix.name + "_refinement.png")
    _save(fig, output, dpi)
    return [output]


def main() -> None:
    args = parse_args()
    input_path, data = _load_input(args.input)
    detected = _supported_kind(data)
    kind = detected if args.kind == "auto" else args.kind
    if kind is None:
        raise ValueError(f"Cannot detect result schema: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / input_path.stem
    print(f"Generating {kind} charts from {input_path}")
    if kind == "static":
        outputs = _static_charts(data, prefix, args.dpi)
    elif kind == "comparison":
        outputs = _comparison_charts(data, prefix, args.dpi)
    elif kind == "semantic-motion":
        outputs = _semantic_motion_chart(data, prefix, args.dpi)
    elif kind == "libero-title":
        outputs = _libero_title_charts(data, prefix, args.dpi)
    else:
        outputs = _refinement_chart(data, prefix, args.dpi)
    manifest = write_json(
        prefix.with_name(prefix.name + "_charts.json"),
        {
            "schema_version": "mini-vlo-chart-manifest/v1",
            "generated_at": utc_now_iso(),
            "source": str(input_path),
            "source_kind": kind,
            "charts": [str(path) for path in outputs],
        },
    )
    print(f"Chart manifest: {manifest}")


if __name__ == "__main__":
    main()
