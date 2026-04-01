#!/usr/bin/env python3
"""Generate evaluation result charts for the README."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
ASSETS_DIR = Path(__file__).parent / "assets"


def load_latest_results() -> dict:
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError("No eval results found in results/")
    with open(files[-1]) as f:
        return json.load(f)


# ── Chart 1: Radar chart of overall metrics ──────────────────────────────

def radar_chart(summary: dict, out: Path):
    labels = [
        "Object\nRecognition F1",
        "Task\nClassification",
        "Action\nROUGE-L",
        "Semantic\nSimilarity",
        "Spatial\nReasoning",
    ]
    values = [
        summary["object_f1"],
        summary["task_accuracy"],
        summary["action_rouge_l"],
        summary["semantic_similarity"],
        summary["spatial_accuracy"],
    ]

    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")

    ax.plot(angles_plot, values_plot, "o-", linewidth=2.5, color="#3498db")
    ax.fill(angles_plot, values_plot, alpha=0.2, color="#3498db")

    for angle, val, label_val in zip(angles, values, values):
        ax.text(angle, val + 0.06, f"{label_val:.2f}", ha="center", fontsize=10,
                fontweight="bold", color="#2c3e50")

    ax.set_title("Mini-VLO Overall Metrics\n(Qwen-VL-Plus)", fontsize=14,
                 fontweight="bold", pad=25)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Radar chart -> {out}")


# ── Chart 2: Per-category bar chart ──────────────────────────────────────

def category_bar_chart(per_scenario: list[dict], out: Path):
    category_map = {
        "pnp": "Pick & Place",
        "oc": "Open / Close",
        "to": "Turn On/Off",
        "sp": "Spatial",
        "ms": "Multi-step",
    }

    cat_scores: dict[str, list[float]] = {v: [] for v in category_map.values()}
    for s in per_scenario:
        prefix = s["scenario_id"].split("_")[0]
        cat_name = category_map.get(prefix, "Other")
        cat_scores[cat_name].append(s["metrics"]["composite"])

    cats = list(cat_scores.keys())
    means = [sum(v) / len(v) if v else 0 for v in cat_scores.values()]
    counts = [len(v) for v in cat_scores.values()]

    colors = ["#3498db", "#e67e22", "#27ae60", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, means, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, mean, count in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"n={count}", ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Composite Score", fontsize=12, fontweight="bold")
    ax.set_title("Composite Score by Task Category", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Category bar -> {out}")


# ── Chart 3: Per-metric breakdown across categories ─────────────────────

def metric_heatmap(per_scenario: list[dict], out: Path):
    category_map = {
        "pnp": "Pick & Place",
        "oc": "Open / Close",
        "to": "Turn On/Off",
        "sp": "Spatial",
        "ms": "Multi-step",
    }
    metric_keys = ["object_f1", "task_accuracy", "action_rouge_l",
                   "semantic_similarity", "spatial_accuracy"]
    metric_labels = ["Object F1", "Task Acc", "ROUGE-L", "Semantic Sim", "Spatial Acc"]

    cat_data: dict[str, dict[str, list[float]]] = {}
    for cat_name in category_map.values():
        cat_data[cat_name] = {m: [] for m in metric_keys}

    for s in per_scenario:
        prefix = s["scenario_id"].split("_")[0]
        cat_name = category_map.get(prefix, "Other")
        for m in metric_keys:
            cat_data[cat_name][m].append(s["metrics"][m])

    cats = list(category_map.values())
    data = np.zeros((len(cats), len(metric_keys)))
    for i, cat in enumerate(cats):
        for j, m in enumerate(metric_keys):
            vals = cat_data[cat][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=10, fontweight="bold")

    for i in range(len(cats)):
        for j in range(len(metric_keys)):
            val = data[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title("Per-Metric Breakdown by Category", fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Heatmap     -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_latest_results()

    print("Generating charts ...")
    radar_chart(data["summary"], ASSETS_DIR / "radar_overall.png")
    category_bar_chart(data["per_scenario"], ASSETS_DIR / "category_scores.png")
    metric_heatmap(data["per_scenario"], ASSETS_DIR / "metric_heatmap.png")
    print("Done!")


if __name__ == "__main__":
    main()
