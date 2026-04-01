#!/usr/bin/env python3
"""Generate the synthetic robot-task benchmark (30 scenarios + images)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.scenario import GroundTruth, Scenario

OUT_DIR = Path(__file__).parent / "benchmark"
IMG_DIR = OUT_DIR / "images"

# ── colour palette ────────────────────────────────────────────────────────
COLORS = {
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "green":  "#27ae60",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "purple": "#9b59b6",
    "brown":  "#8b5e3c",
    "gray":   "#95a5a6",
    "white":  "#ecf0f1",
    "black":  "#2c3e50",
}

# ── drawing primitives ────────────────────────────────────────────────────

def _new_fig():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f0e8")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _draw_table(ax, x=0.5, y=0.5, w=9, h=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor="#d4c5a9", edgecolor="#8b7355", linewidth=2,
    ))
    ax.text(5, 0.2, "TABLE  (top-down view)", ha="center", fontsize=7, color="#8b7355")


def _draw_mug(ax, cx, cy, color="red", label="mug"):
    c = COLORS.get(color, color)
    circle = plt.Circle((cx, cy), 0.4, fc=c, ec="black", lw=1.2)
    ax.add_patch(circle)
    inner = plt.Circle((cx, cy), 0.22, fc="#f5f0e8", ec="black", lw=0.6)
    ax.add_patch(inner)
    ax.text(cx, cy - 0.65, label, ha="center", fontsize=7, weight="bold")


def _draw_bowl(ax, cx, cy, color="blue", label="bowl"):
    c = COLORS.get(color, color)
    ellipse = mpatches.Ellipse((cx, cy), 1.0, 0.7, fc=c, ec="black", lw=1.2, alpha=0.85)
    ax.add_patch(ellipse)
    ax.text(cx, cy - 0.6, label, ha="center", fontsize=7, weight="bold")


def _draw_plate(ax, cx, cy, label="plate"):
    outer = plt.Circle((cx, cy), 0.55, fc="#ecf0f1", ec="#7f8c8d", lw=1.5)
    ax.add_patch(outer)
    inner = plt.Circle((cx, cy), 0.35, fc="#f9f9f9", ec="#bdc3c7", lw=0.8)
    ax.add_patch(inner)
    ax.text(cx, cy - 0.75, label, ha="center", fontsize=7, weight="bold")


def _draw_box(ax, x, y, w, h, color="brown", label="box"):
    c = COLORS.get(color, color)
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=c, ec="black", lw=1.2, alpha=0.8))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, weight="bold", color="white")


def _draw_cabinet(ax, x, y, w, h, label="cabinet"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#a0784c", ec="#5c3d1e", lw=2))
    ax.add_patch(mpatches.Rectangle((x + 0.15, y + 0.15), w - 0.3, h - 0.3,
                                     fc="#c49a6c", ec="#5c3d1e", lw=1))
    knob_x = x + w / 2
    knob_y = y + h / 2
    ax.plot(knob_x, knob_y, "o", color="#5c3d1e", markersize=5)
    ax.text(x + w / 2, y - 0.25, label, ha="center", fontsize=7, weight="bold")


def _draw_drawer(ax, x, y, w, h, label="drawer"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#b8956a", ec="#5c3d1e", lw=2))
    handle_y = y + h / 2
    ax.plot([x + w * 0.3, x + w * 0.7], [handle_y, handle_y], color="#5c3d1e", lw=3)
    ax.text(x + w / 2, y - 0.25, label, ha="center", fontsize=7, weight="bold")


def _draw_appliance(ax, x, y, w, h, label="stove", knob=True):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#555555", ec="black", lw=2, alpha=0.9))
    if knob:
        ax.plot(x + w * 0.2, y + h * 0.3, "o", color="red", markersize=8)
        ax.text(x + w * 0.2, y + h * 0.3 - 0.35, "knob", ha="center", fontsize=5, color="white")
    ax.text(x + w / 2, y + h + 0.2, label, ha="center", fontsize=8, weight="bold")


def _draw_shelf(ax, x, y, w, h, label="shelf"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#c9b896", ec="#7a6540", lw=2))
    for i in range(1, 3):
        ly = y + i * h / 3
        ax.plot([x, x + w], [ly, ly], color="#7a6540", lw=1)
    ax.text(x + w / 2, y - 0.25, label, ha="center", fontsize=7, weight="bold")


def _draw_microwave(ax, x, y, w=2.0, h=1.5, label="microwave"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#4a4a4a", ec="black", lw=2))
    ax.add_patch(mpatches.Rectangle((x + 0.15, y + 0.25), w * 0.55, h * 0.6,
                                     fc="#222", ec="#888", lw=1))
    ax.plot(x + w * 0.85, y + h * 0.5, "o", color=COLORS["green"], markersize=5)
    ax.text(x + w / 2, y - 0.25, label, ha="center", fontsize=7, weight="bold")


def _draw_faucet(ax, cx, cy, label="faucet"):
    ax.plot([cx, cx], [cy, cy + 0.8], color="#7f8c8d", lw=4, solid_capstyle="round")
    ax.annotate("", xy=(cx + 0.5, cy + 0.8), xytext=(cx, cy + 0.8),
                arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=4))
    ax.text(cx, cy - 0.3, label, ha="center", fontsize=7, weight="bold")


def _draw_door(ax, x, y, w, h, label="door"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="#a07850", ec="#5c3d1e", lw=2))
    ax.plot(x + w * 0.85, y + h * 0.5, "o", color="gold", markersize=6)
    ax.text(x + w / 2, y - 0.25, label, ha="center", fontsize=7, weight="bold")


def _save(fig, name: str):
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path.relative_to(Path(__file__).parent))


# ── scenario definitions ──────────────────────────────────────────────────

def _make_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    # ─── Pick-and-Place (8) ───────────────────────────────────────────
    # 1
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 3, 5, "red", "red mug"); _draw_shelf(ax, 7, 6, 2, 3, "shelf")
    p = _save(fig, "pnp_01")
    scenarios.append(Scenario(id="pnp_01", category="pick_and_place", image_path=p,
        instruction="Pick up the red mug and place it on the shelf.",
        ground_truth=GroundTruth(
            objects=["red mug", "shelf", "table"],
            spatial_relations=["red mug ON table", "shelf ON table"],
            task_type="pick_and_place", target_object="red mug", destination="shelf",
            action_sequence=["approach red mug", "grasp red mug", "lift red mug",
                             "move to shelf", "place on shelf", "release"])))

    # 2
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 6, 4, "blue", "blue bowl"); _draw_cabinet(ax, 1.5, 6, 2, 2.5, "cabinet")
    p = _save(fig, "pnp_02")
    scenarios.append(Scenario(id="pnp_02", category="pick_and_place", image_path=p,
        instruction="Pick up the blue bowl and put it inside the cabinet.",
        ground_truth=GroundTruth(
            objects=["blue bowl", "cabinet", "table"],
            spatial_relations=["blue bowl ON table", "cabinet ON table"],
            task_type="pick_and_place", target_object="blue bowl", destination="cabinet",
            action_sequence=["approach blue bowl", "grasp blue bowl", "lift blue bowl",
                             "move to cabinet", "place in cabinet", "release"])))

    # 3
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 7, 3, "green", "green mug"); _draw_plate(ax, 3, 7, "plate")
    p = _save(fig, "pnp_03")
    scenarios.append(Scenario(id="pnp_03", category="pick_and_place", image_path=p,
        instruction="Pick up the green mug and place it on the plate.",
        ground_truth=GroundTruth(
            objects=["green mug", "plate", "table"],
            spatial_relations=["green mug ON table", "plate ON table"],
            task_type="pick_and_place", target_object="green mug", destination="plate",
            action_sequence=["approach green mug", "grasp green mug", "lift green mug",
                             "move to plate", "place on plate", "release"])))

    # 4
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_box(ax, 2, 2, 1.5, 1.2, "orange", "orange box")
    _draw_shelf(ax, 6.5, 5, 2.5, 3.5, "shelf")
    p = _save(fig, "pnp_04")
    scenarios.append(Scenario(id="pnp_04", category="pick_and_place", image_path=p,
        instruction="Pick up the orange box and place it on the shelf.",
        ground_truth=GroundTruth(
            objects=["orange box", "shelf", "table"],
            spatial_relations=["orange box ON table", "shelf ON table"],
            task_type="pick_and_place", target_object="orange box", destination="shelf",
            action_sequence=["approach orange box", "grasp orange box", "lift orange box",
                             "move to shelf", "place on shelf", "release"])))

    # 5
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 2, 3, "yellow", "yellow mug"); _draw_microwave(ax, 5.5, 5.5, label="microwave")
    p = _save(fig, "pnp_05")
    scenarios.append(Scenario(id="pnp_05", category="pick_and_place", image_path=p,
        instruction="Pick up the yellow mug and put it in the microwave.",
        ground_truth=GroundTruth(
            objects=["yellow mug", "microwave", "table"],
            spatial_relations=["yellow mug ON table", "microwave ON table"],
            task_type="pick_and_place", target_object="yellow mug", destination="microwave",
            action_sequence=["approach yellow mug", "grasp yellow mug", "lift yellow mug",
                             "move to microwave", "place in microwave", "release"])))

    # 6
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 7, 2.5, "red", "red bowl")
    _draw_box(ax, 1.5, 6, 2, 1.5, "brown", "brown box")
    p = _save(fig, "pnp_06")
    scenarios.append(Scenario(id="pnp_06", category="pick_and_place", image_path=p,
        instruction="Pick up the red bowl and place it in the brown box.",
        ground_truth=GroundTruth(
            objects=["red bowl", "brown box", "table"],
            spatial_relations=["red bowl ON table", "brown box ON table"],
            task_type="pick_and_place", target_object="red bowl", destination="brown box",
            action_sequence=["approach red bowl", "grasp red bowl", "lift red bowl",
                             "move to brown box", "place in brown box", "release"])))

    # 7
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_plate(ax, 3, 3, "plate"); _draw_mug(ax, 3, 6.5, "purple", "purple mug")
    _draw_cabinet(ax, 7, 2, 2, 2.5, "cabinet")
    p = _save(fig, "pnp_07")
    scenarios.append(Scenario(id="pnp_07", category="pick_and_place", image_path=p,
        instruction="Pick up the purple mug and put it in the cabinet.",
        ground_truth=GroundTruth(
            objects=["purple mug", "plate", "cabinet", "table"],
            spatial_relations=["purple mug ON table", "plate ON table", "cabinet ON table"],
            task_type="pick_and_place", target_object="purple mug", destination="cabinet",
            action_sequence=["approach purple mug", "grasp purple mug", "lift purple mug",
                             "move to cabinet", "place in cabinet", "release"])))

    # 8
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 5, 2, "green", "green bowl")
    _draw_mug(ax, 2, 7, "red", "red mug")
    _draw_shelf(ax, 7, 5, 2, 3.5, "shelf")
    p = _save(fig, "pnp_08")
    scenarios.append(Scenario(id="pnp_08", category="pick_and_place", image_path=p,
        instruction="Pick up the green bowl and place it on the shelf.",
        ground_truth=GroundTruth(
            objects=["green bowl", "red mug", "shelf", "table"],
            spatial_relations=["green bowl ON table", "red mug ON table", "shelf ON table"],
            task_type="pick_and_place", target_object="green bowl", destination="shelf",
            action_sequence=["approach green bowl", "grasp green bowl", "lift green bowl",
                             "move to shelf", "place on shelf", "release"])))

    # ─── Open / Close (8) ─────────────────────────────────────────────
    # 9
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_drawer(ax, 3, 4, 4, 2, "top drawer")
    p = _save(fig, "oc_01")
    scenarios.append(Scenario(id="oc_01", category="open", image_path=p,
        instruction="Open the top drawer.",
        ground_truth=GroundTruth(
            objects=["top drawer", "table"],
            spatial_relations=["top drawer ON table"],
            task_type="open", target_object="top drawer", destination=None,
            action_sequence=["approach top drawer", "grasp handle", "pull drawer open"])))

    # 10
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_drawer(ax, 3, 4, 4, 2, "bottom drawer")
    p = _save(fig, "oc_02")
    scenarios.append(Scenario(id="oc_02", category="close", image_path=p,
        instruction="Close the bottom drawer.",
        ground_truth=GroundTruth(
            objects=["bottom drawer", "table"],
            spatial_relations=["bottom drawer ON table"],
            task_type="close", target_object="bottom drawer", destination=None,
            action_sequence=["approach bottom drawer", "grasp handle", "push drawer closed"])))

    # 11
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_cabinet(ax, 3, 3, 3.5, 4, "cabinet")
    p = _save(fig, "oc_03")
    scenarios.append(Scenario(id="oc_03", category="open", image_path=p,
        instruction="Open the cabinet door.",
        ground_truth=GroundTruth(
            objects=["cabinet", "table"],
            spatial_relations=["cabinet ON table"],
            task_type="open", target_object="cabinet", destination=None,
            action_sequence=["approach cabinet", "grasp cabinet handle", "pull cabinet door open"])))

    # 12
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_cabinet(ax, 3.5, 3, 3, 4, "cabinet")
    p = _save(fig, "oc_04")
    scenarios.append(Scenario(id="oc_04", category="close", image_path=p,
        instruction="Close the cabinet door.",
        ground_truth=GroundTruth(
            objects=["cabinet", "table"],
            spatial_relations=["cabinet ON table"],
            task_type="close", target_object="cabinet", destination=None,
            action_sequence=["approach cabinet", "grasp cabinet door", "push cabinet door closed"])))

    # 13
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_door(ax, 3, 1, 4, 8, "door")
    p = _save(fig, "oc_05")
    scenarios.append(Scenario(id="oc_05", category="open", image_path=p,
        instruction="Open the door.",
        ground_truth=GroundTruth(
            objects=["door"],
            spatial_relations=[],
            task_type="open", target_object="door", destination=None,
            action_sequence=["approach door", "grasp door handle", "pull door open"])))

    # 14
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_door(ax, 3, 1, 4, 8, "door")
    p = _save(fig, "oc_06")
    scenarios.append(Scenario(id="oc_06", category="close", image_path=p,
        instruction="Close the door.",
        ground_truth=GroundTruth(
            objects=["door"],
            spatial_relations=[],
            task_type="close", target_object="door", destination=None,
            action_sequence=["approach door", "grasp door handle", "push door closed"])))

    # 15
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_microwave(ax, 3.5, 4, 3, 2, "microwave")
    p = _save(fig, "oc_07")
    scenarios.append(Scenario(id="oc_07", category="open", image_path=p,
        instruction="Open the microwave door.",
        ground_truth=GroundTruth(
            objects=["microwave", "table"],
            spatial_relations=["microwave ON table"],
            task_type="open", target_object="microwave", destination=None,
            action_sequence=["approach microwave", "grasp microwave handle", "pull microwave door open"])))

    # 16
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_microwave(ax, 3.5, 4, 3, 2, "microwave")
    p = _save(fig, "oc_08")
    scenarios.append(Scenario(id="oc_08", category="close", image_path=p,
        instruction="Close the microwave door.",
        ground_truth=GroundTruth(
            objects=["microwave", "table"],
            spatial_relations=["microwave ON table"],
            task_type="close", target_object="microwave", destination=None,
            action_sequence=["approach microwave", "grasp microwave door", "push microwave door closed"])))

    # ─── Turn On / Off (6) ────────────────────────────────────────────
    # 17
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_appliance(ax, 3, 3, 4, 3, "stove", knob=True)
    p = _save(fig, "to_01")
    scenarios.append(Scenario(id="to_01", category="turn_on", image_path=p,
        instruction="Turn on the stove.",
        ground_truth=GroundTruth(
            objects=["stove", "knob", "table"],
            spatial_relations=["stove ON table", "knob ON stove"],
            task_type="turn_on", target_object="stove", destination=None,
            action_sequence=["approach stove", "grasp knob", "rotate knob to turn on"])))

    # 18
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_appliance(ax, 3, 3, 4, 3, "stove", knob=True)
    p = _save(fig, "to_02")
    scenarios.append(Scenario(id="to_02", category="turn_off", image_path=p,
        instruction="Turn off the stove.",
        ground_truth=GroundTruth(
            objects=["stove", "knob", "table"],
            spatial_relations=["stove ON table", "knob ON stove"],
            task_type="turn_off", target_object="stove", destination=None,
            action_sequence=["approach stove", "grasp knob", "rotate knob to turn off"])))

    # 19
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_faucet(ax, 5, 4, "faucet")
    _draw_bowl(ax, 5, 2.5, "gray", "sink")
    p = _save(fig, "to_03")
    scenarios.append(Scenario(id="to_03", category="turn_on", image_path=p,
        instruction="Turn on the sink faucet.",
        ground_truth=GroundTruth(
            objects=["faucet", "sink", "table"],
            spatial_relations=["faucet ABOVE sink", "sink ON table"],
            task_type="turn_on", target_object="faucet", destination=None,
            action_sequence=["approach faucet", "grasp faucet handle", "rotate handle to turn on"])))

    # 20
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_faucet(ax, 5, 4, "faucet")
    _draw_bowl(ax, 5, 2.5, "gray", "sink")
    p = _save(fig, "to_04")
    scenarios.append(Scenario(id="to_04", category="turn_off", image_path=p,
        instruction="Turn off the sink faucet.",
        ground_truth=GroundTruth(
            objects=["faucet", "sink", "table"],
            spatial_relations=["faucet ABOVE sink", "sink ON table"],
            task_type="turn_off", target_object="faucet", destination=None,
            action_sequence=["approach faucet", "grasp faucet handle", "rotate handle to turn off"])))

    # 21
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_microwave(ax, 3, 3.5, 4, 2.5, "microwave")
    p = _save(fig, "to_05")
    scenarios.append(Scenario(id="to_05", category="turn_on", image_path=p,
        instruction="Turn on the microwave.",
        ground_truth=GroundTruth(
            objects=["microwave", "table"],
            spatial_relations=["microwave ON table"],
            task_type="turn_on", target_object="microwave", destination=None,
            action_sequence=["approach microwave", "press power button"])))

    # 22
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_microwave(ax, 3, 3.5, 4, 2.5, "microwave")
    p = _save(fig, "to_06")
    scenarios.append(Scenario(id="to_06", category="turn_off", image_path=p,
        instruction="Turn off the microwave.",
        ground_truth=GroundTruth(
            objects=["microwave", "table"],
            spatial_relations=["microwave ON table"],
            task_type="turn_off", target_object="microwave", destination=None,
            action_sequence=["approach microwave", "press power button"])))

    # ─── Spatial Reasoning (4) ────────────────────────────────────────
    # 23
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 6, 5, "blue", "blue bowl"); _draw_plate(ax, 3, 5, "plate")
    p = _save(fig, "sp_01")
    scenarios.append(Scenario(id="sp_01", category="move", image_path=p,
        instruction="Move the blue bowl to the left of the plate.",
        ground_truth=GroundTruth(
            objects=["blue bowl", "plate", "table"],
            spatial_relations=["blue bowl ON table", "plate ON table", "blue bowl RIGHT OF plate"],
            task_type="move", target_object="blue bowl", destination="left of plate",
            action_sequence=["approach blue bowl", "grasp blue bowl", "lift blue bowl",
                             "move to left of plate", "place blue bowl", "release"])))

    # 24
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 3, 3, "red", "red mug"); _draw_mug(ax, 7, 7, "blue", "blue mug")
    p = _save(fig, "sp_02")
    scenarios.append(Scenario(id="sp_02", category="move", image_path=p,
        instruction="Move the red mug next to the blue mug.",
        ground_truth=GroundTruth(
            objects=["red mug", "blue mug", "table"],
            spatial_relations=["red mug ON table", "blue mug ON table"],
            task_type="move", target_object="red mug", destination="next to blue mug",
            action_sequence=["approach red mug", "grasp red mug", "lift red mug",
                             "move to next to blue mug", "place red mug", "release"])))

    # 25
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 5, 3, "green", "green bowl"); _draw_box(ax, 2, 6, 2, 1.5, "brown", "brown box")
    p = _save(fig, "sp_03")
    scenarios.append(Scenario(id="sp_03", category="move", image_path=p,
        instruction="Move the green bowl behind the brown box.",
        ground_truth=GroundTruth(
            objects=["green bowl", "brown box", "table"],
            spatial_relations=["green bowl ON table", "brown box ON table"],
            task_type="move", target_object="green bowl", destination="behind brown box",
            action_sequence=["approach green bowl", "grasp green bowl", "lift green bowl",
                             "move to behind brown box", "place green bowl", "release"])))

    # 26
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_plate(ax, 3, 5, "plate"); _draw_mug(ax, 7, 5, "yellow", "yellow mug")
    p = _save(fig, "sp_04")
    scenarios.append(Scenario(id="sp_04", category="move", image_path=p,
        instruction="Move the yellow mug in front of the plate.",
        ground_truth=GroundTruth(
            objects=["yellow mug", "plate", "table"],
            spatial_relations=["yellow mug ON table", "plate ON table"],
            task_type="move", target_object="yellow mug", destination="in front of plate",
            action_sequence=["approach yellow mug", "grasp yellow mug", "lift yellow mug",
                             "move to in front of plate", "place yellow mug", "release"])))

    # ─── Multi-step (4) ──────────────────────────────────────────────
    # 27
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 2, 3, "red", "red mug"); _draw_microwave(ax, 5.5, 5, 3, 2, "microwave")
    p = _save(fig, "ms_01")
    scenarios.append(Scenario(id="ms_01", category="pick_and_place", image_path=p,
        instruction="Pick up the red mug, put it in the microwave, then close the microwave door.",
        ground_truth=GroundTruth(
            objects=["red mug", "microwave", "table"],
            spatial_relations=["red mug ON table", "microwave ON table"],
            task_type="pick_and_place", target_object="red mug", destination="microwave",
            action_sequence=["approach red mug", "grasp red mug", "lift red mug",
                             "move to microwave", "place in microwave", "release",
                             "grasp microwave door", "push microwave door closed"])))

    # 28
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_bowl(ax, 2, 3, "blue", "blue bowl"); _draw_faucet(ax, 6, 5, "faucet")
    _draw_bowl(ax, 6, 3.5, "gray", "sink")
    p = _save(fig, "ms_02")
    scenarios.append(Scenario(id="ms_02", category="pick_and_place", image_path=p,
        instruction="Pick up the blue bowl, place it in the sink, then turn on the faucet.",
        ground_truth=GroundTruth(
            objects=["blue bowl", "faucet", "sink", "table"],
            spatial_relations=["blue bowl ON table", "faucet ABOVE sink", "sink ON table"],
            task_type="pick_and_place", target_object="blue bowl", destination="sink",
            action_sequence=["approach blue bowl", "grasp blue bowl", "lift blue bowl",
                             "move to sink", "place in sink", "release",
                             "approach faucet", "grasp faucet handle", "rotate handle to turn on"])))

    # 29
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_mug(ax, 7, 2, "green", "green mug"); _draw_cabinet(ax, 1.5, 5, 2.5, 3, "cabinet")
    p = _save(fig, "ms_03")
    scenarios.append(Scenario(id="ms_03", category="pick_and_place", image_path=p,
        instruction="Open the cabinet, pick up the green mug, and place it inside the cabinet.",
        ground_truth=GroundTruth(
            objects=["green mug", "cabinet", "table"],
            spatial_relations=["green mug ON table", "cabinet ON table"],
            task_type="pick_and_place", target_object="green mug", destination="cabinet",
            action_sequence=["approach cabinet", "grasp cabinet handle", "pull cabinet door open",
                             "approach green mug", "grasp green mug", "lift green mug",
                             "move to cabinet", "place in cabinet", "release"])))

    # 30
    fig, ax = _new_fig(); _draw_table(ax)
    _draw_drawer(ax, 1.5, 2, 3, 1.8, "drawer")
    _draw_mug(ax, 7, 6, "orange", "orange mug")
    p = _save(fig, "ms_04")
    scenarios.append(Scenario(id="ms_04", category="pick_and_place", image_path=p,
        instruction="Open the drawer, pick up the orange mug, and put it in the drawer.",
        ground_truth=GroundTruth(
            objects=["drawer", "orange mug", "table"],
            spatial_relations=["drawer ON table", "orange mug ON table"],
            task_type="pick_and_place", target_object="orange mug", destination="drawer",
            action_sequence=["approach drawer", "grasp handle", "pull drawer open",
                             "approach orange mug", "grasp orange mug", "lift orange mug",
                             "move to drawer", "place in drawer", "release"])))

    return scenarios


# ── entry point ───────────────────────────────────────────────────────────

def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating 30 benchmark scenarios ...")
    scenarios = _make_scenarios()

    out_path = OUT_DIR / "scenarios.json"
    with open(out_path, "w") as f:
        json.dump([s.model_dump() for s in scenarios], f, indent=2, ensure_ascii=False)

    print(f"  -> {len(scenarios)} scenarios written to {out_path}")
    print(f"  -> Images saved to {IMG_DIR}/")

    # quick sanity print
    cats = {}
    for s in scenarios:
        cats[s.category] = cats.get(s.category, 0) + 1
    print(f"  -> Categories: {cats}")


if __name__ == "__main__":
    main()
