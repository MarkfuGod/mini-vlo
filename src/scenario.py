"""Data models for robot task scenarios and VLM predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


TASK_TYPES = [
    "pick_and_place",
    "open",
    "close",
    "turn_on",
    "turn_off",
    "move",
]


class GroundTruth(BaseModel):
    """Ground-truth annotation for a single robot task scenario."""

    objects: list[str] = Field(description="Objects visible in the scene")
    spatial_relations: list[str] = Field(
        description="Spatial relationships, e.g. 'red mug ON counter'"
    )
    task_type: str = Field(description="One of TASK_TYPES")
    action_sequence: list[str] = Field(
        description="Ordered primitive actions, e.g. ['approach mug', 'grasp mug', ...]"
    )
    target_object: str = Field(description="Primary object to interact with")
    destination: Optional[str] = Field(
        None, description="Where to move/place the object (if applicable)"
    )


class Scenario(BaseModel):
    """A single benchmark scenario."""

    id: str
    category: str
    image_path: str
    instruction: str
    ground_truth: GroundTruth


class Prediction(BaseModel):
    """Structured prediction returned by the VLM (parsed from JSON)."""

    objects: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    task_type: str = ""
    action_sequence: list[str] = Field(default_factory=list)
    target_object: str = ""
    destination: Optional[str] = None
    raw_text: str = Field("", description="Raw VLM output before parsing")


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [Scenario(**s) for s in data]
