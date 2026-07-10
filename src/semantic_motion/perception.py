"""Perception stream for macro-intent and micro-instruction labeling."""

from __future__ import annotations

from pathlib import Path

from src.scenario import Prediction, Scenario
from src.semantic_motion.models import (
    MacroIntent,
    MicroInstruction,
    PerceptionAnnotation,
)
from src.semantic_motion.recognition import RecognitionModel


def _split_action(action: str) -> tuple[str, str]:
    """Best-effort decomposition of a primitive action into verb/object."""
    parts = action.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _prediction_confidence(prediction: Prediction) -> float:
    """Heuristic confidence when the upstream recognizer does not expose one."""
    if prediction.confidence is not None:
        return max(0.0, min(1.0, float(prediction.confidence)))
    checks = [
        bool(prediction.objects),
        bool(prediction.task_type),
        bool(prediction.action_sequence),
        bool(prediction.target_object),
    ]
    return sum(checks) / len(checks)


class PerceptionStream:
    """Labels macro intents and micro instructions with an existing recognizer."""

    def __init__(self, recognizer: RecognitionModel):
        self.recognizer = recognizer

    def label(
        self,
        scenario: Scenario,
        image_root: str | Path,
    ) -> PerceptionAnnotation:
        """Run perception for one benchmark scenario."""
        image_path = Path(image_root) / scenario.image_path
        prediction = self.recognizer.analyze(image_path, scenario.instruction)
        return self.from_prediction(scenario, image_path, prediction)

    def from_prediction(
        self,
        scenario: Scenario,
        image_path: str | Path,
        prediction: Prediction,
    ) -> PerceptionAnnotation:
        """Convert a structured recognition result into Semantic-Motion labels."""
        confidence = _prediction_confidence(prediction)
        macro_intent = MacroIntent(
            task_type=prediction.task_type,
            target_object=prediction.target_object,
            destination=prediction.destination,
            confidence=confidence,
            domain=(
                prediction.domain
                if prediction.domain in {"locomotion", "manipulation", "mixed", "unknown"}
                else "unknown"
            ),
        )

        micro_instructions: list[MicroInstruction] = []
        for idx, action in enumerate(prediction.action_sequence, start=1):
            verb, obj = _split_action(action)
            detail = (
                prediction.action_details[idx - 1]
                if idx <= len(prediction.action_details)
                and isinstance(prediction.action_details[idx - 1], dict)
                else {}
            )
            micro_instructions.append(
                MicroInstruction(
                    step_id=idx,
                    text=str(detail.get("text", action)),
                    verb=str(detail.get("verb", verb)),
                    object=str(detail.get("object", obj)),
                    confidence=confidence,
                    body_part=_optional_text(detail.get("body_part")),
                    contact_state=_optional_text(detail.get("contact_state")),
                    posture=_optional_text(detail.get("posture")),
                    evidence_frame_ids=[
                        int(value)
                        for value in (
                            detail.get("start_image_index"),
                            detail.get("end_image_index"),
                        )
                        if isinstance(value, (int, float))
                    ],
                )
            )

        return PerceptionAnnotation(
            scenario_id=scenario.id,
            image_path=str(image_path),
            source_instruction=scenario.instruction,
            objects=prediction.objects,
            spatial_relations=prediction.spatial_relations,
            macro_intent=macro_intent,
            micro_instructions=micro_instructions,
            raw_recognition=prediction.raw_text,
            metadata={
                "category": scenario.category,
                "recognition_confidence": confidence,
                "prediction_instruction": prediction.instruction,
                "transitions": prediction.transitions,
            },
        )

    def label_many(
        self,
        scenarios: list[Scenario],
        image_root: str | Path,
    ) -> list[PerceptionAnnotation]:
        """Run perception over multiple scenarios."""
        return [self.label(scenario, image_root) for scenario in scenarios]
