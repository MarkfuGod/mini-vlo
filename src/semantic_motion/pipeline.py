"""End-to-end orchestration for the first two Semantic-Motion streams."""

from __future__ import annotations

from pathlib import Path

from src.scenario import Scenario
from src.semantic_motion.augmentation import AugmentationStream, InstructionRewriter
from src.semantic_motion.models import SemanticMotionRecord
from src.semantic_motion.perception import PerceptionStream
from src.semantic_motion.recognition import RecognitionModel


class SemanticMotionPipeline:
    """Runs Perception followed by Augmentation."""

    def __init__(
        self,
        recognizer: RecognitionModel,
        rewriter: InstructionRewriter | None = None,
    ):
        self.perception = PerceptionStream(recognizer)
        self.augmentation = AugmentationStream(rewriter)

    def run_one(
        self,
        scenario: Scenario,
        image_root: str | Path,
        num_variants: int = 3,
    ) -> SemanticMotionRecord:
        annotation = self.perception.label(scenario, image_root=image_root)
        augmented = self.augmentation.augment(
            annotation,
            num_variants=num_variants,
        )
        return SemanticMotionRecord(
            annotation=annotation,
            augmented_instructions=augmented,
        )

    def run(
        self,
        scenarios: list[Scenario],
        image_root: str | Path,
        num_variants: int = 3,
    ) -> list[SemanticMotionRecord]:
        return [
            self.run_one(
                scenario,
                image_root=image_root,
                num_variants=num_variants,
            )
            for scenario in scenarios
        ]
