from __future__ import annotations

import unittest

from src.semantic_motion.augmentation import LLMInstructionRewriter
from src.semantic_motion.models import (
    MacroIntent,
    MicroInstruction,
    PerceptionAnnotation,
)


def annotation() -> PerceptionAnnotation:
    return PerceptionAnnotation(
        scenario_id="sample",
        image_path="frame.jpg",
        source_instruction="Move the red mug to the shelf.",
        objects=["red mug", "shelf"],
        macro_intent=MacroIntent(
            task_type="pick_and_place",
            domain="manipulation",
            target_object="red mug",
            destination="shelf",
        ),
        micro_instructions=[
            MicroInstruction(
                step_id=1,
                text="grasp red mug",
                verb="grasp",
                object="red mug",
                body_part="right hand",
                contact_state="grasp",
            ),
            MicroInstruction(
                step_id=2,
                text="place red mug on shelf",
                verb="place",
                object="red mug",
                body_part="right hand",
                contact_state="release",
            ),
        ],
    )


class FakeEngine:
    model = "fake-independent-rewriter"

    def __init__(self, variants):
        self.variants = variants

    def generate_json(self, *args, **kwargs):
        del args, kwargs
        return {"variants": self.variants}, "{}"


class LLMAugmentationTest(unittest.TestCase):
    def test_includes_llm_variant_without_code_level_validation(self):
        engine = FakeEngine(
            [
                {
                    "text": "Grasp the red mug, then place the red mug on the shelf.",
                    "strategy": "imperative",
                }
            ]
        )
        result = LLMInstructionRewriter(engine=engine).rewrite(annotation(), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["validation"], "disabled")

    def test_does_not_reject_llm_output_with_hardcoded_rules(self):
        engine = FakeEngine(
            [
                {
                    "text": (
                        "Grasp the red mug and blue bowl, then place the red mug "
                        "on the shelf."
                    ),
                    "strategy": "hallucinated",
                }
            ]
        )
        rewriter = LLMInstructionRewriter(engine=engine)
        result = rewriter.rewrite(annotation(), 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(rewriter.last_audit[0]["included"])

    def test_preserves_llm_returned_source_step_ids(self):
        engine = FakeEngine(
            [
                {
                    "text": "Pick up the red mug, then put it on the shelf.",
                    "strategy": "safe_synonyms",
                    "source_step_ids": [2],
                }
            ]
        )
        result = LLMInstructionRewriter(engine=engine).rewrite(annotation(), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_step_ids, [2])


if __name__ == "__main__":
    unittest.main()
