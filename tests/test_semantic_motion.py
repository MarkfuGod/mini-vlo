from __future__ import annotations

import unittest

from src.scenario import GroundTruth, Prediction, Scenario
from src.semantic_motion import (
    SemanticMotionPipeline,
    TemplateInstructionRewriter,
    VideoFrame,
    VideoTaskPipeline,
)


class DummyRecognizer:
    def analyze(self, image_path, instruction):
        return Prediction(
            objects=["red mug", "shelf", "table"],
            spatial_relations=["red mug ON table", "shelf ON table"],
            task_type="pick_and_place",
            action_sequence=[
                "approach red mug",
                "grasp red mug",
                "move to shelf",
                "place on shelf",
            ],
            target_object="red mug",
            destination="shelf",
            raw_text='{"task_type": "pick_and_place"}',
        )


class DummyVideoRecognizer:
    def analyze(self, image_path, instruction):
        path = str(image_path)
        if "blue" in path:
            return Prediction(
                objects=["blue bowl", "cabinet", "table"],
                spatial_relations=["blue bowl ON table", "cabinet ON table"],
                task_type="pick_and_place",
                action_sequence=[
                    "approach blue bowl",
                    "grasp blue bowl",
                    "move to cabinet",
                    "place in cabinet",
                ],
                target_object="blue bowl",
                destination="cabinet",
                raw_text='{"target_object": "blue bowl"}',
            )
        return Prediction(
            objects=["red mug", "shelf", "table"],
            spatial_relations=["red mug ON table", "shelf ON table"],
            task_type="pick_and_place",
            action_sequence=[
                "approach red mug",
                "grasp red mug",
                "move to shelf",
                "place on shelf",
            ],
            target_object="red mug",
            destination="shelf",
            raw_text='{"target_object": "red mug"}',
        )


def make_scenario() -> Scenario:
    return Scenario(
        id="pnp_test",
        category="pick_and_place",
        image_path="benchmark/images/pnp_test.png",
        instruction="Pick up the red mug and place it on the shelf.",
        ground_truth=GroundTruth(
            objects=["red mug", "shelf", "table"],
            spatial_relations=["red mug ON table", "shelf ON table"],
            task_type="pick_and_place",
            action_sequence=[
                "approach red mug",
                "grasp red mug",
                "move to shelf",
                "place on shelf",
            ],
            target_object="red mug",
            destination="shelf",
        ),
    )


class SemanticMotionPipelineTest(unittest.TestCase):
    def test_perception_labels_macro_and_micro_instructions(self):
        pipeline = SemanticMotionPipeline(
            recognizer=DummyRecognizer(),
            rewriter=TemplateInstructionRewriter(),
        )
        record = pipeline.run_one(make_scenario(), image_root=".", num_variants=2)

        self.assertEqual(record.annotation.macro_intent.task_type, "pick_and_place")
        self.assertEqual(record.annotation.macro_intent.target_object, "red mug")
        self.assertEqual(record.annotation.macro_intent.destination, "shelf")
        self.assertEqual(len(record.annotation.micro_instructions), 4)
        self.assertEqual(record.annotation.micro_instructions[0].verb, "approach")
        self.assertEqual(record.annotation.micro_instructions[0].object, "red mug")

    def test_augmentation_generates_requested_variants(self):
        pipeline = SemanticMotionPipeline(
            recognizer=DummyRecognizer(),
            rewriter=TemplateInstructionRewriter(),
        )
        record = pipeline.run_one(make_scenario(), image_root=".", num_variants=3)

        self.assertEqual(len(record.augmented_instructions), 3)
        strategies = {item.strategy for item in record.augmented_instructions}
        self.assertIn("source", strategies)
        self.assertIn("intent_paraphrase", strategies)
        self.assertIn("step_expansion", strategies)

    def test_video_pipeline_segments_when_target_changes(self):
        frames = [
            VideoFrame(
                frame_id=0,
                frame_index=0,
                timestamp_sec=0.0,
                image_path="frame_red_0.jpg",
            ),
            VideoFrame(
                frame_id=1,
                frame_index=1,
                timestamp_sec=1.0,
                image_path="frame_red_1.jpg",
            ),
            VideoFrame(
                frame_id=2,
                frame_index=2,
                timestamp_sec=2.0,
                image_path="frame_blue_2.jpg",
            ),
        ]
        pipeline = VideoTaskPipeline(
            recognizer=DummyVideoRecognizer(),
            rewriter=TemplateInstructionRewriter(),
        )
        record = pipeline.run_frames(
            frames,
            video_path="demo.mp4",
            source_instruction="Infer the demonstration tasks.",
            num_variants=2,
        )

        self.assertEqual(len(record.frames), 3)
        self.assertEqual(len(record.task_segments), 2)
        self.assertEqual(record.task_segments[0].macro_intent.target_object, "red mug")
        self.assertEqual(record.task_segments[1].macro_intent.target_object, "blue bowl")
        self.assertEqual(len(record.task_segments[0].augmented_instructions), 2)


if __name__ == "__main__":
    unittest.main()
