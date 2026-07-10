from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.scenario import Prediction
from src.semantic_motion import (
    SourceInstructionRewriter,
    VideoTaskPipeline,
    build_temporal_windows,
    build_view_bundle,
    validate_view_bundle,
)
from src.semantic_motion.windowing import aggregate_cut_votes, transition_votes


class DummyMultiViewRecognizer:
    def analyze(self, image_path, instruction):
        return self.analyze_many([image_path], instruction)

    def analyze_many(self, image_paths, instruction):
        del image_paths, instruction
        return Prediction(
            objects=["drawer", "handle"],
            task_type="close",
            domain="manipulation",
            instruction="Close the drawer.",
            action_sequence=["reach handle", "push drawer"],
            action_details=[
                {
                    "text": "reach handle",
                    "verb": "reach",
                    "object": "handle",
                    "body_part": "right hand",
                    "contact_state": "approach",
                    "posture": "standing",
                },
                {
                    "text": "push drawer",
                    "verb": "push",
                    "object": "drawer",
                    "body_part": "right hand",
                    "contact_state": "contact",
                    "posture": "standing",
                },
            ],
            target_object="drawer",
            destination="closed position",
            confidence=0.9,
        )


class WindowAwareRecognizer(DummyMultiViewRecognizer):
    def analyze_many(self, image_paths, instruction):
        paths = [str(path) for path in image_paths]
        if any("frame_00000000" in path for path in paths):
            return super().analyze_many(paths, instruction)
        return Prediction(
            objects=["plate"],
            task_type="rotate",
            domain="manipulation",
            instruction="Rotate the plate.",
            action_sequence=["rotate plate"],
            action_details=[
                {
                    "text": "rotate plate",
                    "verb": "rotate",
                    "object": "plate",
                    "body_part": "right hand",
                    "contact_state": "contact",
                }
            ],
            target_object="plate",
            confidence=0.9,
        )


def write_video(path: Path, frame_count: int = 20, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 32),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer unavailable")
    try:
        for index in range(frame_count):
            writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
    finally:
        writer.release()


class MultiViewContractTest(unittest.TestCase):
    def test_paired_bundle_and_windowed_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = root / "fixed.mp4"
            ego = root / "ego.mp4"
            write_video(fixed)
            write_video(ego)
            bundle = build_view_bundle(
                "paired",
                {"fixed": fixed, "ego": ego},
            )
            self.assertEqual(validate_view_bundle(bundle, require_paired=True), [])

            pipeline = VideoTaskPipeline(
                DummyMultiViewRecognizer(),
                rewriter=SourceInstructionRewriter(),
            )
            record = pipeline.run_view_bundle(
                bundle,
                root / "work",
                view_mode="fused",
                macro_frames=4,
                micro_frames=4,
                num_variants=1,
            )
            self.assertEqual(record.view_bundle.sample_id, "paired")
            self.assertEqual(len(record.task_segments), 1)
            self.assertAlmostEqual(record.task_segments[0].end_time_sec, 2.0)
            self.assertEqual(
                set(record.task_segments[0].evidence_by_view),
                {"fixed", "ego"},
            )
            self.assertEqual(
                record.task_segments[0].micro_instructions[0].body_part,
                "right hand",
            )

    def test_hanning_votes_produce_interior_cut(self):
        windows = build_temporal_windows(
            fps=20,
            frame_count=500,
            window_sec=16,
            step_sec=8,
            frames_per_window=16,
        )
        votes = transition_votes(windows[0], [8])
        cuts = aggregate_cut_votes(votes, fps=20, frame_count=500)
        self.assertEqual(len(cuts), 1)
        self.assertGreater(cuts[0], 0)
        self.assertLess(cuts[0], 500)

    def test_micro_plan_uses_segment_start_window_and_audits_alternatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed, ego = root / "fixed.mp4", root / "ego.mp4"
            write_video(fixed, frame_count=45, fps=10.0)
            write_video(ego, frame_count=45, fps=10.0)
            bundle = build_view_bundle("long", {"fixed": fixed, "ego": ego})
            record = VideoTaskPipeline(
                WindowAwareRecognizer(),
                rewriter=SourceInstructionRewriter(),
            ).run_view_bundle(
                bundle,
                root / "work",
                view_mode="fused",
                macro_frames=4,
                micro_frames=4,
                num_variants=1,
            )
            texts = [
                step.text for step in record.task_segments[0].micro_instructions
            ]
            self.assertNotIn("rotate plate", texts)
            self.assertTrue(
                record.task_segments[0].metadata[
                    "alternative_micro_observations"
                ]
            )


if __name__ == "__main__":
    unittest.main()
