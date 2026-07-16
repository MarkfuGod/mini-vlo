from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import video2tasks

from run_video2tasks import normalize_window_output, run_video, run_views


ROOT = Path(__file__).resolve().parents[1]


class DummyWindowClient:
    def infer(self, images_b64, prompt):
        self.image_count = len(images_b64)
        self.prompt = prompt
        return (
            {
                "thought": "switch halfway",
                "transitions": [8],
                "instructions": ["Pick up the object", "Place the object"],
            },
            "{}",
            {"requests": 1, "input_tokens": 10, "output_tokens": 5},
        )


class DummyMultiViewClient:
    def infer(self, images_b64, prompt):
        self.image_count = len(images_b64)
        self.prompt = prompt
        return (
            {
                "transitions": [],
                "instructions": ["Move the object"],
            },
            "{}",
            {"requests": 1, "input_tokens": 10, "output_tokens": 5},
        )


def write_video(path: Path, frame_count: int = 40) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer unavailable")
    for index in range(frame_count):
        writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
    writer.release()


class VendoredVideo2TasksTest(unittest.TestCase):
    def test_import_resolves_to_vendored_source(self):
        source = Path(inspect.getfile(video2tasks)).resolve()
        self.assertTrue(source.is_relative_to(ROOT / "video2tasks"))

    def test_extra_instructions_are_merged_for_upstream_aggregation(self):
        normalized, warnings = normalize_window_output(
            {
                "transitions": [],
                "instructions": ["Place the banana", "extra label"],
            },
            frame_count=16,
        )
        self.assertEqual(
            normalized["instructions"],
            ["Place the banana; extra label"],
        )
        self.assertEqual(len(warnings), 1)

    def test_direct_runner_uses_vendored_windowing_and_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "sample.mp4"
            write_video(video)

            client = DummyWindowClient()
            result = run_video(
                video,
                sample_id="sample",
                client=client,
                target_width=32,
                target_height=32,
            )

        self.assertEqual(client.image_count, 16)
        self.assertIn("Distinct Object", client.prompt)
        self.assertEqual(result["usage"]["requests"], 1)
        self.assertEqual(len(result["prediction"]["segments"]), 2)

    def test_fused_runner_interleaves_synchronized_fixed_and_ego_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = root / "fixed.mp4"
            ego = root / "ego.mp4"
            write_video(fixed, frame_count=20)
            write_video(ego, frame_count=20)
            client = DummyMultiViewClient()

            result = run_views(
                {"fixed": fixed, "ego": ego},
                sample_id="paired",
                client=client,
                view_mode="fused",
                frames_per_window=4,
                target_width=32,
                target_height=32,
            )

        self.assertEqual(client.image_count, 8)
        self.assertIn("synchronized timestamps", client.prompt)
        self.assertIn("fixed then ego", client.prompt)
        self.assertEqual(result["view_mode"], "fused")
        self.assertEqual(set(result["views"]), {"fixed", "ego"})
        self.assertEqual(
            result["window_outputs"][0]["images_per_timestamp"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
