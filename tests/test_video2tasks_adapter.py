from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.baselines import UPSTREAM_REVISION, run_upstream_video2tasks


class Video2TasksAdapterTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("video2tasks") is not None,
        "optional requirements-video2tasks.txt is not installed",
    )
    def test_calls_pinned_upstream_windowing_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            for index in range(40):
                writer.write(np.full((16, 16, 3), index, dtype=np.uint8))
            writer.release()

            calls = []

            def infer(frame_ids, window_id):
                calls.append((window_id, len(frame_ids)))
                return {
                    "transitions": [],
                    "instructions": ["Move the object"],
                }

            result = run_upstream_video2tasks(
                video,
                infer,
                sample_id="sample",
            )
            self.assertEqual(
                result["upstream_revision"],
                UPSTREAM_REVISION,
            )
            self.assertEqual(calls, [(0, 16)])
            self.assertEqual(result["segments"][0]["instruction"], "Move the object")


if __name__ == "__main__":
    unittest.main()
