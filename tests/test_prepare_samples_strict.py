from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.module_c.prepare_samples import (
    PrepareSamplesOptions,
    convert_perception_objects,
)


def perception() -> dict:
    return {
        "video_path": "fixed.mp4",
        "view_bundle": {
            "views": {
                "fixed": {"video_path": "fixed.mp4"},
                "ego": {"video_path": "ego.mp4"},
            },
            "timebase": {"fps": 10.0, "frame_count": 20},
            "trajectory": {
                "source": "test",
                "spatial_unit": "meters",
                "coordinate_frame": "world",
            },
            "provenance": {"dataset": "test"},
        },
        "multi_view_frames": [
            {"timestamp_sec": 0.0},
            {"timestamp_sec": 1.9},
        ],
        "task_segments": [
            {
                "segment_id": "segment_000",
                "start_time_sec": 0.0,
                "end_time_sec": 1.9,
                "task_instruction": "Close the drawer.",
            }
        ],
    }


class PrepareSamplesStrictTest(unittest.TestCase):
    def test_missing_motion_is_not_silently_replaced(self):
        result = convert_perception_objects(
            [perception()],
            PrepareSamplesOptions(),
        )
        self.assertEqual(result.written, 0)
        self.assertEqual(result.skipped, 1)

    def test_real_motion_preserves_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            motion_path = Path(tmp) / "motion.json"
            motion_path.write_text(
                json.dumps(
                    {
                        "spatial_unit": "meters",
                        "tracks": {
                            "hand": {
                                "positions": [
                                    [0.0, 0.0, 0.0],
                                    [0.1, 0.0, 0.0],
                                    [0.2, 0.0, 0.0],
                                    [0.3, 0.0, 0.0],
                                ],
                                "timestamps": [0.0, 0.6, 1.2, 1.9],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = convert_perception_objects(
                [perception()],
                PrepareSamplesOptions(motion_path=motion_path),
            )
            self.assertEqual(result.written, 1)
            sample = result.samples[0]
            self.assertEqual(set(sample["views"]), {"fixed", "ego"})
            self.assertEqual(sample["fps"], 10.0)
            self.assertFalse(sample["motion"]["is_dummy"])

    def test_debug_dummy_is_explicitly_marked(self):
        result = convert_perception_objects(
            [perception()],
            PrepareSamplesOptions(allow_dummy_motion=True),
        )
        self.assertEqual(result.written, 1)
        self.assertTrue(result.samples[0]["motion"]["is_dummy"])
        self.assertEqual(result.samples[0]["motion"]["source"], "debug_dummy")


if __name__ == "__main__":
    unittest.main()
