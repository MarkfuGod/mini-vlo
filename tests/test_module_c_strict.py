from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.module_c.motion_quality import MotionQualityConfig
from src.module_c.refinement import RefinementConfig, refine_samples
from src.module_c.schema import MotionData, MotionTrack, Sample
from src.module_c.semantic_consistency import (
    SemanticConfig,
    verify_semantic_consistency,
)
from src.module_c.sync_checks import SyncConfig


class FailedVerifier:
    def verify(self, video_path, text):
        del video_path, text
        return {
            "label": "consistent",
            "confidence": 1.0,
            "verifier": "qwen_fallback",
            "error_types": ["qwen_request_failed"],
        }


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 16),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer unavailable")
    try:
        for index in range(20):
            writer.write(np.full((16, 16, 3), index, dtype=np.uint8))
    finally:
        writer.release()


def config(*, allow_mock: bool) -> RefinementConfig:
    return RefinementConfig(
        motion_min_score=0.35,
        motion_cfg=MotionQualityConfig(
            max_velocity=2.5,
            max_acceleration=6.0,
            max_jerk=20.0,
            max_jitter_ratio=0.3,
        ),
        semantic_cfg=SemanticConfig(verifier="mock"),
        semantic_min_confidence=0.7,
        allow_mock_keep=allow_mock,
        sync_cfg=SyncConfig(require_paired_views=True),
    )


class UngatedRefinementTest(unittest.TestCase):
    def test_failed_verifier_is_forced_uncertain(self):
        result, reasons = verify_semantic_consistency(
            "missing.mp4",
            "close the drawer",
            FailedVerifier(),
        )
        self.assertEqual(result["label"], "uncertain")
        self.assertIn("semantic_verifier_failed", reasons)

    def test_missing_motion_and_mock_are_retained_with_diagnostics(self):
        sample = Sample(
            sample_id="missing",
            video_path="missing.mp4",
            text="Approach, grasp, move, place and close the drawer handle.",
        )
        result = refine_samples([sample], config(allow_mock=False))[0]
        self.assertEqual(result.decision, "keep")
        self.assertIn("motion_missing", result.reason_codes)
        self.assertIn("mock_verifier_forbidden", result.reason_codes)
        self.assertIn("quality_gates_disabled", result.reason_codes)

    def test_valid_paired_real_motion_can_keep_in_explicit_mock_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed, ego = root / "fixed.mp4", root / "ego.mp4"
            write_video(fixed)
            write_video(ego)
            timestamps = [index / 10 for index in range(20)]
            positions = [[0.01 * index, 0.0, 0.0] for index in range(20)]
            sample = Sample(
                sample_id="valid",
                video_path=str(fixed),
                views={"fixed": str(fixed), "ego": str(ego)},
                text=(
                    "Approach the drawer handle, grasp it, move it inward, "
                    "and close the drawer."
                ),
                motion=MotionData(
                    tracks={
                        "hand": MotionTrack(
                            positions=positions,
                            timestamps=timestamps,
                        )
                    },
                    spatial_unit="meters",
                    source="test_real",
                ),
                fps=10.0,
                frame_count=20,
                segment_start_sec=0.0,
                segment_end_sec=1.9,
            )
            result = refine_samples([sample], config(allow_mock=True))[0]
            self.assertEqual(result.decision, "keep")


if __name__ == "__main__":
    unittest.main()
