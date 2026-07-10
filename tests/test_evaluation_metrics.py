from __future__ import annotations

import unittest

from src.evaluation.corruption import inject_motion_corruption
from src.evaluation.gold import GoldAnnotation, validate_gold
from src.evaluation.metrics import (
    binary_classification_metrics,
    boundary_metrics,
    calibration_metrics,
    segmental_metrics,
)
from src.module_c.motion_quality import MotionQualityConfig, score_motion_quality


class EvaluationMetricsTest(unittest.TestCase):
    def test_boundary_tolerance_is_one_to_one(self):
        result = boundary_metrics([1.1, 5.7, 9.0], [1.0, 6.0], 0.5)
        self.assertEqual(result["tp"], 2.0)
        self.assertEqual(result["fp"], 1.0)
        self.assertEqual(result["fn"], 0.0)

    def test_segmental_iou(self):
        result = segmental_metrics(
            [{"start_sec": 0.0, "end_sec": 2.0, "label": "open"}],
            [{"start_sec": 0.0, "end_sec": 2.5, "label": "open"}],
            0.5,
        )
        self.assertEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["mean_iou"], 0.8)

    def test_refinement_metrics_include_false_keep_and_auroc(self):
        result = binary_classification_metrics(
            ["keep", "drop", "drop", "keep"],
            ["keep", "keep", "drop", "keep"],
            scores=[0.9, 0.6, 0.1, 0.8],
        )
        self.assertAlmostEqual(result["false_keep_rate"], 0.5)
        self.assertGreater(result["auroc"], 0.5)
        calibration = calibration_metrics(
            [True, False, True, True],
            [0.9, 0.6, 0.1, 0.8],
        )
        self.assertIn("ece", calibration)

    def test_spike_corruption_lowers_quality(self):
        timestamps = [index / 20 for index in range(40)]
        positions = [[0.01 * index, 0.0, 0.0] for index in range(40)]
        cfg = MotionQualityConfig(3.0, 20.0, 100.0, 0.3)
        clean = inject_motion_corruption(positions, timestamps, "clean")
        spike = inject_motion_corruption(positions, timestamps, "spike")
        clean_score = score_motion_quality(
            clean.positions,
            clean.timestamps,
            cfg,
        )[0]
        spike_score = score_motion_quality(
            spike.positions,
            spike.timestamps,
            cfg,
        )[0]
        self.assertLess(spike_score, clean_score)

    def test_pending_annotation_is_not_formal_gold(self):
        annotation = GoldAnnotation(
            sample_id="pending",
            source_sample_id="source",
            source_views={"fixed": "fixed.mp4", "ego": "ego.mp4"},
            clip_end_sec=2.0,
            fps=10.0,
            frame_count=20,
        )
        self.assertIn("gold_not_adjudicated", validate_gold(annotation, formal=True))


if __name__ == "__main__":
    unittest.main()
