from __future__ import annotations

import unittest

from src.evaluation.corruption import inject_motion_corruption
from src.evaluation.gold import GoldAnnotation, validate_gold
from src.evaluation.metrics import (
    binary_classification_metrics,
    boundary_metrics,
    calibration_metrics,
    labeled_segment_metrics,
    paired_bootstrap_ci,
    segmental_metrics,
    slot_f1,
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

    def test_labeled_segment_f1_requires_semantic_match(self):
        predicted = [
            {"start_sec": 0.0, "end_sec": 2.0},
            {"start_sec": 2.0, "end_sec": 4.0},
        ]
        gold = [
            {"start_sec": 0.0, "end_sec": 2.0},
            {"start_sec": 2.0, "end_sec": 4.0},
        ]
        result = labeled_segment_metrics(
            predicted,
            gold,
            lambda pred_index, gold_index: pred_index == gold_index == 0,
            iou_threshold=0.5,
        )
        self.assertEqual(result["tp"], 1.0)
        self.assertEqual(result["f1"], 0.5)

    def test_slot_f1_and_paired_bootstrap(self):
        empty_slots = slot_f1([], [])
        self.assertEqual(empty_slots["macro_f1"], 0.0)
        self.assertEqual(empty_slots["pair_count"], 0)
        slots = slot_f1(
            [{"action": "place", "object": "red mug", "destination": "shelf"}],
            [{"action": "place", "object": "mug", "destination": "shelf"}],
        )
        self.assertGreater(slots["macro_f1"], 0.0)
        interval = paired_bootstrap_ci([0.1, 0.2, 0.3], draws=500)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLessEqual(interval["lower"], interval["mean_delta"])
        self.assertGreaterEqual(interval["upper"], interval["mean_delta"])

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
