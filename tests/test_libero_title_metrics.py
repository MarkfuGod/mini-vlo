from __future__ import annotations

import unittest

from tools.evaluate_libero_goal import evaluate_record, title_slots


class LiberoTitleMetricsTest(unittest.TestCase):
    def test_title_slots(self):
        self.assertEqual(
            title_slots("put the bowl on the plate"),
            {
                "action": "pick_and_place",
                "object": "bowl",
                "destination": "plate",
            },
        )
        self.assertEqual(
            title_slots("turn on the stove")["action"],
            "turn_on",
        )

    def test_single_matching_segment_scores_one(self):
        record = {
            "task_segments": [
                {
                    "start_time_sec": 0.0,
                    "end_time_sec": 4.0,
                    "task_instruction": "put the bowl on the plate",
                    "macro_intent": {
                        "task_type": "pick_and_place",
                        "target_object": "bowl",
                        "destination": "plate",
                    },
                }
            ]
        }
        result = evaluate_record(
            record,
            title="put the bowl on the plate",
            duration_sec=4.0,
        )
        self.assertEqual(result["semantic_label_accuracy"], 1.0)
        self.assertEqual(result["boundary_f1"]["0.5s"]["f1"], 1.0)
        self.assertEqual(result["segment_f1_at_iou"]["0.75"]["f1"], 1.0)
        self.assertEqual(
            result["labeled_end_to_end_segment_f1"]["0.75"]["f1"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
