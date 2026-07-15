from __future__ import annotations

import unittest

from compare_video2tasks import build_semantic_motion_prompt, score_output


class CompareFairnessTest(unittest.TestCase):
    def test_semantic_prompt_does_not_receive_filename_or_closed_task_list(self):
        prompt = build_semantic_motion_prompt({"id": "secret_ground_truth_label"})
        self.assertNotIn("secret_ground_truth_label", prompt)
        self.assertNotIn("benchmark task space includes", prompt.lower())
        self.assertIn("without using filenames", prompt.lower())
        self.assertIn("integer image indices only", prompt.lower())

    def test_common_output_scoring_has_no_arbitrary_composite(self):
        sample = {
            "instruction": "Open the bottle.",
            "target_object": "bottle",
            "actions": ["grasp bottle", "twist cap", "open bottle"],
        }
        output = {
            "segments": [
                {
                    "instruction": "Grasp the bottle, twist the cap, and open it.",
                }
            ]
        }
        scores = score_output(sample, output)
        self.assertIn("instruction_token_f1", scores)
        self.assertIn("target_mention_f1", scores)
        self.assertIn("action_coverage_f1", scores)
        self.assertNotIn("composite", scores)


if __name__ == "__main__":
    unittest.main()
