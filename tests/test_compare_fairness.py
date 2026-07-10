from __future__ import annotations

import unittest

from compare_video2tasks import build_semantic_motion_prompt


class CompareFairnessTest(unittest.TestCase):
    def test_semantic_prompt_does_not_receive_filename_or_closed_task_list(self):
        prompt = build_semantic_motion_prompt({"id": "secret_ground_truth_label"})
        self.assertNotIn("secret_ground_truth_label", prompt)
        self.assertNotIn("benchmark task space includes", prompt.lower())
        self.assertIn("without using filenames", prompt.lower())


if __name__ == "__main__":
    unittest.main()
