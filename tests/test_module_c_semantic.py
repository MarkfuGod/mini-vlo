from __future__ import annotations

import unittest
from unittest.mock import patch

from src.module_c.semantic_consistency import SemanticConfig, build_verifier


class SemanticVerifierConfigTest(unittest.TestCase):
    def test_qwen3_vl_flash_verifier_uses_generic_dashscope_config(self):
        with patch.dict(
            "os.environ",
            {
                "DASHSCOPE_BASE_URL": "https://example.test/compatible-mode/v1",
                "VLM_MODEL": "qwen3-vl-flash",
            },
        ):
            verifier = build_verifier(
                SemanticConfig(
                    verifier="qwen3-vl-flash",
                    qwen_vl_api_key_env="TEST_QWEN_API_KEY",
                )
            )

        self.assertEqual(verifier.provider, "qwen3-vl-flash")
        self.assertEqual(verifier.base_url, "https://example.test/compatible-mode/v1")
        self.assertEqual(verifier.api_key_env, "TEST_QWEN_API_KEY")
        self.assertEqual(verifier.model, "qwen3-vl-flash")

    def test_qwen3_vl_plus_remains_supported(self):
        verifier = build_verifier(SemanticConfig(verifier="qwen3-vl-plus"))

        self.assertEqual(verifier.provider, "qwen3-vl-plus")
        self.assertEqual(verifier.model, "qwen3-vl-plus")


if __name__ == "__main__":
    unittest.main()
