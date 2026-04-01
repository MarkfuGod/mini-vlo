"""VLM inference engine — calls Qwen-VL via DashScope OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.scenario import Prediction


def _encode_image_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """Best-effort extraction of a JSON object from the VLM response."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


class VLMEngine:
    """Thin wrapper around the Qwen-VL (DashScope) OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv(
            "DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-plus")

        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY "
                "environment variable."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def analyze(self, image_path: str | Path, instruction: str) -> Prediction:
        """Send image + instruction to the VLM and return a Prediction."""
        b64 = _encode_image_base64(image_path)
        user_text = build_user_prompt(instruction)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or ""
        parsed = _parse_json_response(raw)

        return Prediction(
            objects=parsed.get("objects", []),
            spatial_relations=parsed.get("spatial_relations", []),
            task_type=parsed.get("task_type", ""),
            action_sequence=parsed.get("action_sequence", []),
            target_object=parsed.get("target_object", ""),
            destination=parsed.get("destination"),
            raw_text=raw,
        )
