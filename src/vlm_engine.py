"""VLM inference engine — calls Qwen-VL via DashScope OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from dotenv import load_dotenv

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.scenario import Prediction


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _encode_image_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_data_url(image_path: str | Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    return f"data:{mime_type};base64,{_encode_image_base64(image_path)}"


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
            "DASHSCOPE_BASE_URL",
            os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        self.timeout = float(os.getenv("VLM_TIMEOUT", "300"))
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-plus")

        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY "
                "environment variable."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[str | Path] = (),
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], str]:
        """Run a structured multimodal request and return parsed JSON plus raw text."""
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            }
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw = response.choices[0].message.content or ""
        return _parse_json_response(raw), raw

    @staticmethod
    def _prediction(parsed: dict[str, Any], raw: str) -> Prediction:
        return Prediction(
            objects=parsed.get("objects", []),
            spatial_relations=parsed.get("spatial_relations", []),
            task_type=parsed.get("task_type", ""),
            action_sequence=parsed.get("action_sequence", []),
            target_object=parsed.get("target_object", ""),
            destination=parsed.get("destination"),
            domain=parsed.get("domain", "unknown"),
            instruction=parsed.get("instruction", ""),
            transitions=parsed.get("transitions", []),
            confidence=parsed.get("confidence"),
            action_details=parsed.get("action_details", []),
            raw_text=raw,
        )

    def analyze(self, image_path: str | Path, instruction: str) -> Prediction:
        """Send one image plus instruction to the VLM."""
        return self.analyze_many([image_path], instruction)

    def analyze_many(
        self,
        image_paths: Iterable[str | Path],
        instruction: str,
    ) -> Prediction:
        """Analyze ordered evidence from one or more synchronized views."""
        parsed, raw = self.generate_json(
            SYSTEM_PROMPT,
            build_user_prompt(instruction),
            image_paths,
            temperature=0.1,
            max_tokens=1536,
        )
        return self._prediction(parsed, raw)
