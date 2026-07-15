"""Adapters around existing recognition models.

The Semantic-Motion streams depend only on this small protocol, so a trained
VLM/VLO recognizer can be swapped in without changing the pipeline code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from src.scenario import Prediction


class RecognitionModel(Protocol):
    """Minimal interface required by the perception stream."""

    def analyze(self, image_path: str | Path, instruction: str) -> Prediction:
        """Return structured robot-task understanding for an image/instruction."""

    def analyze_many(
        self,
        image_paths: Iterable[str | Path],
        instruction: str,
    ) -> Prediction:
        """Return one prediction from ordered multi-image evidence."""


class VLMRecognitionModel:
    """Recognition adapter backed by the repo's existing ``VLMEngine``."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ):
        # Lazy import keeps tests and offline framework code independent from
        # the OpenAI client unless this adapter is actually used.
        from src.vlm_engine import VLMEngine

        self.engine = VLMEngine(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model(self) -> str:
        return self.engine.model

    @property
    def base_url(self) -> str:
        return self.engine.base_url

    @property
    def timeout(self) -> float:
        return self.engine.timeout

    def analyze(self, image_path: str | Path, instruction: str) -> Prediction:
        return self.engine.analyze(image_path, instruction)

    def analyze_many(
        self,
        image_paths: Iterable[str | Path],
        instruction: str,
    ) -> Prediction:
        return self.engine.analyze_many(image_paths, instruction)
