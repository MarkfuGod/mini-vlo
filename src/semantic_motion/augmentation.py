"""Augmentation stream for instruction rewriting."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Protocol

from src.semantic_motion.models import (
    AugmentedInstruction,
    PerceptionAnnotation,
)


class InstructionRewriter(Protocol):
    """Minimal interface for LLM- or template-based instruction augmentation."""

    def rewrite(
        self,
        annotation: PerceptionAnnotation,
        num_variants: int,
    ) -> list[AugmentedInstruction]:
        """Return rewritten instructions for one perception annotation."""


def _norm(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _join_steps(annotation: PerceptionAnnotation) -> str:
    steps = [step.text for step in annotation.micro_instructions]
    if not steps:
        return annotation.source_instruction
    if len(steps) == 1:
        return steps[0]
    return ", then ".join(steps[:-1]) + f", and finally {steps[-1]}"


def _intent_phrase(annotation: PerceptionAnnotation) -> str:
    intent = annotation.macro_intent
    task = intent.task_type.replace("_", " ")
    target = intent.target_object or "the target object"
    if intent.destination:
        return f"{task} {target} to {intent.destination}"
    return f"{task} {target}"


class TemplateInstructionRewriter:
    """Deterministic debug-only rewriter for offline tests."""

    def rewrite(
        self,
        annotation: PerceptionAnnotation,
        num_variants: int,
    ) -> list[AugmentedInstruction]:
        step_ids = [step.step_id for step in annotation.micro_instructions]
        step_plan = _join_steps(annotation)
        intent = _intent_phrase(annotation)
        target = annotation.macro_intent.target_object or "the target object"
        destination = annotation.macro_intent.destination

        candidates = [
            AugmentedInstruction(
                text=annotation.source_instruction,
                strategy="source",
                source_step_ids=step_ids,
            ),
            AugmentedInstruction(
                text=f"Complete the task: {intent}.",
                strategy="intent_paraphrase",
                source_step_ids=step_ids,
            ),
            AugmentedInstruction(
                text=f"Use this sequence to manipulate {target}: {step_plan}.",
                strategy="step_expansion",
                source_step_ids=step_ids,
            ),
        ]

        if destination:
            candidates.append(
                AugmentedInstruction(
                    text=(
                        f"Move {target} so that it ends at {destination}, "
                        f"following these actions: {step_plan}."
                    ),
                    strategy="goal_conditioned",
                    source_step_ids=step_ids,
                )
            )

        for candidate in candidates:
            candidate.metadata["debug_only"] = True
            candidate.metadata["rewriter"] = "template"
        return candidates[: max(0, num_variants)]


class SourceInstructionRewriter:
    """No-op mode that preserves source text without claiming augmentation."""

    def rewrite(
        self,
        annotation: PerceptionAnnotation,
        num_variants: int,
    ) -> list[AugmentedInstruction]:
        if num_variants <= 0:
            return []
        return [
            AugmentedInstruction(
                text=annotation.source_instruction,
                strategy="source_only",
                source_step_ids=[
                    step.step_id for step in annotation.micro_instructions
                ],
                metadata={"rewriter": "none", "augmented": False},
            )
        ]


class LLMInstructionRewriter:
    """OpenAI-compatible instruction augmentation without code-level gating."""

    PROMPT_VERSION = "ungated_rewriter_v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        engine: Any | None = None,
        prompt_file: str | Path | None = None,
    ):
        if engine is None:
            from src.vlm_engine import VLMEngine

            engine = VLMEngine(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
        self.engine = engine
        configured_prompt = prompt_file or os.getenv(
            "AUGMENTATION_PROMPT_FILE",
            "src/semantic_motion/llm_augmentation_prompt.txt",
        )
        prompt_path = Path(configured_prompt)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).resolve().parents[2] / prompt_path
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.prompt_file = str(prompt_path)
        self.last_audit: list[dict[str, Any]] = []

    def rewrite(
        self,
        annotation: PerceptionAnnotation,
        num_variants: int,
    ) -> list[AugmentedInstruction]:
        if num_variants <= 0:
            return []
        user_prompt = (
            f"Generate {num_variants} distinct instruction variants.\n"
            f"Annotation JSON: {annotation.model_dump_json()}\n"
            "Return {\"variants\":[{\"text\":\"...\",\"strategy\":\"...\","
            "\"source_step_ids\":[]}]}."
        )
        parsed, raw = self.engine.generate_json(
            self.system_prompt,
            user_prompt,
            (),
            temperature=0.4,
            max_tokens=1536,
        )
        variants = parsed.get("variants", [])
        if not isinstance(variants, list):
            self.last_audit = [{"included": False, "reason": "invalid_response"}]
            return []

        accepted: list[AugmentedInstruction] = []
        seen: set[str] = set()
        self.last_audit = []
        step_ids = [step.step_id for step in annotation.micro_instructions]
        for item in variants:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            key = _norm(text)
            if not text or key in seen:
                continue
            seen.add(key)
            returned_step_ids = item.get("source_step_ids")
            variant_step_ids = (
                [int(value) for value in returned_step_ids if isinstance(value, int)]
                if isinstance(returned_step_ids, list)
                else step_ids
            )
            accepted.append(
                AugmentedInstruction(
                    text=text,
                    strategy=str(item.get("strategy", "llm_paraphrase")),
                    source_step_ids=variant_step_ids,
                    metadata={
                        "rewriter": "llm",
                        "model": getattr(self.engine, "model", ""),
                        "prompt_version": self.PROMPT_VERSION,
                        "prompt_file": self.prompt_file,
                        "validation": "disabled",
                        "raw_response": raw,
                    },
                )
            )
            self.last_audit.append(
                {"text": text, "included": True, "validation": "disabled"}
            )
            if len(accepted) >= num_variants:
                break
        return accepted


class AugmentationStream:
    """Generates diverse instruction variants from perception annotations."""

    def __init__(self, rewriter: InstructionRewriter | None = None):
        self.rewriter = rewriter or SourceInstructionRewriter()

    def augment(
        self,
        annotation: PerceptionAnnotation,
        num_variants: int = 3,
    ) -> list[AugmentedInstruction]:
        """Run instruction rewriting for one annotation."""
        return self.rewriter.rewrite(annotation, num_variants=num_variants)

    def augment_many(
        self,
        annotations: list[PerceptionAnnotation],
        num_variants: int = 3,
    ) -> list[list[AugmentedInstruction]]:
        """Run instruction rewriting over multiple annotations."""
        return [
            self.augment(annotation, num_variants=num_variants)
            for annotation in annotations
        ]
