"""Augmentation stream for instruction rewriting."""

from __future__ import annotations

from typing import Protocol

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
    """Deterministic fallback rewriter for offline framework runs.

    Production use can replace this with an LLM-backed object implementing
    ``InstructionRewriter`` while keeping the stream API unchanged.
    """

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

        return candidates[: max(0, num_variants)]


class AugmentationStream:
    """Generates diverse instruction variants from perception annotations."""

    def __init__(self, rewriter: InstructionRewriter | None = None):
        self.rewriter = rewriter or TemplateInstructionRewriter()

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
