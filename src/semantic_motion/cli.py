"""Shared CLI construction helpers for Semantic-Motion entry points."""

from __future__ import annotations

import argparse
from typing import Any

from src.semantic_motion.augmentation import (
    LLMInstructionRewriter,
    SourceInstructionRewriter,
    TemplateInstructionRewriter,
)


def add_rewriter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rewriter",
        choices=["llm", "template", "none"],
        default="llm",
        help=(
            "Instruction rewriting backend. 'llm' accepts model output without "
            "code-level fact gates; 'template' is deterministic debug mode."
        ),
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Text rewrite model (defaults to the perception model).",
    )
    parser.add_argument(
        "--augmentation-prompt",
        default=None,
        help="Optional prompt file for the LLM rewriter.",
    )


def build_rewriter(
    args: argparse.Namespace,
    *,
    perception_model: str | None,
) -> Any:
    if args.rewriter == "llm":
        return LLMInstructionRewriter(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.rewrite_model or perception_model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            prompt_file=args.augmentation_prompt,
        )
    if args.rewriter == "template":
        return TemplateInstructionRewriter()
    return SourceInstructionRewriter()
