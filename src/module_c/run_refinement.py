from __future__ import annotations

import argparse

from .refinement import (
    load_config,
    load_samples,
    refine_samples,
    save_results,
    save_results_pretty,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module C refinement.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--input", required=True, help="Input JSONL samples.")
    parser.add_argument("--output", required=True, help="Output JSONL results.")
    parser.add_argument(
        "--pretty-output",
        default="",
        help="Optional pretty JSON path for human-readable inspection.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_samples(args.input)
    results = refine_samples(samples, cfg)
    save_results(results, args.output)
    if args.pretty_output:
        save_results_pretty(results, args.pretty_output)
    print(f"Processed {len(results)} samples -> {args.output}")
    if args.pretty_output:
        print(f"Pretty JSON saved to: {args.pretty_output}")


if __name__ == "__main__":
    main()

