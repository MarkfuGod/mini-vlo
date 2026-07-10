"""External baseline adapters."""

from src.baselines.video2tasks import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    run_upstream_video2tasks,
    upstream_prompt,
)

__all__ = [
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_REVISION",
    "run_upstream_video2tasks",
    "upstream_prompt",
]
