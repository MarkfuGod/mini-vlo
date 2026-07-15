"""Shared runtime helpers for reproducible command-line entry points."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def git_revision(root: str | Path) -> str:
    """Return the current Git revision without mutating repository state."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def file_sha256(path: str | Path) -> str:
    """Hash one input file for result provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    """Hash prompt/config text for result provenance."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON atomically enough for CLI result artifacts."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
