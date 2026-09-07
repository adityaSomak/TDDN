"""Per-version provenance record."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_provenance(path: Path, **fields) -> None:
    path.write_text(json.dumps(fields, indent=2, default=str))
