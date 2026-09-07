"""Path resolution for the recaptioning pipeline.

All heavy data lives outside the repo, under RECAPTION_ROOT (env override), defaulting
to shared_utils.paths.LOCAL_DATA_ROOT/recaptioning -- consistent with the rest of the
repo's convention of resolving heavy/local data through shared_utils.paths rather than
hardcoding absolute paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from shared_utils.paths import LOCAL_DATA_ROOT  # noqa: E402


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


RECAPTION_ROOT: Path = _env_path("RECAPTION_ROOT", LOCAL_DATA_ROOT / "recaptioning")
COORDINATION_DIR: Path = _env_path("RECAPTION_COORDINATION_DIR", RECAPTION_ROOT)

SOURCE_DIR = RECAPTION_ROOT / "source"
MANIFEST_DIR = RECAPTION_ROOT / "manifest"
DOWNLOAD_DIR = RECAPTION_ROOT / "download"
PLAN_DIR = COORDINATION_DIR / "plan"
CAPTIONS_DIR = COORDINATION_DIR / "captions"
VERSIONS_DIR = RECAPTION_ROOT / "versions"
LOGS_DIR = RECAPTION_ROOT / "logs"


def ensure_dirs() -> None:
    for d in (SOURCE_DIR, MANIFEST_DIR, DOWNLOAD_DIR, PLAN_DIR, CAPTIONS_DIR,
              VERSIONS_DIR, LOGS_DIR, CAPTIONS_DIR / ".rejected"):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_ROOT", "RECAPTION_ROOT", "COORDINATION_DIR", "SOURCE_DIR", "MANIFEST_DIR",
    "DOWNLOAD_DIR", "PLAN_DIR", "CAPTIONS_DIR", "VERSIONS_DIR", "LOGS_DIR", "ensure_dirs",
]
