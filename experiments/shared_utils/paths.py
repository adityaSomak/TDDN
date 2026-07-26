"""Repository-relative path constants.

All paths resolve against ``REPO_ROOT`` (computed from this file's location)
unless overridden by the corresponding environment variable.

Environment overrides
---------------------
    EXPERIMENTS_DATASETS_ROOT      override ``DATASETS_ROOT``
    EXPERIMENTS_CHECKPOINTS_ROOT   override ``CHECKPOINTS_ROOT``
    EXPERIMENTS_FEATURES_ROOT      override ``FEATURES_ROOT``
    EXPERIMENTS_LOCAL_DATA_ROOT    override ``LOCAL_DATA_ROOT``
"""

from __future__ import annotations

import os
from pathlib import Path


_THIS = Path(__file__).resolve()
REPO_ROOT: Path = _THIS.parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


DATASETS_ROOT: Path = _env_path(
    "EXPERIMENTS_DATASETS_ROOT",
    REPO_ROOT / "datasets",
)

CHECKPOINTS_ROOT: Path = _env_path(
    "EXPERIMENTS_CHECKPOINTS_ROOT",
    REPO_ROOT / "experiments" / "shared_utils" / "feature_extraction" / "checkpoints",
)

FEATURES_ROOT: Path = _env_path(
    "EXPERIMENTS_FEATURES_ROOT",
    REPO_ROOT / ".features_cache",
)

# Heavy data that is deliberately not committed and must be supplied locally
# (see datasets/_local/README.md for what belongs here and which tracks need it).
LOCAL_DATA_ROOT: Path = _env_path(
    "EXPERIMENTS_LOCAL_DATA_ROOT",
    REPO_ROOT / "datasets" / "_local",
)


__all__ = ["REPO_ROOT", "DATASETS_ROOT", "CHECKPOINTS_ROOT", "FEATURES_ROOT",
           "LOCAL_DATA_ROOT"]
