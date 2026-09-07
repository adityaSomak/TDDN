"""YAML config loading with CLI > per-model value > defaults precedence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve(cli_value: Any, model_value: Any, default_value: Any) -> Any:
    """CLI flag > per-model config value > default. None means "not set"."""
    if cli_value is not None:
        return cli_value
    if model_value is not None:
        return model_value
    return default_value
