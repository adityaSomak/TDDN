"""Prompt registry for the Puzzle_Understanding eval pipeline.

Per-puzzle data lives in sibling modules (``maze_solve.py``, ``nqueens.py``,
``chess.py``, ``checker_move.py``, ``wood_slide.py``). The registry merges
each module with the shared defaults defined at the top of this file so
callers always see the same shape.

This module also owns the dataset path constants. Every consumer of the
prompts package imports the constants from here rather than redefining them.

Public API
----------

    load_prompts()                              -> {puzzle: {full_eval, seg_eval, ...}}
    seg_eval_prompt(puzzle, mode, variant=None) -> str
    full_eval_config(puzzle)                    -> dict (absolute paths)
    tddn_prompts(puzzle)                        -> dict[str, str]
    detect_answer_type(puzzle, record)          -> str

Path constants
--------------

    DATASETS_ROOT   the repository's datasets/ directory (overridable via
                    the EXPERIMENTS_DATASETS_ROOT env var)
    ALGO_FULL       AlgoPuzzleVQA full-eval tree
    ALGO_STAR       AlgoPuzzleVQA_star seg-eval tree (maze + nqueens)
    CHESS_PVQA      Puzzle_Perception chess seg-eval tree
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PUZZLES = ("checker_move", "chess", "maze_solve", "nqueens", "wood_slide")


# ---------- shared defaults --------------------------------------------------
# Merged into every per-puzzle FORMAT_INSTRUCTIONS / SYSTEM_PROMPT in
# `_build_cfg`. Per-task entries override these.

FORMAT_INSTRUCTIONS_DEFAULTS = {
    "number":  "\nRespond with only the number.",
    "boolean": "\nRespond with only True or False.",
}

SYSTEM_PROMPT_SUFFIX = (
    "You must respond with ONLY the final answer in the exact format "
    "requested. Do not include any explanation, reasoning, or extra text."
)


# ---------- dataset path resolution ------------------------------------------

DATASETS_ROOT = Path(os.environ.get(
    "EXPERIMENTS_DATASETS_ROOT",
    Path(__file__).resolve().parents[3] / "datasets",
))

ALGO_FULL = DATASETS_ROOT / "Existing_Datasets" / "PVQA" / "AlgoPuzzleVQA"
ALGO_STAR = DATASETS_ROOT / "Existing_Datasets" / "PVQA" / "AlgoPuzzleVQA_star"
CHESS_PVQA = DATASETS_ROOT / "Puzzle_Perception" / "PVQA" / "test" / "chess"


# ---------- prompt config assembly -------------------------------------------

def _build_cfg(mod) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if hasattr(mod, "SYSTEM_PROMPT") or hasattr(mod, "EVAL_JSONL"):
        sp = getattr(mod, "SYSTEM_PROMPT", "").rstrip()
        if sp and not sp.endswith(SYSTEM_PROMPT_SUFFIX):
            sp = sp + "\n" + SYSTEM_PROMPT_SUFFIX
        eval_jsonl_rel = getattr(mod, "EVAL_JSONL", None)
        cfg["full_eval"] = {
            "eval_jsonl": (ALGO_FULL / eval_jsonl_rel) if eval_jsonl_rel else None,
            "system_prompt": sp,
            "format_instructions": {**FORMAT_INSTRUCTIONS_DEFAULTS,
                                     **getattr(mod, "FORMAT_INSTRUCTIONS", {})},
        }
    if hasattr(mod, "SEG_EVAL"):
        cfg["seg_eval"] = mod.SEG_EVAL
    if hasattr(mod, "TDDN_CLASS_PROMPTS"):
        # Class-name prompts for mask generation. The canonical copy is in
        # datasets/.../mask_generation/src/overlays/tddn_prompts.py — this
        # one is included so the registry is self-contained.
        cfg["tddn_class_prompts"] = mod.TDDN_CLASS_PROMPTS
    return cfg


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, dict[str, Any]]:
    return {name: _build_cfg(importlib.import_module(f".{name}", package=__name__))
            for name in PUZZLES}


def seg_eval_prompt(puzzle: str, mode: str, variant: str | None = None) -> str:
    block = load_prompts()[puzzle]["seg_eval"]
    if variant is not None:
        block = block[variant]
    if mode not in block:
        raise KeyError(f"{puzzle}.seg_eval{f'.{variant}' if variant else ''} has no '{mode}' mode")
    return block[mode].rstrip("\n")


def full_eval_config(puzzle: str) -> dict[str, Any]:
    cfg = load_prompts()[puzzle].get("full_eval")
    if cfg is None:
        raise KeyError(f"{puzzle} has no full_eval section")
    return cfg


def tddn_prompts(puzzle: str) -> dict[str, str]:
    return load_prompts()[puzzle]["tddn_class_prompts"]


def detect_answer_type(puzzle: str, record: dict) -> str:
    """Return the record's `answer_type` field. Every full-eval JSONL row carries it."""
    return record["answer_type"]
