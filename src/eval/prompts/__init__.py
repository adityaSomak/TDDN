"""Prompt registry for puzzle eval pipelines.

Per-puzzle data lives in sibling modules (maze_solve.py, nqueens.py, chess.py,
checker_move.py, wood_slide.py). Shared boilerplate lives in _common.py and is
merged into each puzzle by load_prompts() so callers always see the same shape.

Public API:
    load_prompts()                       -> {puzzle: {full_eval, seg_eval, tddn_class_prompts}}
    seg_eval_prompt(puzzle, mode, variant=None) -> str
    full_eval_config(puzzle)             -> dict
    tddn_prompts(puzzle)                 -> dict[str, str]
    detect_answer_type(puzzle, record)   -> str
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from . import _common


PUZZLES = ("checker_move", "chess", "maze_solve", "nqueens", "wood_slide")


def _build_cfg(mod) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if hasattr(mod, "SYSTEM_PROMPT") or hasattr(mod, "EVAL_JSONL"):
        sp = getattr(mod, "SYSTEM_PROMPT", "").rstrip()
        suffix = _common.SYSTEM_PROMPT_SUFFIX
        if sp and not sp.endswith(suffix):
            sp = sp + "\n" + suffix
        fe: dict[str, Any] = {
            "eval_jsonl": getattr(mod, "EVAL_JSONL", None),
            "answer_type_source": getattr(mod, "ANSWER_TYPE_SOURCE", "record_field"),
            "system_prompt": sp,
            "format_instructions": {**_common.FORMAT_INSTRUCTIONS,
                                     **getattr(mod, "FORMAT_INSTRUCTIONS", {})},
        }
        if hasattr(mod, "DERIVE_RULES"):
            fe["derive_rules"] = mod.DERIVE_RULES
        cfg["full_eval"] = fe
    if hasattr(mod, "SEG_EVAL"):
        cfg["seg_eval"] = mod.SEG_EVAL
    if hasattr(mod, "TDDN_CLASS_PROMPTS"):
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
    cfg = full_eval_config(puzzle)
    src = cfg.get("answer_type_source", "record_field")
    if src == "record_field":
        return record["answer_type"]
    if src != "derived":
        raise ValueError(f"unknown answer_type_source '{src}' for {puzzle}")
    rules = cfg["derive_rules"]
    if rules.get("mcq_when_options") and record.get("options") is not None:
        return "mcq"
    qprefix = record["question_id"].split("_")[0]
    if qprefix in rules.get("question_id_overrides", {}):
        return rules["question_id_overrides"][qprefix]
    ans = record.get("answer", "")
    if ans in rules.get("answer_text_overrides", {}):
        return rules["answer_text_overrides"][ans]
    for prefix, t in rules.get("answer_starts_with", {}).items():
        if ans.startswith(prefix):
            return t
    return rules["default"]
