"""Versioned caption normalizer. Raw vLLM output is stored as-is by run_caption.py;
normalization happens here, at assemble time, so fixing this can't require re-captioning.
"""
from __future__ import annotations

import re
import unicodedata

VERSION = "v1"

_PREAMBLE_RE = re.compile(
    r"^\s*(here'?s?\s+(is\s+)?(a\s+|an\s+|the\s+)?description[:\-]?\s*|"
    r"sure[,!]?\s*|certainly[,!]?\s*)",
    re.IGNORECASE,
)
_MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_caption(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = _PREAMBLE_RE.sub("", text)
    text = _MARKDOWN_BULLET_RE.sub("", text)
    text = text.strip().strip('"').strip("'").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    # strip zero-width / control characters
    text = "".join(c for c in text if c == " " or not unicodedata.category(c).startswith("C"))
    return text.strip()
