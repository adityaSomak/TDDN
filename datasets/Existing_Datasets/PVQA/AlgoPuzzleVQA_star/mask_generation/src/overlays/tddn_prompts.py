"""Class-name prompts for the TDDN tip-adapter pipeline.

One short natural-language phrase per visual class, used to build the
text-prototype classifier in ``tddn_mask.py``.
"""

from __future__ import annotations

_PROMPTS: dict[str, dict[str, str]] = {
    "maze_solve": {
        "path": "a plain empty white maze cell with no markings",
        "S": "a small green right-pointing arrow inside a white maze cell, marking the entry",
        "E": "a small blue right-pointing arrow inside a white maze cell, marking the exit",
    },
    "nqueens": {
        "empty_square": "an empty pink or blue square on a checkered chess board, with no piece on it",
        "queen": "a chess queen — a tall white piece with a many-pointed crown on top",
    },
    "chess": {
        "background":   "the wooden frame around a chess board",
        "light_square": "an empty light-coloured pink chess square with no piece on it",
        "dark_square":  "an empty dark-coloured blue chess square with no piece on it",
        "w_pawn":   "a white chess pawn, a short rounded piece with a small ball on top",
        "w_knight": "a white chess knight, shaped like a horse head",
        "w_bishop": "a white chess bishop, a tall slim piece with a pointed top and a slit",
        "w_rook":   "a white chess rook, a short cylindrical piece with battlements on top",
        "w_queen":  "a white chess queen, a tall piece with a many-pointed crown",
        "w_king":   "a white chess king, the tallest piece, topped with a small cross",
        "b_pawn":   "a black chess pawn, a short rounded dark piece with a small ball on top",
        "b_knight": "a black chess knight, shaped like a dark horse head",
        "b_bishop": "a black chess bishop, a tall slim dark piece with a pointed top and a slit",
        "b_rook":   "a black chess rook, a short cylindrical dark piece with battlements on top",
        "b_queen":  "a black chess queen, a tall dark piece with a many-pointed crown",
        "b_king":   "a black chess king, the tallest dark piece, topped with a small cross",
    },
}


def tddn_prompts(puzzle: str) -> dict[str, str]:
    return _PROMPTS[puzzle]
