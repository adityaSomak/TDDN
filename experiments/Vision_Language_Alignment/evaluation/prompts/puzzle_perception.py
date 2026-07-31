"""Puzzle-Perception segmentation class names (11 classes, ids 0..10).

Human-readable phrases rather than the dataset's snake_case field names, since
these are embedded by the text encoder to form the zero-shot classifier.

The two board-background entries (chess, tower-of-hanoi) are described by their
appearance rather than both being the literal string "background": identical
strings produce identical classifier rows, and an ``argmax`` tie always resolves
to the lower id, so one of them could never be predicted at all.
"""

PUZZLE_PERCEPTION_CLASSES = [
    "black maze wall",
    "light gray maze path",
    "colored circle destination marker",
    "wooden table surface",
    "white chess board square",
    "dark chess board square",
    "white chess piece",
    "black chess piece",
    "pink background wall",
    "brown wooden vertical peg",
    "colored rectangular block",
]

assert len(PUZZLE_PERCEPTION_CLASSES) == 11, \
    f"Expected 11 classes, got {len(PUZZLE_PERCEPTION_CLASSES)}"
