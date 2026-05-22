"""Natural-language class prompts for the unified puzzle-perception
segmentation dataset.

The dataset (``datasets/Puzzle_Perception/Segmentation/``) ships a 30-class
label space — eight maze classes, fifteen chess classes, and seven Tower of
Hanoi classes — with one shared offset table. Mask PNGs already store the
unified ids in ``[0, 29]``; no remapping is needed at load time.

Open-vocabulary segmentation matches each patch against text embeddings of
these class names. The names below are written for the text encoder
(human-readable phrases, no underscores, source-task qualifiers where two
classes would otherwise share a name), keyed by the unified id.
"""

# 30 entries indexed by unified class id (matches classes.yaml exactly).
PUZZLE_CLASSES = [
    # ---- maze (ids 0..7) ----
    "maze wall",
    "maze corridor",
    "maze start marker",
    "maze destination A",
    "maze destination B",
    "maze destination C",
    "maze destination D",
    "maze destination E",
    # ---- chess (ids 8..22) ----
    "chess board background",
    "chess white square",
    "chess black square",
    "white pawn chess piece",
    "white knight chess piece",
    "white bishop chess piece",
    "white rook chess piece",
    "white queen chess piece",
    "white king chess piece",
    "black pawn chess piece",
    "black knight chess piece",
    "black bishop chess piece",
    "black rook chess piece",
    "black queen chess piece",
    "black king chess piece",
    # ---- hanoi (ids 23..29) ----
    "tower of hanoi background",
    "tower of hanoi peg",
    "tower of hanoi disk one",
    "tower of hanoi disk two",
    "tower of hanoi disk three",
    "tower of hanoi disk four",
    "tower of hanoi disk five",
]

NUM_CLASSES = len(PUZZLE_CLASSES)  # 30
assert NUM_CLASSES == 30, f"Expected 30 classes, got {NUM_CLASSES}"

# Global-index ranges owned by each source task — useful for per-task mIoU.
DATASET_SLICES = {
    "maze":  (0, 8),
    "chess": (8, 23),
    "hanoi": (23, 30),
}
