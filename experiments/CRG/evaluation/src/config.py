"""Dataset paths, decode hyperparameters and the task system prompts.

Paths resolve through ``shared_utils.paths``, so the repo-wide
``EXPERIMENTS_DATASETS_ROOT`` / ``EXPERIMENTS_LOCAL_DATA_ROOT`` overrides apply here
too. The experiment folder holds only code, configs and small result files; every
board image, question spec and answer CSV lives under ``datasets/``.

Two path families, and the split is the point:

``PVQA_*``    the committed CRG dataset. Everything eval needs.
``LEGACY_*``  the 269-sample chess segmentation set, which is NOT committed (see
              ``datasets/_local/README.md``). Only board *regeneration* and
              ``--redetect`` read it; plain eval never touches it.

Model ids and per-model launch flags are not here — they live in
``configs/models.yaml``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
CRG_DIR = _HERE.parents[2]                     # experiments/CRG
EXPERIMENTS_DIR = CRG_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))       # shared_utils.*

from shared_utils.paths import DATASETS_ROOT, LOCAL_DATA_ROOT, REPO_ROOT  # noqa: E402

CONFIG_DIR = CRG_DIR / "configs"
MODELS_YAML = CONFIG_DIR / "models.yaml"

# ---------------------------------------------------------------------------
# Results (small json + tables) live inside the experiment folder
# ---------------------------------------------------------------------------
RESULTS_DIR = CRG_DIR / "evaluation" / "results"
TASK_RESULTS = {"nqueens": RESULTS_DIR / "nqueens", "chess": RESULTS_DIR / "chess"}

# ---------------------------------------------------------------------------
# The committed CRG dataset
# ---------------------------------------------------------------------------
PVQA_ROOT = DATASETS_ROOT / "Puzzle_Perception" / "PVQA"

# Board images are committed unconverted, so the extension differs per task:
# chess boards are generated PNGs, N-Queens boards are the source JPEGs.
BOARD_EXT = {"chess": ".png", "nqueens": ".jpg"}


def task_dir(task: str) -> Path:
    return PVQA_ROOT / task


def questions_path(task: str) -> Path:
    return task_dir(task) / "questions.yaml"


def answers_path(task: str) -> Path:
    return task_dir(task) / "answers.csv"


def detections_path(task: str) -> Path:
    """Cached TDDN detections — what the tddn arm's blacked cells are derived from."""
    return task_dir(task) / "tddn_detections.json"


def board_path(task: str, image_id: str) -> Path:
    return task_dir(task) / "images" / f"{image_id}{BOARD_EXT[task]}"


# ---------------------------------------------------------------------------
# Locally-supplied legacy data (NOT committed) — regeneration / --redetect only
# ---------------------------------------------------------------------------
LEGACY_CHESS_DIR = LOCAL_DATA_ROOT / "chess_seg269" / "data"
LEGACY_CHESS_IMAGES = LEGACY_CHESS_DIR / "images"
LEGACY_CHESS_MASKS = LEGACY_CHESS_DIR / "masks"
LEGACY_CHESS_TEXT_REPR = LEGACY_CHESS_DIR / "text_repr.json"


def require_legacy_chess() -> None:
    """Fail with an actionable message instead of degrading silently.

    Without this, the three legacy-dependent paths each fail unhelpfully: the sprite
    builder returns ``None`` sprites, the TDDN support cache skips every missing pid
    and then crashes inside ``torch.stack([])``, and the validator scores zero boards.
    """
    missing = [p for p in (LEGACY_CHESS_IMAGES, LEGACY_CHESS_MASKS, LEGACY_CHESS_TEXT_REPR)
               if not p.exists()]
    if missing:
        raise SystemExit(
            "the 269-sample chess segmentation set is not committed and is missing:\n"
            + "".join(f"  {p}\n" for p in missing)
            + f"place it under {LOCAL_DATA_ROOT}/chess_seg269/ (or point\n"
              "EXPERIMENTS_LOCAL_DATA_ROOT elsewhere) — see datasets/_local/README.md.\n"
              "Only board regeneration and --redetect need it; plain eval does not.")


def dinov3_root() -> str | None:
    """Meta DINOv3 source tree, via ``DINOV3_ROOT``.

    Returns None when unset, which is correct if ``dinov3`` is pip-installed
    (``pip install -e <path>/dinov3``) — the repo-wide convention, matching
    ``datasets/.../mask_generation/src/overlays/tddn_loader.py``.
    """
    raw = os.environ.get("DINOV3_ROOT")
    return raw or None


# ---------------------------------------------------------------------------
# System prompts (task framing prepended to every question)
# ---------------------------------------------------------------------------
# Kept byte-identical to experiments/Puzzle_Understanding/prompts/nqueens.py's
# SYSTEM_PROMPT_RECALL so the raw baselines of the two experiments are comparable.
NQUEENS_SYSTEM_PROMPT = (
    "You are looking at an N-Queens board: a chess-style grid of square cells in two "
    "alternating colors. Some cells contain a queen (a crown icon); the rest are empty. "
    "The board has no borders or margins — every square is part of the grid — and may be "
    "8x8, 9x9, 10x10, or 11x11. Use 0-based indexing: the top-left cell is (0, 0); row "
    "indices increase downward, column indices increase rightward. "
    "Answer directly and concisely; do not explain."
)
CHESS_SYSTEM_PROMPT = (
    "You are looking at an 8x8 chess board. White pieces are light/hollow; "
    "black pieces are dark/solid. Answer directly and concisely."
)
SYSTEM_PROMPT = {"nqueens": NQUEENS_SYSTEM_PROMPT, "chess": CHESS_SYSTEM_PROMPT}

# ---------------------------------------------------------------------------
# Chess board encoding
# ---------------------------------------------------------------------------
CHESS_CELL_PX = 64                       # boards render at 8 * 64 = 512 px
CHESS_BOARD_N = 8
# token -> per-pixel class id, shared by the renderer, the TDDN support cache and the
# per-cell prediction. 0 is background (the wooden frame): emitted only by the TDDN
# 15-class head, never by the renderer. 1-2 are squares, 3-14 the 12 piece classes.
CHESS_TOKEN2ID = {
    "white_sq": 1, "black_sq": 2,
    "w_pawn": 3, "w_knight": 4, "w_bishop": 5, "w_rook": 6, "w_queen": 7, "w_king": 8,
    "b_pawn": 9, "b_knight": 10, "b_bishop": 11, "b_rook": 12, "b_queen": 13, "b_king": 14,
}
CHESS_PIECES = [t for t, i in CHESS_TOKEN2ID.items() if i >= 3]


def build_prompt(system_prompt: str, question: str, options) -> str:
    """system prompt + question + a closed 'answer with only ...' instruction.

    The closed instruction is load-bearing: it makes the model's FIRST answer token
    the decision, which is what lets the CRG decode read one position instead of
    running a generation loop.
    """
    return (f"{system_prompt}\n\n{question}\nAnswer with only "
            + " or ".join(str(o) for o in options) + ".")


__all__ = [n for n in dir() if not n.startswith("_")]
