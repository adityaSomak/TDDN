"""TDDN: text-promptable segmentation from a fused DINOv3 + CleanDIFT encoder.

TDDN = the fused image encoder + a Tip-Adapter head (text-prototype logits blended
with a few-shot support cache). It localizes the queried region for the deployable
CRG-TDDN arm.

**Not on the default eval path.** Its predictions are cached in the dataset
(``tddn_detections.json``), so eval reads them and never imports this module. This is
only loaded by ``--redetect`` (recompute detections, e.g. after the encoder changes)
and ``--validate-tddn`` (score the chess detector against GT). Both need DINOv3, the
Tip-Adapter checkpoints, and — for chess — the locally-supplied 269-board set.

Two detectors share the same primitives:
  * ``NQueensDetector`` : 3-class {empty, queen, boundary}; returns per-board queen
    bounding boxes (fractional centroid + box) from connected components.
  * ``ChessDetector``   : 15-class chess pieces/squares; returns an 8x8 per-cell
    class-id map. Its support cache is built from the real chess dataset masks.

Tuning constants live in ``configs/models.yaml`` under ``defaults.tddn`` so the
checkpoint choice is declared with the rest of the run configuration.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage
from torchvision import transforms

from . import config, data

PATCH_SIZE = 16
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# N-Queens prompts (3-class). Boundary is a real class: cell-edge support patches keep
# it a 1-patch band so it stops over-claiming cell interiors.
NQUEENS_PROMPTS = {
    "empty_square": "an empty pink or blue square on a checkered chess board, with no piece on it",
    "queen": "a chess queen — a tall white piece with a many-pointed crown on top",
    "boundary": "the thin border line between two adjacent squares on a checkered chess board",
}
_QUEEN_CLASS = 1
# dense empty-cell support: centre + 4 interior points away from edges
_EMPTY_OFFSETS = [(.5, .5), (.3, .3), (.3, .7), (.7, .3), (.7, .7)]

# Chess prompts (15-class, chess-local id order 0..14).
CHESS_PROMPTS = {
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
}

_QBOX_PAD = 6  # px pad around a TDDN queen bbox (native res) before it is blacked


@lru_cache(maxsize=1)
def tuning() -> dict:
    """``defaults.tddn`` from configs/models.yaml."""
    return yaml.safe_load(config.MODELS_YAML.read_text())["defaults"]["tddn"]


def _ensure_paths() -> None:
    """Put shared_utils and (if set) the DINOv3 source tree on sys.path.

    Called from the detector constructors rather than at import time, so importing
    this module does not require the trained-model environment — the same discipline
    ``mask_generation/src/overlays/tddn_loader.py`` documents.
    """
    if str(config.EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(config.EXPERIMENTS_DIR))
    root = config.dinov3_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)


class _FusedEncoder:
    """Shared fused-encoder + Tip-Adapter primitives (model loaded once)."""

    def __init__(self, device: str = "cuda", grid: int | None = None):
        _ensure_paths()
        try:
            from shared_utils.feature_extraction.loaders import load_fused_dinov3_cd
        except ImportError as e:
            raise SystemExit(
                f"cannot import the fused DINOv3 encoder ({e}).\n"
                "Set DINOV3_ROOT=/path/to/dinov3, or `pip install -e <path>/dinov3`.") from e

        t = tuning()
        self.device = device
        self.grid = grid or t["grid"]
        self.img_size = self.grid * PATCH_SIZE
        self.tip_alpha, self.tip_beta = t["tip_alpha"], t["tip_beta"]
        ckpts = tuple(t["checkpoint_step"])
        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        print(f"loading TDDN (fused dinov3+cleandift, avg ckpts {ckpts}) ...", flush=True)
        self.model, _ = load_fused_dinov3_cd(device, avg_ckpts=ckpts,
                                             common_grid_override=self.grid)
        self.model.eval()
        self.prompts: dict[str, str] = {}
        self.text_cls: torch.Tensor | None = None     # (n_classes, D)
        self.cache_keys: torch.Tensor | None = None
        self.cache_labels: torch.Tensor | None = None

    @property
    def n_classes(self) -> int:
        return len(self.prompts)

    @torch.no_grad()
    def patches(self, img: Image.Image) -> torch.Tensor:
        """L2-normalized patch embeddings (n_patches, D) for one image."""
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        out = self.model.image_encoder.forward_head_only(
            *self.model.image_encoder.forward_backbone_only(img_t))
        return F.normalize(out.patch_tokens.squeeze(0), dim=-1)

    @torch.no_grad()
    def _text_prototype(self, prompt: str) -> torch.Tensor:
        tok = self.model.text_encoder.tokenizer
        enc = tok([prompt], padding="max_length", truncation=True, max_length=77,
                  return_tensors="pt")
        bb, _ = self.model.text_encoder.forward_backbone_only(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        out = self.model.text_encoder.forward_head_only(bb, enc["attention_mask"].to(self.device))
        full = F.normalize(out.aligned, dim=-1)
        patch_half = F.normalize(full[:, full.shape[-1] // 2:], dim=-1)
        return F.normalize(patch_half.mean(0, keepdim=True), dim=-1)

    def set_prompts(self, prompts: dict[str, str]) -> None:
        self.prompts = dict(prompts)
        self.text_cls = torch.cat([self._text_prototype(p) for p in prompts.values()], dim=0)

    def _set_cache(self, keys: list, labels: list) -> None:
        if not keys:
            raise SystemExit("TDDN support cache is empty — no support board yielded any "
                             "patch. Check the support pid list and the mask files.")
        self.cache_keys = torch.stack(keys).to(self.device)
        self.cache_labels = torch.tensor(labels, dtype=torch.long, device=self.device)

    def fuse(self, patches: torch.Tensor) -> torch.Tensor:
        """Tip-Adapter: text-prototype logits + cache-affinity logits -> (P, n_classes)."""
        text_logits = patches @ self.text_cls.T
        affinity = torch.exp(-self.tip_beta * (1 - patches @ self.cache_keys.T))
        cv = F.one_hot(self.cache_labels, num_classes=self.n_classes).float()
        cv = cv / cv.sum(0).clamp(min=1).unsqueeze(0)
        return self.tip_alpha * (affinity @ cv) + text_logits

    def patch_labels(self, img: Image.Image) -> np.ndarray:
        """Per-patch argmax labels reshaped to (grid, grid)."""
        logits = self.fuse(self.patches(img))
        return logits.argmax(dim=-1).reshape(self.grid, self.grid).cpu().numpy()

    def pixel_labels(self, img: Image.Image) -> np.ndarray:
        """Per-patch argmax then NEAREST upsample to (img_size, img_size)."""
        lab = self.patch_labels(img).astype(np.uint8)
        return np.array(Image.fromarray(lab).resize((self.img_size, self.img_size),
                                                     Image.NEAREST))


class NQueensDetector(_FusedEncoder):
    """Detect queens (boxes + centroids) on N-Queens boards via TDDN."""

    def build_cache(self) -> None:
        """Build the Tip-Adapter cache from the pinned support boards.

        The support boards are 10 of the same 100 boards the eval scores, so those 10
        contribute cache entries derived from their own GT queen cells. The support
        list is pinned as a literal in configs/models.yaml rather than sliced out of
        the board ordering at runtime, so it cannot drift when the board list changes.
        """
        self.set_prompts(NQUEENS_PROMPTS)
        rows = {r["image_id"]: r["board"] for r in data.load_rows("nqueens")}
        keys, labels = [], []

        def patch_idx(fy: float, fx: float) -> int:
            pr = min(int(fy * self.grid), self.grid - 1)
            pc = min(int(fx * self.grid), self.grid - 1)
            return pr * self.grid + pc

        for pid in tuning()["nqueens_support_pids"]:
            grid = data.parse_grid(rows[pid])
            N = len(grid)
            patches = self.patches(data.load_board_image("nqueens", pid)).cpu()
            for r in range(N):
                for c in range(N):
                    if grid[r][c].upper() == "Q":
                        keys.append(patches[patch_idx((r + .5) / N, (c + .5) / N)])
                        labels.append(_QUEEN_CLASS)
                    else:
                        seen = set()
                        for oy, ox in _EMPTY_OFFSETS:
                            idx = patch_idx((r + oy) / N, (c + ox) / N)
                            if idx not in seen:
                                seen.add(idx)
                                keys.append(patches[idx])
                                labels.append(0)
            for r in range(N):                 # vertical interior edges -> boundary
                for c in range(N - 1):
                    keys.append(patches[patch_idx((r + .5) / N, (c + 1.0) / N)])
                    labels.append(2)
            for r in range(N - 1):             # horizontal interior edges -> boundary
                for c in range(N):
                    keys.append(patches[patch_idx((r + 1.0) / N, (c + .5) / N)])
                    labels.append(2)
        self._set_cache(keys, labels)

    def detect(self, img: Image.Image) -> list[dict]:
        """Queen detections as [{'cy', 'cx', 'box': [fy0, fy1, fx0, fx1]}] (fractional)."""
        W, H = img.size
        lab = self.patch_labels(img)
        comps, n = ndimage.label(lab == _QUEEN_CLASS)
        out: list[dict] = []
        for k in range(1, n + 1):
            ys, xs = np.where(comps == k)
            if len(ys) == 0:
                continue
            y0 = max(0, int(ys.min() * H / self.grid) - _QBOX_PAD)
            y1 = min(H, int((ys.max() + 1) * H / self.grid) + _QBOX_PAD)
            x0 = max(0, int(xs.min() * W / self.grid) - _QBOX_PAD)
            x1 = min(W, int((xs.max() + 1) * W / self.grid) + _QBOX_PAD)
            out.append({"cy": (ys.mean() + .5) / self.grid,
                        "cx": (xs.mean() + .5) / self.grid,
                        "box": [y0 / H, y1 / H, x0 / W, x1 / W]})
        return out


class ChessDetector(_FusedEncoder):
    """Predict an 8x8 per-cell class-id map on chess boards via TDDN."""

    def build_cache(self) -> None:
        """Cache from the 10 lowest-numbered real boards' per-pixel GT masks.

        These are real photographed-style renders from the 269-board segmentation set,
        entirely separate from the 800 synthetic CRG boards the eval scores, and they
        are excluded from ``validate()``'s scored set below.
        """
        config.require_legacy_chess()
        self.set_prompts(CHESS_PROMPTS)
        keys, labels = [], []
        for pid in tuning()["chess_support_pids"]:
            ip = config.LEGACY_CHESS_IMAGES / f"{pid}.png"
            mp = config.LEGACY_CHESS_MASKS / f"{pid}.png"
            if not (ip.exists() and mp.exists()):
                raise SystemExit(f"chess support board {pid} missing: {ip} / {mp}")
            patches = self.patches(Image.open(ip).convert("RGB")).cpu()
            mask = np.array(Image.open(mp))
            H, W = mask.shape
            ch, cw = H // self.grid, W // self.grid
            for r in range(self.grid):
                for c in range(self.grid):
                    cell = mask[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                    vals, counts = np.unique(cell, return_counts=True)
                    dom = int(vals[counts.argmax()])
                    if 0 <= dom <= 14:
                        keys.append(patches[r * self.grid + c])
                        labels.append(dom)
        self._set_cache(keys, labels)

    def detect(self, img: Image.Image) -> list[list[int]]:
        """8x8 per-cell label = mode over the central half of each cell block."""
        pred_full = self.pixel_labels(img)
        N = config.CHESS_BOARD_N
        cs = self.img_size // N
        out = [[0] * N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                y0, x0 = r * cs + cs // 4, c * cs + cs // 4
                blk = pred_full[y0:y0 + cs // 2, x0:x0 + cs // 2]
                vals, counts = np.unique(blk, return_counts=True)
                out[r][c] = int(vals[counts.argmax()])
        return out


# ---------------------------------------------------------------------------
# redetect: refresh the cached detections the tddn arm reads
# ---------------------------------------------------------------------------
def redetect(task: str) -> dict:
    """Re-run the detector over every board and return the detections map.

    Same shape as the committed ``tddn_detections.json``, so the caller can either
    diff against it or overwrite it.
    """
    ids = sorted({r["image_id"] for r in data.load_rows(task)})
    det = ChessDetector() if task == "chess" else NQueensDetector()
    det.build_cache()
    out = {}
    for i, iid in enumerate(ids):
        out[iid] = det.detect(data.load_board_image(task, iid))
        if (i + 1) % 100 == 0:
            print(f"  TDDN {i + 1}/{len(ids)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# validate: chess piece-detection quality vs GT (a diagnostic, not an eval arm)
# ---------------------------------------------------------------------------
_TYPES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
_ID2NAME = {v: k for k, v in config.CHESS_TOKEN2ID.items()}


def validate_chess() -> dict:
    """Score the chess detector on the real boards it was NOT cached from.

    Piece-presence P/R/F1, plus TYPE / COLOR / EXACT accuracy on localized pieces and
    per-class cell F1. GT is used only for scoring. Needs the locally-supplied 269-board
    set (both the images and their text_repr GT grids).
    """
    config.require_legacy_chess()
    det = ChessDetector()
    det.build_cache()

    entries = json.loads(config.LEGACY_CHESS_TEXT_REPR.read_text())
    boards = {e["filename"].replace(".png", ""): e["text_representation"] for e in entries}
    support = set(tuning()["chess_support_pids"])
    pids = [p for p in sorted(boards) if p not in support]
    print(f"scoring {len(pids)} non-support boards", flush=True)

    is_piece = lambda i: i >= 3
    color = lambda i: 0 if i <= 8 else 1        # 0 white, 1 black
    ptype = lambda i: (i - 3) % 6

    TP = FP = FN = TN = loc = type_ok = color_ok = exact = 0
    cls_tp, cls_fp, cls_fn = defaultdict(int), defaultdict(int), defaultdict(int)
    N = config.CHESS_BOARD_N
    for i, pid in enumerate(pids):
        pred = det.detect(Image.open(config.LEGACY_CHESS_IMAGES / f"{pid}.png").convert("RGB"))
        rows = boards[pid].strip().split("\n")
        for r in range(N):
            toks = rows[r].split(":")
            for c in range(N):
                gt, pr = config.CHESS_TOKEN2ID[toks[c]], pred[r][c]
                go, po = is_piece(gt), is_piece(pr)
                TP += go and po; FP += (not go) and po
                FN += go and (not po); TN += (not go) and (not po)
                if go and po and gt == pr:
                    cls_tp[gt] += 1
                if po and (not go or gt != pr):
                    cls_fp[pr] += 1
                if go and (not po or gt != pr):
                    cls_fn[gt] += 1
                if go and po:
                    loc += 1
                    tc, cc = ptype(gt) == ptype(pr), color(gt) == color(pr)
                    type_ok += tc; color_ok += cc; exact += tc and cc
        if (i + 1) % 25 == 0 or i == len(pids) - 1:
            print(f"  [{i + 1}/{len(pids)}]", flush=True)

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    perclass = {}
    for cid in range(3, 15):
        tp, fp, fn = cls_tp[cid], cls_fp[cid], cls_fn[cid]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        perclass[_ID2NAME[cid]] = round(2 * p * rc / (p + rc) if p + rc else 0.0, 3)

    return {"n_boards": len(pids),
            "presence": {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3)},
            "localized_n": loc,
            "type_acc": round(type_ok / loc, 3) if loc else None,
            "color_acc": round(color_ok / loc, 3) if loc else None,
            "exact_acc": round(exact / loc, 3) if loc else None,
            "perclass_f1": perclass}
