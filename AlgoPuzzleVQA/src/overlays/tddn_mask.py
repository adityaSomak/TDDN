"""Generate TDDN-mask overlays for maze, nqueens, or chess.

Per-patch class logits = text_logits + alpha * cache_affinity_logits, bilinearly
upsampled to image resolution and argmaxed to a pixel-level class map. The
class map is alpha-blended onto the image with a per-puzzle palette.

Class prompts are loaded from src/eval/prompts/<puzzle>.py. Tip-Adapter
support cache uses K=10 GT-labelled images per puzzle (see SUPPORT_PIDS_*).

Output: seg_data/<puzzle>/tddn_mask/<pid>_overlay.jpg
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.overlays.mask_utils import parse_grid, wall_mask_from_grey
from src.overlays.tddn_loader import load_alignment_model
from src.eval.prompts import tddn_prompts


DEVICE = "cuda:0"
IMG_SIZE = 1024
PATCH_SIZE = 16
PATCH_GRID = IMG_SIZE // PATCH_SIZE   # 64

ALGO_ROOT = Path(__file__).resolve().parents[2]
SEG_DATA = ALGO_ROOT / "seg_data"
CHESS_DATASET = Path("/data/shanmukha/datasets/chess_dataset/test")

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SUPPORT_PIDS_MAZE = ["0029", "0056", "0065", "0069", "0070",
                     "0095", "0103", "0107", "0117", "0144"]
SUPPORT_PIDS_NQ = ["1860", "0353", "0905", "1289", "1273",
                   "0938", "1731", "0065", "1323", "0056"]
SUPPORT_PIDS_CHESS = [f"00062{i}" for i in range(5, 10)] + [f"0006{i}" for i in (30, 31, 32, 33, 34)]

MAZE_PALETTE = {
    "wall": np.array([200, 120, 60], np.uint8),
    "path": np.array([220, 235, 215], np.uint8),
    "S":    np.array([0, 200, 0], np.uint8),
    "E":    np.array([200, 0, 0], np.uint8),
}
NQUEENS_PALETTE = {"queen": np.array([140, 0, 0], np.uint8)}
CHESS_PALETTE = np.array([
    [0,   0,   0],   [0,   0,   0],   [0,   0,   0],
    [140, 0,   0],   [0,   100, 0],   [0,   0,   140],
    [140, 140, 0],   [120, 0,   120], [0,   100, 120],
    [255, 140, 140], [140, 255, 140], [140, 200, 255],
    [255, 255, 140], [255, 150, 220], [150, 230, 230],
], dtype=np.uint8)
CHESS_PIECE_LO = 3   # don't blend background or empty squares


# Model interface helpers

@torch.no_grad()
def get_patches(model, img_pil: Image.Image) -> torch.Tensor:
    """Return L2-normalised patch embeddings of shape (PATCH_GRID*PATCH_GRID, D)."""
    img_t = PREPROCESS(img_pil).unsqueeze(0).to(DEVICE)
    out = model.image_encoder.forward_head_only(*model.image_encoder.forward_backbone_only(img_t))
    return F.normalize(out.patch_tokens.squeeze(0), dim=-1)


@torch.no_grad()
def encode_text_proto(model, prompts: list[str]) -> torch.Tensor:
    """Mean of L2-normalised patch-half text embeddings, shape (1, D)."""
    tok = model.text_encoder.tokenizer
    enc = tok(prompts, padding="max_length", truncation=True,
              max_length=77, return_tensors="pt")
    bb, _ = model.text_encoder.forward_backbone_only(
        enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    out = model.text_encoder.forward_head_only(bb, enc["attention_mask"].to(DEVICE))
    full = F.normalize(out.aligned, dim=-1)
    patch_half = F.normalize(full[:, full.shape[-1] // 2:], dim=-1)
    return F.normalize(patch_half.mean(0, keepdim=True), dim=-1)


def build_text_classifier(model, class_to_prompt: dict[str, str]) -> torch.Tensor:
    """Stack one prototype per class in dict insertion order."""
    return torch.cat([encode_text_proto(model, [p]) for p in class_to_prompt.values()], dim=0)


def tip_adapter_fuse(
    patches: torch.Tensor,
    text_cls: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_labels: torch.Tensor,
    n_classes: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Fuse text logits with mean-affinity logits over the support cache.

    cv = column-normalised one-hot labels handles class imbalance: each
    cached patch contributes equally per class regardless of class size.
    """
    text_logits = patches @ text_cls.T
    affinity = torch.exp(-beta * (1 - patches @ cache_keys.T))
    cv = F.one_hot(cache_labels, num_classes=n_classes).float()
    cv = cv / cv.sum(0).clamp(min=1).unsqueeze(0)
    return alpha * (affinity @ cv) + text_logits


def patches_to_pixels(logits: torch.Tensor, n_classes: int) -> np.ndarray:
    """Bilinear-upsample per-class logits to IMG_SIZE then argmax."""
    grid = logits.reshape(PATCH_GRID, PATCH_GRID, n_classes).permute(2, 0, 1).unsqueeze(0)
    full = F.interpolate(grid, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return full.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int8)


def cell_centre_patch_idx(r: int, c: int, R: int, C: int) -> int:
    pr = min(int((r + 0.5) * PATCH_GRID / R), PATCH_GRID - 1)
    pc = min(int((c + 0.5) * PATCH_GRID / C), PATCH_GRID - 1)
    return pr * PATCH_GRID + pc


# Maze pipeline

def _maze_cache(model, by_pid: dict, class_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    keys, labels = [], []
    for pid in SUPPORT_PIDS_MAZE:
        if pid not in by_pid:
            continue
        rec = by_pid[pid]
        grid = parse_grid(rec["text_representation_start-position"])
        R, C = len(grid), len(grid[0])
        img_path = ALGO_ROOT / "maze_solve" / rec["image_path"]
        if not img_path.exists():
            continue
        patches = get_patches(model, Image.open(img_path).convert("RGB")).cpu()
        for r in range(R):
            for c in range(C):
                v = grid[r][c].upper()
                cls = {"0": "path", "S": "S", "E": "E"}.get(v)
                if cls is None:
                    continue
                keys.append(patches[cell_centre_patch_idx(r, c, R, C)])
                labels.append(class_idx[cls])
    return torch.stack(keys).to(DEVICE), torch.tensor(labels, dtype=torch.long, device=DEVICE)


def _maze_overlay(orig_rgb: np.ndarray, pred_full: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Composite: walls from grey-threshold, paths/S/E from model prediction."""
    H, W = orig_rgb.shape[:2]
    pred = np.array(Image.fromarray(pred_full.astype(np.uint8)).resize((W, H), Image.NEAREST))
    out = orig_rgb.copy().astype(np.float32)
    wall = wall_mask_from_grey(orig_rgb)
    out[wall] = 0.5 * out[wall] + 0.5 * MAZE_PALETTE["wall"]
    nonwall = ~wall
    for cls_idx, name in enumerate(class_names):
        sel = nonwall & (pred == cls_idx)
        if sel.any():
            out[sel] = 0.5 * out[sel] + 0.5 * MAZE_PALETTE[name].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def run_maze(model, args, out_dir: Path) -> None:
    prompts = tddn_prompts("maze_solve")              # {path, S, E}
    class_names = list(prompts)
    class_idx = {n: i for i, n in enumerate(class_names)}
    text_cls = build_text_classifier(model, prompts)

    rows = list(csv.DictReader(open(ALGO_ROOT / "maze_solve" / "maze_solve_v2.csv")))
    by_pid = {Path(r["image_path"]).parts[-2]: r for r in rows}
    cache_keys, cache_labels = _maze_cache(model, by_pid, class_idx)
    print(f"[tip-cache] maze: {len(cache_keys)} patches across {len(class_names)} classes")

    pids = (args.ids if args.ids else sorted(by_pid))
    if args.limit:
        pids = pids[: args.limit]
    print(f"[run] maze {len(pids)} overlays -> {out_dir}")
    t0 = time.time()
    for i, pid in enumerate(pids):
        rec = by_pid[pid]
        img = Image.open(ALGO_ROOT / "maze_solve" / rec["image_path"]).convert("RGB")
        logits = tip_adapter_fuse(get_patches(model, img), text_cls, cache_keys, cache_labels,
                                  len(class_names), args.alpha, args.beta)
        pred = patches_to_pixels(logits, len(class_names))
        out_arr = _maze_overlay(np.array(img), pred, class_names)
        Image.fromarray(out_arr).save(out_dir / f"{pid}_overlay.jpg", quality=95)
        if (i + 1) % 20 == 0 or i == len(pids) - 1:
            print(f"  [{i+1}/{len(pids)}] elapsed {time.time()-t0:.0f}s")


# NQueens pipeline

def _nqueens_cache(model, by_puzzle: dict) -> tuple[torch.Tensor, torch.Tensor]:
    keys, labels = [], []
    for pid in SUPPORT_PIDS_NQ:
        rec = by_puzzle.get(pid, {}).get("q3")
        if rec is None:
            continue
        queens = ast.literal_eval(rec["answer"])
        N = max(max(r, c) for r, c in queens) + 1
        img_path = ALGO_ROOT / "nqueens" / rec["image_path"]
        if not img_path.exists():
            continue
        patches = get_patches(model, Image.open(img_path).convert("RGB")).cpu()
        queen_set = {(r, c) for r, c in queens}
        for r in range(N):
            for c in range(N):
                keys.append(patches[cell_centre_patch_idx(r, c, N, N)])
                labels.append(1 if (r, c) in queen_set else 0)
    return torch.stack(keys).to(DEVICE), torch.tensor(labels, dtype=torch.long, device=DEVICE)


def _nqueens_overlay(orig_rgb: np.ndarray, pred_full: np.ndarray) -> np.ndarray:
    H, W = orig_rgb.shape[:2]
    mask = np.array(Image.fromarray(pred_full.astype(np.uint8)).resize((W, H), Image.NEAREST))
    out = orig_rgb.copy()
    sel = mask == 1
    if sel.any():
        out[sel] = (0.5 * orig_rgb[sel] + 0.5 * NQUEENS_PALETTE["queen"]).astype(np.uint8)
    return out


def run_nqueens(model, args, out_dir: Path) -> None:
    prompts = tddn_prompts("nqueens")                 # {empty_square, queen}
    text_cls = build_text_classifier(model, prompts)

    with open(ALGO_ROOT / "nqueens" / "nqueens_eval.jsonl") as f:
        recs = [json.loads(l) for l in f]
    by_puzzle: dict[str, dict] = {}
    for r in recs:
        by_puzzle.setdefault(r["puzzle_id"], {})[r["question_id"]] = r

    cache_keys, cache_labels = _nqueens_cache(model, by_puzzle)
    print(f"[tip-cache] nqueens: {len(cache_keys)} patches "
          f"queen={int((cache_labels==1).sum())} empty={int((cache_labels==0).sum())}")

    pids = args.ids if args.ids else sorted(p for p in by_puzzle if "q3" in by_puzzle[p])
    if args.limit:
        pids = pids[: args.limit]
    print(f"[run] nqueens {len(pids)} overlays -> {out_dir}")
    t0 = time.time()
    for i, pid in enumerate(pids):
        rec = by_puzzle[pid]["q3"]
        img = Image.open(ALGO_ROOT / "nqueens" / rec["image_path"]).convert("RGB")
        logits = tip_adapter_fuse(get_patches(model, img), text_cls, cache_keys, cache_labels,
                                  len(prompts), args.alpha, args.beta)
        pred = patches_to_pixels(logits, len(prompts))
        Image.fromarray(_nqueens_overlay(np.array(img), pred)).save(
            out_dir / f"{pid}_overlay.jpg", quality=95)
        if (i + 1) % 20 == 0 or i == len(pids) - 1:
            print(f"  [{i+1}/{len(pids)}] elapsed {time.time()-t0:.0f}s")


# Chess pipeline

def _chess_cache(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch-aligned support cache: one label per patch (dominant pixel class)."""
    keys, labels = [], []
    img_dir, mask_dir = CHESS_DATASET / "images", CHESS_DATASET / "masks"
    for pid in SUPPORT_PIDS_CHESS:
        ip, mp = img_dir / f"{pid}.png", mask_dir / f"{pid}.png"
        if not (ip.exists() and mp.exists()):
            continue
        img = Image.open(ip).convert("RGB")
        mask = np.array(Image.open(mp))
        patches = get_patches(model, img).cpu()
        H, W = mask.shape
        ch, cw = H // PATCH_GRID, W // PATCH_GRID
        for r in range(PATCH_GRID):
            for c in range(PATCH_GRID):
                cell = mask[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                vals, counts = np.unique(cell, return_counts=True)
                dom = int(vals[counts.argmax()])
                if not (0 <= dom <= 14):
                    continue
                keys.append(patches[r * PATCH_GRID + c])
                labels.append(dom)
    return torch.stack(keys).to(DEVICE), torch.tensor(labels, dtype=torch.long, device=DEVICE)


def _chess_overlay(orig_imgsize_rgb: np.ndarray, pred_full: np.ndarray) -> np.ndarray:
    out = orig_imgsize_rgb.copy()
    for cls in range(CHESS_PIECE_LO, 15):
        sel = pred_full == cls
        if sel.any():
            out[sel] = (0.5 * orig_imgsize_rgb[sel] + 0.5 * CHESS_PALETTE[cls]).astype(np.uint8)
    return out


def run_chess(model, args, out_dir: Path) -> None:
    prompts = tddn_prompts("chess")                   # 15 classes in YAML order
    text_cls = build_text_classifier(model, prompts)
    cache_keys, cache_labels = _chess_cache(model)
    by_cls = {int(l): int((cache_labels == l).sum()) for l in cache_labels.unique().tolist()}
    print(f"[tip-cache] chess: {len(cache_keys)} patches across {len(by_cls)} classes -- {by_cls}")

    if args.ids:
        pids = args.ids
    else:
        text_repr = json.load(open(CHESS_DATASET / "text_repr.json"))
        pids = sorted(entry["filename"].replace(".png", "") for entry in text_repr)
    if args.limit:
        pids = pids[: args.limit]
    print(f"[run] chess {len(pids)} overlays -> {out_dir}")
    t0 = time.time()
    for i, pid in enumerate(pids):
        ip = CHESS_DATASET / "images" / f"{pid}.png"
        if not ip.exists():
            print(f"  [WARN] {pid} missing")
            continue
        img = Image.open(ip).convert("RGB")
        logits = tip_adapter_fuse(get_patches(model, img), text_cls, cache_keys, cache_labels,
                                  len(prompts), args.alpha, args.beta)
        pred = patches_to_pixels(logits, len(prompts))
        orig_imgsize = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        Image.fromarray(_chess_overlay(orig_imgsize, pred)).save(
            out_dir / f"{pid}_overlay.jpg", quality=95)
        if (i + 1) % 10 == 0 or i == len(pids) - 1:
            print(f"  [{i+1}/{len(pids)}] elapsed {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--puzzle", choices=["maze", "nqueens", "chess"], required=True)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.5)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = SEG_DATA / args.puzzle / "tddn_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_alignment_model(device=DEVICE, common_grid=PATCH_GRID)
    {"maze": run_maze, "nqueens": run_nqueens, "chess": run_chess}[args.puzzle](model, args, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
