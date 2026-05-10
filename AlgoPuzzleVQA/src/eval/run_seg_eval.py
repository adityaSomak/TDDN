"""Run a VLM seg-eval over maze, nqueens, and/or chess.

Modes:
    raw           raw image only.
    oracle_mask   raw + image with oracle-mask overlay  (seg_data/<puzzle>/oracle_mask/).
    tddn_mask     raw + image with TDDN-mask  overlay   (seg_data/<puzzle>/tddn_mask/).

The two non-raw modes also include a raw run; seg_delta.py averages the two
raw measurements to form the Img baseline used in the delta metrics.

Output JSON: seg_eval_results/<model_tag>_<mode>.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from src.eval import backends
from src.eval.prompts import seg_eval_prompt


ALGO_ROOT = Path(__file__).resolve().parents[2]
SEG_DATA = ALGO_ROOT / "seg_data"
RESULTS_DIR = ALGO_ROOT / "seg_eval_results"
MAZE_DIR = ALGO_ROOT / "maze_solve"
NQ_DIR = ALGO_ROOT / "nqueens"
CHESS_DATASET = Path("/data/shanmukha/datasets/chess_dataset/test")


def _msgs(path: Path, prompt: str) -> list[dict]:
    """Build a single user-message payload (image + text) for a chat-completions call."""
    b64 = backends.encode(path)
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": f"data:{backends.mime_for(path)};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]


# Response parsing

def extract_grid(resp: str) -> list[str]:
    if "GRID:" in resp:
        after = resp.split("GRID:")[-1].strip()
        return [l.strip() for l in after.splitlines() if ";" in l.strip()]
    return [l.strip() for l in resp.splitlines() if ";" in l.strip()]


def extract_int(resp: str) -> int | None:
    if not resp:
        return None
    if "ANSWER:" in resp:
        m = re.search(r"-?\d+", resp.split("ANSWER:")[-1])
        if m:
            return int(m.group())
    nums = re.findall(r"-?\d+", resp)
    return int(nums[-1]) if nums else None


# Scoring

def _iter_cells(grid_lines: list[str], gt_text: str) -> tuple[list[tuple[str, str]], float, bool]:
    """Walk overlapping cells of pred grid and ground-truth grid.

    Returns (pairs, cell_acc_pct, dim_ok). `pairs` is the flat list of
    (pred_value, gt_value) per overlapping cell; per-class TP/FP/FN are
    derived from it by individual scorers below.
    """
    gt_rows = [[c.strip() for c in l.split(";")]
               for l in gt_text.strip().splitlines() if l.strip()]
    pairs: list[tuple[str, str]] = []
    for r in range(min(len(grid_lines), len(gt_rows))):
        pred = [c.strip() for c in grid_lines[r].split(";")]
        for c in range(min(len(pred), len(gt_rows[r]))):
            pairs.append((pred[c], gt_rows[r][c]))
    cor = sum(1 for pv, gv in pairs if pv == gv)
    cell_acc = cor / len(pairs) * 100 if pairs else 0
    pred_N = len(grid_lines[0].split(";")) if grid_lines else 0
    dim_ok = (len(grid_lines) == len(gt_rows)
              and pred_N == (len(gt_rows[0]) if gt_rows else 0))
    return pairs, cell_acc, dim_ok


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) * 100 if tp + fp else 0
    r = tp / (tp + fn) * 100 if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def score_maze(grid_lines: list[str], gt_text: str) -> tuple[float, dict[str, float], bool]:
    pairs, acc, dim_ok = _iter_cells(grid_lines, gt_text)
    f1s: dict[str, float] = {}
    for k in ("1", "0", "S", "E"):
        tp = sum(1 for p, g in pairs if p == k and g == k)
        fp = sum(1 for p, g in pairs if p == k and g != k)
        fn = sum(1 for p, g in pairs if p != k and g == k)
        f1s[k] = _f1(tp, fp, fn)
    return acc, f1s, dim_ok


def score_nq(grid_lines: list[str], gt_text: str) -> tuple[float, float, bool]:
    pairs, acc, dim_ok = _iter_cells(grid_lines, gt_text)
    tp = sum(1 for p, g in pairs if p == "Q" and g == "Q")
    fp = sum(1 for p, g in pairs if p == "Q" and g != "Q")
    fn = sum(1 for p, g in pairs if p != "Q" and g == "Q")
    return acc, _f1(tp, fp, fn), dim_ok


def score_chess_grid(grid_lines: list[str], gt_text: str) -> tuple[float, float, bool]:
    """Exact piece-type-match F1: a wrong piece counts as both FP and FN."""
    pairs, acc, dim_ok = _iter_cells(grid_lines, gt_text)
    pieces = {"1", "2", "3", "4", "5", "6"}
    tp = sum(1 for p, g in pairs if g in pieces and p == g)
    fp = sum(1 for p, g in pairs if p in pieces and p != g)
    fn = sum(1 for p, g in pairs if g in pieces and p != g)
    return acc, _f1(tp, fp, fn), dim_ok


# Task-list construction

def _overlay_dir(puzzle: str, mode: str) -> Path:
    return SEG_DATA / puzzle / mode


# maze and nqueens share an identical (raw + optional overlay) loop, differing
# only in prompt key and where the raw image lives.
SIMPLE_TASKS = {
    "maze": {
        "prompt_key": "maze_solve",
        "raw_img": lambda pid: next(
            iter(list((MAZE_DIR / "images" / pid).glob("*.jpg"))
                 + list((MAZE_DIR / "images" / pid).glob("*.png"))),
            None),
    },
    "nqueens": {
        "prompt_key": "nqueens",
        "raw_img": lambda pid: NQ_DIR / "images" / pid / "nqueens.jpg",
    },
}


def _add_simple(tasks: list[tuple], name: str, gts: dict, args, do_seg: bool) -> None:
    spec = SIMPLE_TASKS[name]
    raw_pr = seg_eval_prompt(spec["prompt_key"], "raw")
    seg_pr = seg_eval_prompt(spec["prompt_key"], args.mode) if do_seg else None
    for pid in sorted(gts[name]):
        raw = spec["raw_img"](pid)
        if raw is None or not raw.exists():
            continue
        seg = _overlay_dir(name, args.mode) / f"{pid}_overlay.jpg" if do_seg else None
        if do_seg and not seg.exists():
            continue
        tasks.append((name, "raw", pid, raw, raw_pr))
        if do_seg:
            tasks.append((name, args.mode, pid, seg, seg_pr))


def build_tasks(args, gts: dict) -> list[tuple]:
    """Return a flat list of (task, variant, pid, image_path, prompt) tuples.

    `variant` is "raw" or args.mode (oracle_mask | tddn_mask) so the scorer
    can bucket raw vs overlay calls.
    """
    tasks: list[tuple] = []
    do_seg = args.mode in ("oracle_mask", "tddn_mask")

    for name in ("maze", "nqueens"):
        if name in args.tasks:
            _add_simple(tasks, name, gts, args, do_seg)

    # (task_name, prompt_variant) for each chess sub-task selected.
    chess_subs: list[tuple[str, str]] = []
    if "chess_count" in args.tasks:
        chess_subs += [("chess_pieces", "pieces"), ("chess_empty", "empty")]
    if "chess_grid" in args.tasks:
        chess_subs += [("chess_black", "grid_black"), ("chess_white", "grid_white")]
    if chess_subs:
        modes = ("raw", *(("oracle_mask", "tddn_mask") if do_seg else ()))
        ch_prompts = {(v, m): seg_eval_prompt("chess", m, variant=v)
                      for _, v in chess_subs for m in modes}
        for pid in sorted(gts["chess"]):
            raw = CHESS_DATASET / "images" / f"{pid}.png"
            seg = _overlay_dir("chess", args.mode) / f"{pid}_overlay.jpg" if do_seg else None
            if not raw.exists() or (do_seg and not seg.exists()):
                continue
            for task_name, variant in chess_subs:
                tasks.append((task_name, "raw", pid, raw, ch_prompts[(variant, "raw")]))
                if do_seg:
                    tasks.append((task_name, args.mode, pid, seg, ch_prompts[(variant, args.mode)]))

    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def load_gts(args) -> dict:
    gts: dict = {"maze": {}, "nqueens": {}, "chess": {}}
    if "maze" in args.tasks:
        with open(MAZE_DIR / "maze_solve_v2.csv") as f:
            for row in csv.DictReader(f):
                gts["maze"][row["image_path"].split("/")[1]] = row["text_representation_start-position"].strip()
    if "nqueens" in args.tasks:
        with open(NQ_DIR / "nqueens_v2.csv") as f:
            for row in csv.DictReader(f):
                gts["nqueens"][row["image_path"].split("/")[1]] = row["text-representation_start-position"].strip()
    if "chess_count" in args.tasks or "chess_grid" in args.tasks:
        for entry in json.loads((CHESS_DATASET / "text_repr.json").read_text()):
            pid = Path(entry["filename"]).stem
            gts["chess"][pid] = {
                "pieces": entry["num_pieces"], "empty": entry["num_empty_cells"],
                "black":  entry["black_repr"], "white": entry["white_repr"],
            }
    return gts


# Per-task scoring dispatch

def _score(task: str, pid: str, resp: str, gts: dict) -> dict[str, float]:
    """Return {metric_name: value} for one (task, response). Drives bucket aggregation."""
    if task == "maze":
        cell, f1s, dim = score_maze(extract_grid(resp), gts["maze"][pid])
        return {"cell": cell, "wall": f1s["1"], "path": f1s["0"], "dim": int(dim)}
    if task == "nqueens":
        cell, f1, dim = score_nq(extract_grid(resp), gts["nqueens"][pid])
        return {"cell": cell, "f1": f1, "dim": int(dim)}
    if task in ("chess_pieces", "chess_empty"):
        kind = "pieces" if task == "chess_pieces" else "empty"
        gt = gts["chess"][pid][kind]
        pred = extract_int(resp)
        return {
            "parse_ok": int(pred is not None),
            "exact":    int(pred == gt),
            "abs_err":  abs(pred - gt) if pred is not None else 64,
        }
    # chess_black / chess_white
    kind = "black" if task == "chess_black" else "white"
    cell, f1, dim = score_chess_grid(extract_grid(resp), gts["chess"][pid][kind])
    return {"cell": cell, "f1": f1, "dim": int(dim)}


# Main

async def _amain(args) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    gts = load_gts(args)
    tasks = build_tasks(args, gts)
    print(f"Model: {args.model}  Backend: {args.backend}  Mode: {args.mode}  Tasks: {args.tasks}")
    print(f"Total calls: {len(tasks)}")

    sem = asyncio.Semaphore(args.concurrency)
    if args.backend == "openai":
        import openai
        client = openai.AsyncOpenAI()
        coros = [backends.call_openai(client, args.model, _msgs(p, prompt), sem, args.reasoning)
                 for _, _, _, p, prompt in tasks]
        responses = await tqdm_asyncio.gather(*coros)
    else:
        async with aiohttp.ClientSession() as session:
            coros = [backends.call_vllm(session, args.ports[i % len(args.ports)],
                                         args.model, _msgs(p, prompt), sem)
                     for i, (_, _, _, p, prompt) in enumerate(tasks)]
            responses = await tqdm_asyncio.gather(*coros)

    # Buckets are allocated lazily — only tasks/variants that fired appear.
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (task, variant, pid, _, _), resp in zip(tasks, responses):
        for k, v in _score(task, pid, resp, gts).items():
            buckets[task][variant][k].append(v)

    overlay_key = args.mode if args.mode != "raw" else None
    bucket_keys = ("raw", overlay_key) if overlay_key else ("raw",)
    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    out: dict = {"model": args.model, "backend": args.backend,
                 "reasoning": args.reasoning, "mode": args.mode}
    for task, sub in buckets.items():
        out[task] = {k: {m: avg(lst) for m, lst in sub[k].items()}
                     for k in bucket_keys if sub.get(k)}

    _print_report(out, bucket_keys)

    model_tag = args.model.split("/")[-1]
    if args.reasoning:
        model_tag += f"_{args.reasoning}"
    out_path = RESULTS_DIR / f"{model_tag}_{args.mode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


def _print_report(out: dict, bucket_keys: tuple[str, ...]) -> None:
    headers = {
        "maze":         ("MAZE",         ["cell", "wall", "path", "dim"]),
        "nqueens":      ("NQUEENS",      ["cell", "f1",   "dim"]),
        "chess_pieces": ("CHESS pieces", ["exact", "abs_err", "parse_ok"]),
        "chess_empty":  ("CHESS empty",  ["exact", "abs_err", "parse_ok"]),
        "chess_black":  ("CHESS black",  ["cell", "f1", "dim"]),
        "chess_white":  ("CHESS white",  ["cell", "f1", "dim"]),
    }
    for t, (lbl, ks) in headers.items():
        if t not in out:
            continue
        print(f"\n=== {lbl} ===")
        print(f"{'Variant':<12} " + "  ".join(f"{k:>6}" for k in ks))
        for v in bucket_keys:
            if v in out[t]:
                row = "  ".join(f"{out[t][v].get(k, 0):>6.1f}" for k in ks)
                print(f"{v:<12} " + row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["openai", "vllm"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["raw", "oracle_mask", "tddn_mask"], required=True,
                   help="raw: raw-image only. oracle_mask / tddn_mask: raw + overlay of that kind.")
    p.add_argument("--tasks", nargs="+", default=["maze", "nqueens"],
                   choices=["maze", "nqueens", "chess_count", "chess_grid"])
    p.add_argument("--ports", nargs="+", type=int, default=[8001, 8002, 8003, 8004])
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--reasoning", default=None,
                   choices=["default", "low", "medium", "high"])
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total number of calls.")
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
