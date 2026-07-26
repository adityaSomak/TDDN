"""Run a VLM seg-eval over maze, nqueens, and/or chess.

Modes:
    raw           raw image only.
    oracle_mask   raw + image with oracle-mask overlay (under each puzzle's seg_data/oracle_mask/).
    tddn_mask     raw + image with TDDN-mask overlay   (under each puzzle's seg_data/tddn_mask/).

The two non-raw modes also fire the raw run; the analyzer averages the two
raw measurements to form the ``Img`` baseline used in the delta metrics.

Output JSON: ``results/seg_eval/<model_tag>[_<reasoning>]_<mode>.json``.

The same script also serves as the reporter for already-computed results
when invoked with ``--analyze``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm_asyncio

import utils
from prompts import ALGO_STAR, CHESS_SEG269, seg_eval_prompt


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "seg_eval"


def _msgs(path: Path, prompt: str) -> list[dict]:
    """Build a single user-message payload (image + text) for a chat-completions call."""
    b64 = utils.encode(path)
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": f"data:{utils.mime_for(path)};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]


# ---------- response parsing -------------------------------------------------

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


# ---------- scoring ----------------------------------------------------------

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


# ---------- task-list construction -------------------------------------------

MAZE_DATA = ALGO_STAR / "maze" / "data"
NQ_DATA = ALGO_STAR / "nqueens" / "data"
MAZE_SEG = ALGO_STAR / "maze" / "seg_data"
NQ_SEG = ALGO_STAR / "nqueens" / "seg_data"
CHESS_DATA = CHESS_SEG269 / "data"
CHESS_SEG = CHESS_SEG269 / "seg_data"


SIMPLE_TASKS = {
    "maze": {
        "prompt_key": "maze_solve",
        "raw_img": lambda pid: next(
            iter(list((MAZE_DATA / "images" / pid).glob("*.jpg"))
                 + list((MAZE_DATA / "images" / pid).glob("*.png"))),
            None),
        "seg_root": MAZE_SEG,
    },
    "nqueens": {
        "prompt_key": "nqueens",
        "raw_img": lambda pid: NQ_DATA / "images" / pid / "nqueens.jpg",
        "seg_root": NQ_SEG,
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
        seg = spec["seg_root"] / args.mode / f"{pid}_overlay.jpg" if do_seg else None
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
            raw = CHESS_DATA / "images" / f"{pid}.png"
            seg = CHESS_SEG / args.mode / f"{pid}_overlay.jpg" if do_seg else None
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
        with open(MAZE_DATA / "maze_solve_v2.csv") as f:
            for row in csv.DictReader(f):
                gts["maze"][row["image_path"].split("/")[1]] = row["text_representation_start-position"].strip()
    if "nqueens" in args.tasks:
        with open(NQ_DATA / "nqueens_v2.csv") as f:
            for row in csv.DictReader(f):
                gts["nqueens"][row["image_path"].split("/")[1]] = row["text-representation_start-position"].strip()
    if "chess_count" in args.tasks or "chess_grid" in args.tasks:
        # The chess seg set is not committed. Without this check a missing tree yields
        # zero chess tasks and the run still reports success, which reads as "the model
        # scored nothing" rather than "the data was never there".
        if not (CHESS_DATA / "text_repr.json").exists():
            raise SystemExit(
                f"the chess seg-eval data is missing: {CHESS_DATA}\n"
                f"It is not committed — supply it under {CHESS_SEG269} (or point\n"
                "EXPERIMENTS_LOCAL_DATA_ROOT elsewhere); see datasets/_local/README.md.\n"
                "The maze and nqueens tasks do not need it.")
        for entry in json.loads((CHESS_DATA / "text_repr.json").read_text()):
            pid = Path(entry["filename"]).stem
            gts["chess"][pid] = {
                "pieces": entry["num_pieces"], "empty": entry["num_empty_cells"],
                "black":  entry["black_repr"], "white": entry["white_repr"],
            }
    return gts


# ---------- per-task scoring dispatch ----------------------------------------

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


# ---------- eval loop --------------------------------------------------------

async def _amain(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gts = load_gts(args)
    tasks = build_tasks(args, gts)
    print(f"Model: {args.model}  Backend: {args.backend}  Mode: {args.mode}  Tasks: {args.tasks}")
    print(f"Total calls: {len(tasks)}")

    sem = asyncio.Semaphore(args.concurrency)
    if args.backend == "openai":
        import openai
        client = openai.AsyncOpenAI()
        coros = [utils.call_openai(client, args.model, _msgs(p, prompt), sem, args.reasoning)
                 for _, _, _, p, prompt in tasks]
        responses = await tqdm_asyncio.gather(*coros)
    else:
        async with aiohttp.ClientSession() as session:
            coros = [utils.call_vllm(session, args.ports[i % len(args.ports)],
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

    tag = utils.model_tag(args.model)
    if args.reasoning:
        tag += f"_{args.reasoning}"
    out_path = RESULTS_DIR / f"{tag}_{args.mode}.json"
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


# ---------- --analyze (cross-run delta table) --------------------------------

def _maze_metric(d: dict, variant: str) -> Optional[float]:
    return d.get("maze", {}).get(variant, {}).get("cell")


def _nq_metric(d: dict, variant: str) -> Optional[float]:
    return d.get("nqueens", {}).get(variant, {}).get("f1")


def _chess_metric(d: dict, variant: str) -> Optional[float]:
    vals = [d.get(k, {}).get(variant, {}).get("f1")
            for k in ("chess_black", "chess_white")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


METRICS = {"chess": _chess_metric, "maze": _maze_metric, "nqueens": _nq_metric}


def _discover_models(model_filter: Optional[list[str]] = None) -> list[str]:
    """Return base model tags that have both _oracle_mask.json and _tddn_mask.json."""
    oracle = {p.name[:-len("_oracle_mask.json")]
              for p in RESULTS_DIR.glob("*_oracle_mask.json")}
    tddn = {p.name[:-len("_tddn_mask.json")]
            for p in RESULTS_DIR.glob("*_tddn_mask.json")}
    paired = sorted(oracle & tddn)
    if model_filter:
        paired = [m for m in paired if any(f in m for f in model_filter)]
    return paired


def _deltas(model: str, task: str) -> Optional[dict]:
    m_path = RESULTS_DIR / f"{model}_oracle_mask.json"
    pm_path = RESULTS_DIR / f"{model}_tddn_mask.json"
    if not (m_path.exists() and pm_path.exists()):
        return None
    M = json.loads(m_path.read_text())
    PM = json.loads(pm_path.read_text())
    fn = METRICS[task]
    raw_M, seg_M = fn(M, "raw"), fn(M, "oracle_mask")
    raw_PM, seg_PM = fn(PM, "raw"), fn(PM, "tddn_mask")
    if None in (raw_M, raw_PM):
        return None
    img = (raw_M + raw_PM) / 2
    return dict(
        raw_M=raw_M, seg_M=seg_M, raw_PM=raw_PM, seg_PM=seg_PM, Img=img,
        dM=(seg_M - img) if seg_M is not None else None,
        dPM=(seg_PM - img) if seg_PM is not None else None,
        dPMM=(seg_PM - seg_M) if (seg_M is not None and seg_PM is not None) else None,
    )


def _fmt(x: Optional[float], width: int = 5, sign: bool = False) -> str:
    if x is None:
        return f"{'N/A':>{width}}"
    return f"{x:+{width}.1f}" if sign else f"{x:{width}.1f}"


def _print_text_table(models: list[str], tasks: list[str]) -> None:
    print("# Img = (raw_M + raw_PM) / 2;  dM = seg_M - Img;  dPM = seg_PM - Img;  dPMM = dPM - dM")
    print()
    print(f"{'Model':<28} {'TASK':<8} | "
          f"{'raw_M':>6} {'seg_M':>6} | "
          f"{'raw_PM':>6} {'seg_PM':>6} | "
          f"{'Img':>6} {'dM':>6} {'dPM':>6} {'dPMM':>6}")
    print("-" * 100)
    for m in models:
        for t in tasks:
            d = _deltas(m, t)
            if d is None:
                continue
            print(f"{m[:28]:<28} {t:<8} | "
                  f"{_fmt(d['raw_M'])} {_fmt(d['seg_M'])} | "
                  f"{_fmt(d['raw_PM'])} {_fmt(d['seg_PM'])} | "
                  f"{_fmt(d['Img'])} {_fmt(d['dM'], sign=True)} "
                  f"{_fmt(d['dPM'], sign=True)} {_fmt(d['dPMM'], sign=True)}")
        print()


def _print_latex_rows(models: list[str], tasks: list[str]) -> None:
    """One row per model: Model & Img,dM,dPM,dPMM per task & meanDPMM."""
    print(r"% Model & " + " & ".join(f"{t}(Img,dM,dPM,dPMM)" for t in tasks) + r" & meanDPMM")
    for m in models:
        cells = [m[:28]]
        pmms: list[float] = []
        for t in tasks:
            d = _deltas(m, t)
            for k in ("Img", "dM", "dPM", "dPMM"):
                cells.append(_fmt(d[k] if d else None, sign=(k != "Img")))
            if d and d["dPMM"] is not None:
                pmms.append(d["dPMM"])
        mean = sum(pmms) / len(pmms) if pmms else None
        cells.append(_fmt(mean, sign=True))
        print(" & ".join(c.strip() for c in cells) + r"  \\")


def analyze(args) -> None:
    models = _discover_models(args.model_filter)
    if not models:
        print(f"No paired (_oracle_mask.json + _tddn_mask.json) result JSONs found in {RESULTS_DIR}")
        return
    print(f"Found {len(models)} model(s) with paired runs:")
    for m in models:
        print(f"  {m}")
    print()
    _print_text_table(models, args.tasks)
    if args.latex:
        print()
        print("=" * 100)
        print()
        _print_latex_rows(models, args.tasks)


# ---------- CLI --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["openai", "vllm"],
                   help="required unless --analyze")
    p.add_argument("--model",
                   help="model id; required unless --analyze")
    p.add_argument("--mode", choices=["raw", "oracle_mask", "tddn_mask"],
                   help="required unless --analyze")
    p.add_argument("--tasks", nargs="+", default=["maze", "nqueens"],
                   choices=["maze", "nqueens", "chess_count", "chess_grid", "chess"],
                   help="(analyze) supports 'chess' as a single label that aggregates "
                        "chess_count + chess_grid")
    p.add_argument("--ports", nargs="+", type=int, default=[8001, 8002, 8003, 8004])
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--reasoning", default=None,
                   choices=["default", "low", "medium", "high"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap total number of calls (debug)")
    p.add_argument("--analyze", action="store_true",
                   help="skip API calls; print Img/dM/dPM/dPMM delta table from existing results/")
    p.add_argument("--model-filter", nargs="*",
                   help="(analyze) substring filter on model tag")
    p.add_argument("--latex", action="store_true",
                   help="(analyze) also print LaTeX rows after the text table")
    args = p.parse_args()

    if args.analyze:
        # Allow shorthand task list "chess" in analyze mode → expand to chess_count+chess_grid is
        # not meaningful for analyze (analyze reads from chess_* keys directly), so just normalise
        # to the three high-level metrics.
        analyze_tasks = []
        for t in args.tasks:
            if t in ("chess_count", "chess_grid", "chess"):
                if "chess" not in analyze_tasks:
                    analyze_tasks.append("chess")
            elif t in ("maze", "nqueens") and t not in analyze_tasks:
                analyze_tasks.append(t)
        args.tasks = analyze_tasks or ["chess", "maze", "nqueens"]
        analyze(args)
        return

    if not args.backend or not args.model or not args.mode:
        p.error("--backend, --model and --mode are required unless --analyze is set")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
