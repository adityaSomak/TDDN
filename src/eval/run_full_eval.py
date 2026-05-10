"""Run a Q1..QN VLM eval for one full-eval task.

Per-task config (system prompt, format instructions, JSONL path, answer-type
resolution) lives in src/eval/prompts/<task>.yaml. Question records come from
<task>/<task>_eval.jsonl.

Output: <task>/eval_results_low_detail/<model>/<reasoning>/{results.jsonl, report.txt}
Resume-safe: existing results.jsonl entries are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from src.eval import backends
from src.eval.prompts import full_eval_config, detect_answer_type


ALGO_ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_SIZE = 512   # match open-source models' input resolution


def encode_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.BILINEAR)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Answer checking

def _strip_markdown(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _safe_literal(s: str):
    """ast.literal_eval with truncated-bracket recovery."""
    for suffix in ("", "]", ")]", "}", "]}"):
        try:
            return ast.literal_eval(s + suffix)
        except (ValueError, SyntaxError):
            continue
    return None


def check_answer(answer_type: str, expected: str, model_response: str) -> bool:
    """Return True iff `model_response` matches `expected` for `answer_type`."""
    resp = _strip_markdown(model_response)

    if answer_type == "mcq":
        letter = resp.strip(".)")[:1].upper() if resp else ""
        return letter == expected.upper()

    if answer_type in ("boolean", "color"):
        return resp.lower() == expected.lower()

    if answer_type == "color_list":
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if rv is None or ev is None:
            return resp == expected
        return sorted(rv) == sorted(ev)

    if answer_type in ("list", "position_list"):
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if rv is None or ev is None:
            return resp == expected
        return sorted(map(str, rv)) == sorted(map(str, ev))

    if answer_type in ("integer_list_rows", "integer_list_cols"):
        rv = _safe_literal(resp) or _safe_literal(f"[{resp}]")
        ev = _safe_literal(expected)
        if rv is None or ev is None:
            return False
        return sorted(rv) == sorted(ev)

    if answer_type == "dimension_list":
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if rv is None or ev is None:
            return False
        return sorted(map(str, rv)) == sorted(map(str, ev))

    if answer_type == "coordinate":
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if rv is None:
            m = re.search(r"(\d+)\D+(\d+)", resp)
            if m and ev is not None:
                return (int(m.group(1)), int(m.group(2))) == tuple(ev)
            return False
        return tuple(rv) == tuple(ev) if ev is not None else False

    if answer_type in ("coordinate_list", "coordinate_list_long"):
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if rv is None or ev is None:
            return False
        try:
            return sorted(map(tuple, rv)) == sorted(map(tuple, ev))
        except TypeError:
            return False

    if answer_type == "dimensions":
        m_r = re.search(r"(\d+)\s*[xX×]\s*(\d+)", resp)
        m_e = re.search(r"(\d+)\s*[xX×]\s*(\d+)", expected)
        if m_r and m_e:
            return (m_r.group(1), m_r.group(2)) == (m_e.group(1), m_e.group(2))
        return resp.strip().lower() == expected.strip().lower()

    if answer_type in ("cell_dict_empty_first", "cell_dict_occupied_first"):
        rv, ev = _safe_literal(resp), _safe_literal(expected)
        if not (isinstance(rv, dict) and isinstance(ev, dict)):
            return False
        try:
            return (sorted(map(tuple, rv.get("empty_cells", []))) ==
                    sorted(map(tuple, ev.get("empty_cells", []))) and
                    sorted(map(tuple, rv.get("occupied_cells", []))) ==
                    sorted(map(tuple, ev.get("occupied_cells", []))))
        except TypeError:
            return False

    # number / position / fallback
    try:
        return float(resp) == float(expected)
    except ValueError:
        return resp == expected


# Eval loop

async def run(args) -> None:
    cfg = full_eval_config(args.task)
    eval_jsonl = ALGO_ROOT / cfg["eval_jsonl"]
    out_dir = (ALGO_ROOT / args.task / "eval_results_low_detail" /
               args.model.split("/")[-1] / (args.reasoning or "default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, report_path = out_dir / "results.jsonl", out_dir / "report.txt"

    records = [json.loads(l) for l in open(eval_jsonl)]
    if args.limit:
        records = records[: args.limit]

    done: dict[tuple, dict] = {}
    if results_path.exists():
        for line in open(results_path):
            rec = json.loads(line)
            done[(rec["puzzle_id"], rec["question_id"])] = rec
    todo = [r for r in records if (r["puzzle_id"], r["question_id"]) not in done]
    print(f"[{args.task} | {args.model} | {args.reasoning or 'default'}] "
          f"total={len(records)} done={len(done)} todo={len(todo)}")

    image_cache: dict[str, str] = {}
    for rec in todo:
        p = rec["image_path"]
        if p not in image_cache:
            image_cache[p] = encode_image(ALGO_ROOT / args.task / p)

    sem = asyncio.Semaphore(args.concurrency)
    openai_client = http_session = None
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set")
        import openai
        openai_client = openai.AsyncOpenAI()
    else:
        import aiohttp
        http_session = aiohttp.ClientSession()

    out_file = open(results_path, "a")

    async def call(idx: int, record: dict) -> dict:
        atype = detect_answer_type(args.task, record)
        question = record["question"] + cfg["format_instructions"][atype]
        b64 = image_cache[record["image_path"]]
        msgs = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": question},
            ]},
        ]
        if args.backend == "openai":
            resp = await backends.call_openai(openai_client, args.model, msgs, sem, args.reasoning)
        else:
            port = args.ports[idx % len(args.ports)]
            resp = await backends.call_vllm(http_session, port, args.model, msgs, sem)
        ok = check_answer(atype, record["answer"], resp)
        out = {**record, "answer_type_resolved": atype, "model_response": resp, "correct": ok}
        out_file.write(json.dumps(out) + "\n")
        out_file.flush()
        return out

    try:
        new_results = await tqdm_asyncio.gather(
            *[call(i, r) for i, r in enumerate(todo)],
            desc=f"{args.task} {args.model}",
        )
    finally:
        out_file.close()
        if http_session is not None:
            await http_session.close()

    for r in new_results:
        done[(r["puzzle_id"], r["question_id"])] = r
    all_results = [done[(r["puzzle_id"], r["question_id"])]
                   for r in records if (r["puzzle_id"], r["question_id"]) in done]
    _generate_report(all_results, args, report_path)


def _generate_report(results: list[dict], args, report_path: Path) -> None:
    lines: list[str] = []
    p = lambda s="": (lines.append(s), print(s))
    n, c = len(results), sum(1 for r in results if r["correct"])
    p("=" * 70)
    p(f"EVAL REPORT -- {args.task} | {args.model} (reasoning={args.reasoning or 'default'})")
    p(f"Total: {n}  Correct: {c}  Accuracy: {c/n*100:.1f}%")
    p("=" * 70)

    qstats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    tstats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        qpre = r["question_id"].split("_")[0]
        qstats[qpre]["total"] += 1; qstats[qpre]["correct"] += int(r["correct"])
        atype = r.get("answer_type_resolved") or r.get("answer_type", "unknown")
        tstats[atype]["total"] += 1; tstats[atype]["correct"] += int(r["correct"])

    p("\n--- Per Question Type ---")
    p(f"{'Q':<6} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    for q in sorted(qstats, key=lambda x: int(re.sub(r"\D", "", x) or 0)):
        s = qstats[q]
        p(f"{q:<6} {s['correct']:>8} {s['total']:>6} {s['correct']/s['total']*100:>9.1f}%")

    p("\n--- Per Answer Type ---")
    for at in sorted(tstats):
        s = tstats[at]
        p(f"{at:<28} {s['correct']:>6}/{s['total']:<6} ({s['correct']/s['total']*100:.1f}%)")

    p("\n--- Failure Sample ---")
    failures = [r for r in results if not r["correct"]]
    for r in failures[:10]:
        p(f"  pid={r['puzzle_id']} q={r['question_id']} "
          f"expected={r['answer']!r} got={r['model_response'][:60]!r}")
    if len(failures) > 10:
        p(f"  ... and {len(failures) - 10} more")
    p("=" * 70)
    report_path.write_text("\n".join(lines))
    print(f"Report -> {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True,
                    choices=["checker_move", "maze_solve", "nqueens", "wood_slide"])
    ap.add_argument("--backend", choices=["openai", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default=None,
                    choices=["default", "low", "medium", "high"])
    ap.add_argument("--ports", nargs="+", type=int, default=[8001, 8002, 8003, 8004])
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
