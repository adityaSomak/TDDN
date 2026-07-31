"""Build the named checkpoints the evaluations load: ``tdn`` and ``tddn``.

Both are derived from the raw training steps that stay alongside them:

    vith_roberta_v3_coco_ft/ckpt/99            -> .../ckpt/tdn
    fused_dinov3_cleandift_coco_ft/ckpt/99,149 -> .../ckpt/tddn   (averaged)

Only the trainable parameters (projection heads + logit scale) are stored, which
is exactly what ``load_trainable_state`` reads back; the frozen backbones come
from HuggingFace at load time.

Averaging weights before the forward pass is equivalent to averaging them at load
time, which is what the evaluations did previously — one model, one forward,
either way. This script asserts that equivalence rather than assuming it: every
tensor it writes is reloaded and compared with ``torch.equal``.

Usage
-----
    python build_named_checkpoints.py             # build what's missing, then verify
    python build_named_checkpoints.py --force     # rebuild even if present
    python build_named_checkpoints.py --verify-only

Runs on CPU in a few minutes; no GPU needed. ``HF_TOKEN`` must be set for the
gated DINOv3 / RoBERTa weights, and ``dinov3`` must be importable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))          # experiments/ on the import path
if os.environ.get("DINOV3_ROOT"):                  # unneeded if dinov3 is pip-installed
    sys.path.insert(0, os.environ["DINOV3_ROOT"])

from shared_utils.feature_extraction.loaders import _build_alignment_model  # noqa: E402
from shared_utils.feature_extraction.text_alignment import (  # noqa: E402
    average_states,
    load_trainable_state,
)

# name -> (checkpoint tree, source step directories to combine)
TARGETS = {
    "tdn": ("vith_roberta_v3_coco_ft", ("99",)),
    "tddn": ("fused_dinov3_cleandift_coco_ft", ("99", "149")),
}


def _build_model(tree: Path) -> torch.nn.Module:
    """Instantiate the alignment model to supply state-dict keys and shapes."""
    model, config = _build_alignment_model(tree / "config.yaml", "cpu", torch.float32)
    if config.use_cleandift or getattr(config, "use_fused_encoder", False):
        model.load_cleandift_backbone("cpu")
    return model


def _combine(model: torch.nn.Module, ckpt_root: Path, steps: tuple[str, ...]) -> dict:
    states = [load_trainable_state(model, ckpt_root / s) for s in steps]
    return average_states(states) if len(states) > 1 else states[0]


def _compare(model: torch.nn.Module, expected: dict, path: Path) -> bool:
    """Reload ``path`` and report whether it is bit-identical to ``expected``."""
    reloaded = load_trainable_state(model, path)
    if set(reloaded) != set(expected):
        print(f"    FAIL: key sets differ "
              f"({len(expected)} expected vs {len(reloaded)} on disk)")
        return False
    bad = [k for k in expected if not torch.equal(expected[k], reloaded[k])]
    if bad:
        worst = max((expected[k].double() - reloaded[k].double()).abs().max().item()
                    for k in bad)
        print(f"    FAIL: {len(bad)}/{len(expected)} tensors differ, "
              f"worst abs diff {worst:.3e} (e.g. {bad[0]})")
        return False
    print(f"    OK: {len(expected)}/{len(expected)} tensors bit-identical")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", choices=sorted(TARGETS), action="append",
                    help="build only this checkpoint (repeatable); default is all")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the target directory already exists")
    ap.add_argument("--verify-only", action="store_true",
                    help="compare existing targets against their sources, write nothing")
    args = ap.parse_args()

    ok = True
    for name in args.name or sorted(TARGETS):
        tree_name, steps = TARGETS[name]
        tree = _HERE / tree_name
        target = tree / "ckpt" / name
        print(f"\n{name}: {tree_name}/ckpt/{{{','.join(steps)}}} -> ckpt/{name}")

        if args.verify_only and not target.is_dir():
            print(f"    SKIP: {target} does not exist")
            ok = False
            continue
        if target.is_dir() and not (args.force or args.verify_only):
            print(f"    exists; verifying (pass --force to rebuild)")

        model = _build_model(tree)
        expected = _combine(model, tree / "ckpt", steps)
        print(f"    combined {len(steps)} source ckpt(s): {len(expected)} tensors")

        if not target.is_dir() or args.force:
            if target.is_dir():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            dcp.save({"model": expected}, checkpoint_id=str(target))
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            print(f"    wrote {target} ({size / 2**20:,.0f} MiB)")

        ok &= _compare(model, expected, target)
        del model

    print("\n" + ("all named checkpoints match their sources"
                  if ok else "MISMATCH — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
