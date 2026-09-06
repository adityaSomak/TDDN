"""Convert legacy raw DCP steps into flat Hugging Face release checkpoints.

Examples
--------
    python build_named_checkpoints.py --source-root /path/to/checkpoints
    python build_named_checkpoints.py --source-root /path/to/checkpoints --name tddn

The source root must contain the original training trees and ``ckpt/<step>``
directories. Outputs are deterministic CPU ``model.safetensors`` files under
this directory's ``TDN/`` and ``TDDN/`` release folders.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
if os.environ.get("DINOV3_ROOT"):
    sys.path.insert(0, os.environ["DINOV3_ROOT"])

from shared_utils.feature_extraction.loaders import _build_alignment_model  # noqa: E402
from shared_utils.feature_extraction.text_alignment import (  # noqa: E402
    average_states,
    load_trainable_state,
)

TARGETS = {
    "tdn": ("TDN", "vith_roberta_v3_coco_ft", ("99",)),
    "tddn": ("TDDN", "fused_dinov3_cleandift_coco_ft", ("99", "149")),
}


def _build_model(tree: Path) -> torch.nn.Module:
    model, config = _build_alignment_model(tree / "config.yaml", "cpu", torch.float32)
    if config.use_cleandift or getattr(config, "use_fused_encoder", False):
        model.load_cleandift_backbone("cpu")
    return model


def _expected_state(model: torch.nn.Module, tree: Path, steps: tuple[str, ...]) -> dict:
    states = [load_trainable_state(model, tree / "ckpt" / step) for step in steps]
    state = average_states(states) if len(states) > 1 else states[0]
    return {key: state[key].detach().cpu().contiguous() for key in sorted(state)}


def _verify(expected: dict, output: Path) -> bool:
    actual = load_file(str(output), device="cpu")
    if set(actual) != set(expected):
        print(f"    FAIL: key sets differ ({len(expected)} expected, {len(actual)} found)")
        return False
    bad = [key for key in expected if not torch.equal(expected[key], actual[key])]
    if bad:
        worst = max(
            (expected[key].double() - actual[key].double()).abs().max().item()
            for key in bad
        )
        print(f"    FAIL: {len(bad)} tensors differ; worst absolute difference {worst:.3e}")
        return False
    print(f"    OK: {len(expected)} tensors are bit-identical")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="directory containing the legacy checkpoint trees")
    parser.add_argument("--name", choices=sorted(TARGETS), action="append",
                        help="convert only this model; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing model.safetensors")
    parser.add_argument("--verify-only", action="store_true",
                        help="compare existing Safetensors outputs without writing")
    args = parser.parse_args()

    ok = True
    source_root = args.source_root.expanduser().resolve()
    for name in args.name or sorted(TARGETS):
        release_name, legacy_name, steps = TARGETS[name]
        source = source_root / legacy_name
        target = _HERE / release_name
        output = target / "model.safetensors"
        print(f"\n{name}: {legacy_name}/ckpt/{{{','.join(steps)}}} -> {release_name}/model.safetensors")

        required = [source / "config.yaml", *(source / "ckpt" / s / ".metadata" for s in steps)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            print(f"    FAIL: missing source files: {missing}")
            ok = False
            continue
        if not (target / "config.json").is_file():
            print(f"    FAIL: missing release config: {target / 'config.json'}")
            ok = False
            continue

        model = _build_model(source)
        expected = _expected_state(model, source, steps)

        if args.verify_only and not output.is_file():
            print(f"    FAIL: missing output: {output}")
            ok = False
        else:
            if not args.verify_only:
                target.mkdir(parents=True, exist_ok=True)
                if not (target / "training_config.yaml").exists():
                    shutil.copy2(source / "config.yaml", target / "training_config.yaml")
            if not args.verify_only and (args.force or not output.is_file()):
                save_file(expected, str(output), metadata={
                    "format": "pt",
                    "variant": name,
                    "source_steps": ",".join(steps),
                })
                print(f"    wrote {output} ({output.stat().st_size / 2**20:,.0f} MiB)")
            ok &= _verify(expected, output)
        del model

    print("\n" + ("all release checkpoints verified" if ok else "verification failed"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
