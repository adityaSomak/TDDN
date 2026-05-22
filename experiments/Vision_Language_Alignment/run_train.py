"""Vision_Language_Alignment training entry point.

Two-round protocol:

  - ``--round 1``  LAION + COCO contrastive pretrain
                   (5000 iters, lr=1e-3, gradient-cache ×16).
  - ``--round 2``  COCO-only fine-tune
                   (200-500 iters, lr=1e-4, gradient-cache ×32,
                   resumes from the round-1 final).

Three config layers are merged at startup:

  1. ``configs/training/round_<n>.yaml`` (per-round defaults).
  2. ``configs/models.yaml:<tag>.arch`` (architecture knobs).
  3. ``configs/models.yaml:<tag>.round_<n>`` (variant-specific
     iteration overrides).

The merged config is written to
``<out_root>/checkpoints/<variant>-round<n>/config.yaml`` for
reproducibility, then handed off to ``training.src.runner.run_training``.

Launch under ``torchrun`` (single- or multi-GPU)::

    torchrun --nproc_per_node=N python run_train.py --variant tdn --round 1
    torchrun --nproc_per_node=N python run_train.py --variant tddn --round 2 \\
        --resume-checkpoint <path/to/round1/ckpt/4999>

``torchrun --nproc_per_node=1`` is fine for sanity checks; the script
relies on a process group for FSDP and DCP checkpointing, so a plain
``python run_train.py ...`` invocation will exit immediately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[1]
_HERE_PARENT = _THIS.parent
for _p in (_EXPERIMENTS, _HERE_PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_HERE = _THIS.parent
_CONFIG_MODELS = _HERE / "configs" / "models.yaml"
_CONFIG_TRAIN_DIR = _HERE / "configs" / "training"
_DEFAULT_OUT_ROOT = _HERE / "training"


def _merged_config(variant: str, round_n: int) -> dict:
    """Merge round YAML + per-variant arch + per-variant round overrides."""
    models = yaml.safe_load(_CONFIG_MODELS.read_text())
    train = yaml.safe_load((_CONFIG_TRAIN_DIR / f"round_{round_n}.yaml").read_text())

    index: dict[str, dict] = {}
    for group in ("baselines", "trained"):
        index.update(models.get(group, {}) or {})
    if variant not in index:
        raise SystemExit(f"Unknown variant {variant!r}; choices: {sorted(index)}")
    spec = index[variant]

    merged = dict(train)
    merged["model_tag"] = variant
    merged["backbone"] = spec.get("backbone")
    merged["text_encoder"] = spec.get("text_encoder")
    merged["transform"] = spec.get("transform", {})
    merged["arch"] = spec.get("arch", {})
    if round_n == 2:
        # Round-2 per-variant overrides take precedence on iteration counts.
        merged["optim"].update(spec.get("round_2", {}) or {})
    return merged


def main() -> None:
    """CLI entry: parse arguments, merge configs, dump to ckpt dir, log status."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", required=True, choices=("tdn", "tddn"))
    p.add_argument("--round", type=int, required=True, choices=(1, 2))
    p.add_argument("--out-root", type=Path, default=_DEFAULT_OUT_ROOT,
                   help="Root for checkpoints/<variant>-round<n>/ and logs/<variant>-round<n>/. "
                        "Defaults to experiments/Vision_Language_Alignment/training/.")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Override the iteration count (e.g. for quick dry runs).")
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Round-2 fine-tune resume path (round-1 ckpt dir).")
    args = p.parse_args()

    cfg = _merged_config(args.variant, args.round)
    if args.max_iterations is not None:
        cfg["optim"]["max_iterations"] = args.max_iterations
    if args.resume_checkpoint is not None:
        cfg["resume_checkpoint"] = str(args.resume_checkpoint)

    run_dir = args.out_root / "checkpoints" / f"{args.variant}-round{args.round}"
    log_dir = args.out_root / "logs" / f"{args.variant}-round{args.round}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"[merged config] {run_dir / 'config.yaml'}")
    print(yaml.safe_dump({
        "variant": args.variant, "round": args.round,
        "max_iterations": cfg["optim"]["max_iterations"],
        "lr": cfg["optim"]["lr"],
        "grad_cache_multiplier": cfg["optim"]["grad_cache_multiplier"],
        "resume_checkpoint": cfg.get("resume_checkpoint"),
    }, default_flow_style=False))

    # Initialize the process group from the torchrun-set env vars.
    import os
    import torch
    import torch.distributed as dist
    if "RANK" not in os.environ:
        raise SystemExit(
            "run_train.py must be launched under torchrun "
            "(e.g. `torchrun --nproc_per_node=1 run_train.py ...`)."
        )
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from training.src.runner import run_training
    try:
        run_training(
            cfg, run_dir=run_dir, log_dir=log_dir,
            max_iterations=args.max_iterations,
            resume_checkpoint=args.resume_checkpoint,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
