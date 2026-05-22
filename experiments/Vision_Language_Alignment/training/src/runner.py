"""Top-level training driver for the alignment model.

Two-round protocol controlled by the round YAML:

  - Round 1: LAION + COCO contrastive pretrain
            (5000 iters, lr=1e-3, grad-cache ×16).
  - Round 2: COCO-only fine-tune
            (500 iters TDN / 200 iters TDDN, lr=1e-4, grad-cache ×32,
            resumes from a round-1 final via ``--resume-checkpoint``).

The runner translates the nested merged config dict (see
``run_train.py``) into a flat ``AlignConfig``, constructs the
``AlignmentModel``, wraps it with FSDP + AC + ``torch.compile``,
builds the optimizer + cosine LR schedule + COCO/LAION dataset (with
an infinite rank-sharded sampler), and runs the training loop.

Checkpoints are written under
``<out_root>/checkpoints/<variant>-round<n>/ckpt/<step>/`` as DCP
shards in the format ``shared_utils...load_trainable_state`` consumes.

A process group is required (the FSDP wrap, the all-gathered InfoNCE
loss, and the DCP checkpoint format all assume ``torch.distributed``
is initialized). Launch via ``torchrun --nproc_per_node=N``; a plain
``python`` invocation raises immediately.

Public API
----------
    run_training(merged_cfg, run_dir, log_dir, *, max_iterations=None,
                 resume_checkpoint=None)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from shared_utils.feature_extraction.text_alignment import AlignConfig, AlignmentModel
from shared_utils.paths import REPO_ROOT

from .checkpoint import (
    find_latest_checkpoint,
    keep_last_n_checkpoints,
    load_dcp,
    register_frozen_dont_save_hooks,
    save_dcp,
)
from .data import InfiniteShardedSampler, build_train_dataset
from .losses import StructureLoss
from .parallelize import setup_fsdp
from .train_step import (
    is_no_decay,
    linear_warmup_cosine_decay,
    train_step,
)


logger = logging.getLogger("vla.training")

# ---------------------------------------------------------------------------
# Backbone-tag -> HuggingFace model id resolution (mirrors the inference
# registry; kept here so the training driver has no hidden dependency on
# the registry module).
# ---------------------------------------------------------------------------

_VISION_BACKBONE_HF_ID = {
    "vith-roberta":    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "fused-dinov3-cd": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

_TEXT_ENCODER_NAME = {
    "roberta-large":   "sentence-transformers/all-roberta-large-v1",
}


# ---------------------------------------------------------------------------
# Config flattening
# ---------------------------------------------------------------------------

def _resolve_data_path(value: Optional[str]) -> str:
    """Resolve a YAML data path against REPO_ROOT; pass through empty strings."""
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def build_align_config(merged: dict) -> AlignConfig:
    """Translate the nested merged YAML into a flat ``AlignConfig``.

    The merged dict is the one constructed by ``run_train.py`` from
    ``configs/training/round_<n>.yaml`` + ``configs/models.yaml:<tag>.arch``
    + ``configs/models.yaml:<tag>.round_<n>``.
    """
    optim_cfg = merged.get("optim", {}) or {}
    loss_cfg = merged.get("loss", {}) or {}
    trainer_cfg = merged.get("trainer", {}) or {}
    data_cfg = merged.get("data", {}) or {}
    arch_cfg = merged.get("arch", {}) or {}

    backbone_tag = merged.get("backbone")
    text_tag = merged.get("text_encoder")
    if backbone_tag not in _VISION_BACKBONE_HF_ID:
        raise ValueError(f"Unknown backbone {backbone_tag!r}; expected one of "
                         f"{list(_VISION_BACKBONE_HF_ID)}")
    if text_tag not in _TEXT_ENCODER_NAME:
        raise ValueError(f"Unknown text encoder {text_tag!r}; expected one of "
                         f"{list(_TEXT_ENCODER_NAME)}")

    use_fused = bool(arch_cfg.get("use_fused_encoder", False))
    precision = trainer_cfg.get("precision", "bf16-mixed")
    param_dtype = "bf16" if "bf16" in precision else "fp32"
    reduce_dtype = "fp32" if "mixed" in precision else param_dtype

    return AlignConfig(
        # --- vision ---
        vision_backbone_hf_model_id=_VISION_BACKBONE_HF_ID[backbone_tag],
        vision_model_train_img_size=merged.get("transform", {}).get("input_size", 336),
        embed_dim=arch_cfg.get("embed_dim", 2560),
        vision_num_head_blocks=arch_cfg.get("vision_num_head_blocks", 2),
        head_blocks_drop_path=arch_cfg.get("head_blocks_drop_path", 0.0),
        use_rope_in_head=arch_cfg.get("use_rope_in_head", True),
        use_linear_projection=arch_cfg.get("use_linear_projection", False),
        # --- text ---
        text_encoder_name=_TEXT_ENCODER_NAME[text_tag],
        text_layer_idx=arch_cfg.get("text_layer_idx", 24),
        text_num_head_blocks=arch_cfg.get("text_num_head_blocks", 2),
        text_head_blocks_drop_path=arch_cfg.get("text_head_blocks_drop_path", 0.1),
        # --- fused / cleandift ---
        use_fused_encoder=use_fused,
        cleandift_proj_dim=arch_cfg.get("cleandift_proj_dim", 512),
        cleandift_common_grid=arch_cfg.get("cleandift_common_grid", 21),
        # --- optim ---
        lr=float(optim_cfg.get("lr", 1.0e-3)),
        min_lr=float(optim_cfg.get("min_lr", 0.0)),
        weight_decay=float(optim_cfg.get("weight_decay", 1.0e-4)),
        batch_size=int(optim_cfg.get("batch_size", 64)),
        warmup_iterations=int(optim_cfg.get("warmup_iterations", 500)),
        max_iterations=int(optim_cfg.get("max_iterations", 5000)),
        grad_cache_multiplier=int(optim_cfg.get("grad_cache_multiplier", 16)),
        # --- loss ---
        clip_temperature=float(loss_cfg.get("clip_temperature", 0.05)),
        structure_lambda=float(loss_cfg.get("structure_lambda", 10.0)),
        structure_temperature=float(loss_cfg.get("structure_temperature", 0.05)),
        structure_levels=int(loss_cfg.get("structure_levels", 1)),
        structure_warmup_steps=int(loss_cfg.get("structure_warmup_steps", 500)),
        structure_centering=loss_cfg.get("structure_centering", "mean"),
        structure_distance=loss_cfg.get("structure_distance", "cosine"),
        structure_weighting=loss_cfg.get("structure_weighting", "none"),
        structure_margin=float(loss_cfg.get("structure_margin", 0.0)),
        structure_center_first=bool(loss_cfg.get("structure_center_first", False)),
        # --- parallelization ---
        use_fsdp=bool(trainer_cfg.get("use_fsdp", True)),
        use_ac=bool(trainer_cfg.get("use_ac", True)),
        do_compile=bool(trainer_cfg.get("do_compile", True)),
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        # --- data ---
        coco_root=_resolve_data_path(data_cfg.get("coco_root", "")),
        coco_ann_file=_resolve_data_path(data_cfg.get("coco_ann_file", "")),
        laion_shards=_resolve_data_path(data_cfg.get("laion_shards", "")),
        # --- checkpointing ---
        resume_checkpoint=merged.get("resume_checkpoint"),
    )


# ---------------------------------------------------------------------------
# Image / text preprocessing
# ---------------------------------------------------------------------------

def _train_transform(img_size: int):
    """Stochastic crop + horizontal flip + ImageNet normalization."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _make_collate(tokenizer):
    """Build a collate that stacks images and tokenizes captions for RoBERTa."""
    def collate(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        text_inputs = tokenizer(
            list(captions), padding="longest", truncation=True,
            return_tensors="pt",
        )
        return images, text_inputs
    return collate


# ---------------------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------------------

def run_training(
    merged_cfg: dict,
    run_dir: Path,
    log_dir: Path,
    *,
    max_iterations: Optional[int] = None,
    resume_checkpoint: Optional[Path] = None,
) -> None:
    """Train the alignment model with the merged config under ``run_dir``."""
    cfg = build_align_config(merged_cfg)
    if max_iterations is not None:
        cfg.max_iterations = max_iterations
    if resume_checkpoint is not None:
        cfg.resume_checkpoint = str(resume_checkpoint)

    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    loss_log = log_dir / "loss_log.jsonl"

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}"
                          if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Detect whether a resume / round-1 ckpt will overlay the heads;
    # if so we skip the random init inside ``setup_fsdp`` to avoid
    # wasted work.
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    will_load_ckpt = latest_ckpt is not None or bool(cfg.resume_checkpoint)

    # --- model ---
    if not distributed:
        raise RuntimeError(
            "run_training expects torch.distributed to be initialized; "
            "launch via `torchrun --nproc_per_node=N run_train.py ...`."
        )
    model = AlignmentModel(cfg)
    tokenizer = model.text_encoder.tokenizer
    setup_fsdp(model, cfg, should_init=not will_load_ckpt)
    if cfg.use_fused_encoder or getattr(cfg, "use_cleandift", False):
        model.load_cleandift_backbone(device)
    register_frozen_dont_save_hooks(model)

    # --- optimizer (decay split: no-decay on biases / 1-D params / logit_scale) ---
    named = list(model.named_parameters())
    no_decay = [p for n, p in named if is_no_decay(n, p) and p.requires_grad]
    decay = [p for n, p in named if not is_no_decay(n, p) and p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0},
         {"params": decay,    "weight_decay": cfg.weight_decay}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )
    if is_main:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {n_trainable:,} / {n_total:,}")

    lr_schedule = linear_warmup_cosine_decay(
        cfg.lr, cfg.warmup_iterations, cfg.max_iterations, cfg.min_lr,
    )

    # --- losses ---
    structure_loss = None
    if cfg.structure_lambda > 0:
        structure_loss = StructureLoss(
            base_lambda=cfg.structure_lambda,
            warmup_steps=max(cfg.structure_warmup_steps, 1),
            temperature=cfg.structure_temperature,
            levels=cfg.structure_levels,
            weighting=cfg.structure_weighting,
            margin=cfg.structure_margin,
            centering=cfg.structure_centering,
            distance=cfg.structure_distance,
            center_first=cfg.structure_center_first,
        ).to(device)
        # Resume puts us at ``start_iter`` rather than 0, so advance
        # the warmup counter accordingly.
        structure_loss.train_step = 0  # set after resume below

    # --- data ---
    transform = _train_transform(cfg.vision_model_train_img_size)
    dataset = build_train_dataset(
        coco_root=Path(cfg.coco_root) if cfg.coco_root else None,
        coco_ann_file=Path(cfg.coco_ann_file) if cfg.coco_ann_file else None,
        laion_shards=Path(cfg.laion_shards) if cfg.laion_shards else None,
        transform=transform,
    )
    sampler = InfiniteShardedSampler(dataset, seed=cfg.seed)
    n_workers = cfg.num_workers
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
        collate_fn=_make_collate(tokenizer),
    )

    # --- resume / load ---
    # (``setup_fsdp`` above already handled the fresh-init branch when
    # no checkpoint was detected, so the only paths left here are the
    # two load variants.)
    start_iter = 0
    if latest_ckpt is not None:
        start_iter = load_dcp(latest_ckpt, model, optimizer=optimizer) + 1
        if structure_loss is not None:
            structure_loss.train_step = start_iter
        logger.info(f"Resumed from {latest_ckpt} -> starting iter {start_iter}")
    elif cfg.resume_checkpoint:
        # Round-2 fine-tune: overlay the heads with the round-1 final
        # weights but keep a fresh LR schedule and optimizer state.
        load_dcp(Path(cfg.resume_checkpoint), model, optimizer=None, strict=False)
        logger.info(f"Initialized weights from {cfg.resume_checkpoint}")

    # --- training loop ---
    data_iter = iter(loader)
    t0 = time.time()
    for cur_iter in range(start_iter, cfg.max_iterations):
        metrics = train_step(
            model=model, data_iter=data_iter, loader=loader,
            optimizer=optimizer, structure_loss=structure_loss,
            n_accum=cfg.grad_cache_multiplier,
            cur_iter=cur_iter, lr=lr_schedule[cur_iter],
            temperature=cfg.clip_temperature,
            label_smoothing=cfg.label_smoothing,
            gradient_clip=cfg.gradient_clip, device=device,
        )

        log_every = max(1, min(10, cfg.max_iterations // 4))
        if is_main and (cur_iter + 1) % log_every == 0:
            elapsed = time.time() - t0
            samples = cfg.batch_size * cfg.grad_cache_multiplier * world_size
            sps = samples * (cur_iter - start_iter + 1) / max(elapsed, 1e-6)
            with open(loss_log, "a") as f:
                f.write(json.dumps({"iter": cur_iter + 1,
                                    "samples_per_sec": sps,
                                    **metrics}) + "\n")
            print(f"iter {cur_iter + 1:5d}/{cfg.max_iterations}  "
                  f"loss={metrics['total']:.4f}  "
                  f"clip={metrics['contrastive']:.4f}  "
                  f"struct={metrics['structure']:.4f}  "
                  f"lr={metrics['lr']:.2e}", flush=True)

        is_last = (cur_iter + 1) == cfg.max_iterations
        save_every = max(50, cfg.max_iterations // 5)
        if is_last or (cur_iter + 1) % save_every == 0:
            save_dcp(ckpt_dir / str(cur_iter + 1), model,
                     iteration=cur_iter + 1, optimizer=optimizer)
            if is_main:
                keep_last_n_checkpoints(ckpt_dir, n=3)

    # Always persist the merged config alongside the final ckpt for audit.
    (run_dir / "config.yaml").write_text(yaml.safe_dump(merged_cfg, sort_keys=False))
    if is_main:
        logger.info("Training complete.")
