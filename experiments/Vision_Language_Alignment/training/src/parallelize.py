"""FSDP2 + selective activation checkpointing + torch.compile setup.

The trained alignment model has two components:

  - **Frozen backbones** (DINOv3 vision, optional CleanDIFT SD pipeline,
    RoBERTa text). These never receive gradients but are still FSDP-
    wrapped so each rank only holds a shard — important on multi-GPU.
  - **Trainable heads** (per-encoder attention blocks, projections, the
    optional CleanDIFT MLPs, the fusion MLP, ``logit_scale``). These
    receive gradients and are wrapped in the same FSDP mesh.

Wrap order is *AC → torch.compile → FSDP* (selective AC wraps each
trainable head block, ``torch.compile`` then compiles those blocks +
the text projection, FSDP shards the resulting parameters across
ranks). The function also asserts the frozen-backbone invariants
(``.eval() + requires_grad = False``) after wrapping and optionally
re-initializes the trainable heads.

When ``torch.distributed`` is not initialized (e.g. ad-hoc inspection
from a Python REPL), the FSDP wrap is skipped but AC / compile /
freezing / init still run, so the model is in a consistent state.

Public API
----------
    setup_fsdp(model, config, *, should_init=True) -> None
        ``should_init`` controls whether ``model.init_weights()`` is
        called at the end. Pass ``False`` when a checkpoint is about
        to overlay the trainable heads.
"""
from __future__ import annotations

import logging
from contextlib import suppress
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger("vla.training")


_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# Operators whose forward activations we want to recompute during
# backward when activation checkpointing is enabled. Everything else
# is kept (matmul + scaled dot-product attention dominate compute).
_AC_SAVE_LIST = [
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
]


def _make_ac_wrapper():
    """Return a ``checkpoint_wrapper`` partial pre-configured with the save list."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    save_list = list(_AC_SAVE_LIST)
    with suppress(AttributeError):
        save_list.append(torch.ops.xformers_flash3.flash_fwd.default)
    return partial(
        checkpoint_wrapper,
        context_fn=partial(create_selective_checkpoint_contexts, save_list),
        preserve_rng_state=True,
    )


def _ac_wrap_blocks(module_list: nn.ModuleList, label: str) -> None:
    """Apply selective AC in-place to every non-Identity block in ``module_list``."""
    wrapper = _make_ac_wrapper()
    n = 0
    for i, block in enumerate(module_list):
        if not isinstance(block, nn.Identity):
            module_list[i] = wrapper(block)
            n += 1
    logger.info(f"Applied selective AC to {n} {label} head blocks")


def _fsdp_wrap_blocks(module_list: nn.ModuleList, fsdp_kwargs: dict) -> None:
    """Shard each block in a ``nn.ModuleList`` under the given FSDP mesh."""
    from torch.distributed._composable.fsdp import fully_shard
    for block in module_list:
        fully_shard(block, **fsdp_kwargs, reshard_after_forward=True)


def _freeze_module(module: nn.Module) -> None:
    """Put ``module`` in eval mode and disable grad on its parameters."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _compile_blocks(module_list: nn.ModuleList, label: str) -> None:
    """Apply ``torch.compile`` to each block in a ``nn.ModuleList`` in-place."""
    for i, block in enumerate(module_list):
        try:
            module_list[i] = torch.compile(block)
        except Exception as e:  # noqa: BLE001 - compilation is best-effort
            logger.warning(f"torch.compile failed on {label} block {i}: {e}")
            return
    logger.info(f"Compiled {len(module_list)} {label} head blocks")


def setup_fsdp(model: nn.Module, config, *, should_init: bool = True) -> None:
    """Move the model to CUDA, apply selective AC + ``torch.compile`` +
    FSDP2 sharding, then assert backbones are frozen + in eval mode.

    Optionally reinitializes the trainable heads (``should_init=True``);
    callers about to overlay a checkpoint should pass
    ``should_init=False`` to avoid the wasted work.

    When ``torch.distributed`` is not initialized this function only
    moves the model to CUDA, asserts the eval/freeze invariants, and
    (optionally) reinitializes the heads — sufficient for single-GPU
    sanity runs.

    Args:
        model: an ``AlignmentModel`` instance built on CPU with
            frozen backbones already loaded (CleanDIFT is loaded
            separately after this call).
        config: ``AlignConfig`` providing ``use_fsdp``, ``use_ac``,
            ``do_compile``, ``param_dtype``, ``reduce_dtype``,
            ``text_num_head_blocks``, ``use_fused_encoder``.
        should_init: when True, calls ``model.init_weights()`` at the
            end to randomly initialize the trainable heads.
    """
    model.cuda()
    logger.info("Moved model to CUDA")

    image_encoder = model.image_encoder
    text_encoder = model.text_encoder
    use_fused = getattr(config, "use_fused_encoder", False)

    distributed = dist.is_available() and dist.is_initialized()
    if not config.use_fsdp or not distributed:
        # Non-distributed path (e.g. interactive inspection): still
        # apply ``torch.compile`` to the trainable heads and assert
        # the frozen-backbone invariants, so the model is in the same
        # post-setup state as the distributed path.
        if getattr(config, "do_compile", False):
            _compile_blocks(image_encoder.blocks, "image")
            if config.text_num_head_blocks > 0:
                _compile_blocks(text_encoder.blocks, "text")
        _freeze_module(image_encoder.backbone)
        _freeze_module(text_encoder.backbone)
        if should_init:
            model.init_weights()
            logger.info("Initialized trainable heads from scratch.")
        return

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    world_mesh = init_device_mesh("cuda", mesh_shape=(world_size,),
                                  mesh_dim_names=("dp",))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP[config.param_dtype],
        reduce_dtype=_DTYPE_MAP[config.reduce_dtype],
    )
    fsdp_kwargs = {"mesh": world_mesh["dp"], "mp_policy": mp_policy}

    # ----- Activation checkpointing on trainable head blocks -----
    if config.use_ac:
        _ac_wrap_blocks(image_encoder.blocks, "image")
        if config.text_num_head_blocks > 0:
            _ac_wrap_blocks(text_encoder.blocks, "text")

    # ----- torch.compile on trainable head blocks + text projection -----
    if getattr(config, "do_compile", False):
        _compile_blocks(image_encoder.blocks, "image")
        if config.text_num_head_blocks > 0:
            _compile_blocks(text_encoder.blocks, "text")
        try:
            text_encoder.projection = torch.compile(text_encoder.projection)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"torch.compile failed on text projection: {e}")

    # ----- FSDP wrapping -----
    # Image side: shard the DINOv3 backbone, the head blocks, and any
    # trainable projection layers. For the fused encoder we also
    # shard the per-layer CleanDIFT MLPs and the fusion MLP.
    if use_fused:
        dino_backbone = image_encoder.backbone
        _fsdp_wrap_blocks(dino_backbone.layer, fsdp_kwargs)
        fully_shard(dino_backbone, **fsdp_kwargs, reshard_after_forward=True)
        for mlp in (image_encoder.mlp_2, image_encoder.mlp_5, image_encoder.mlp_8):
            fully_shard(mlp, **fsdp_kwargs, reshard_after_forward=True)
        fully_shard(image_encoder.fusion_mlp, **fsdp_kwargs, reshard_after_forward=True)
    else:
        vision_backbone = image_encoder.backbone
        _fsdp_wrap_blocks(vision_backbone.layer, fsdp_kwargs)
        fully_shard(vision_backbone, **fsdp_kwargs, reshard_after_forward=True)

    _fsdp_wrap_blocks(image_encoder.blocks, fsdp_kwargs)
    fully_shard(image_encoder, **fsdp_kwargs, reshard_after_forward=True)

    # Text side: shard the frozen RoBERTa transformer layers and the
    # trainable head + projection.
    rb = text_encoder.backbone
    if hasattr(rb, "encoder") and hasattr(rb.encoder, "layer"):
        for layer in rb.encoder.layer:
            fully_shard(layer, **fsdp_kwargs, reshard_after_forward=True)
    fully_shard(rb, **fsdp_kwargs, reshard_after_forward=True)

    if config.text_num_head_blocks > 0:
        _fsdp_wrap_blocks(text_encoder.blocks, fsdp_kwargs)
    fully_shard(text_encoder.projection, **fsdp_kwargs, reshard_after_forward=True)
    fully_shard(text_encoder, **fsdp_kwargs, reshard_after_forward=True)

    # ----- Re-assert frozen-backbone invariants after FSDP wrap -----
    _freeze_module(image_encoder.backbone)
    _freeze_module(text_encoder.backbone)

    if should_init:
        model.init_weights()
        logger.info("Initialized trainable heads from scratch.")

    logger.info(f"FSDP enabled (world_size={world_size}, "
                f"param_dtype={config.param_dtype}, reduce_dtype={config.reduce_dtype})")
