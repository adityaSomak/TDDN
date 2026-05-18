"""
CleanDIFT image encoder with trainable alignment heads.

Architecture:
  - Frozen SD 2.1 VAE + CleanDIFT UNet backbone (t=0, empty text embedding)
  - Persistent forward hooks capture layers 2, 5, 8 (up_blocks[l//3].resnets[l%3])
  - Per-layer 2-layer MLP projection: C_l → 512
  - Bilinear interpolation to common 21×21 grid (native for layer 8 @ 336px input)
  - Concatenate along feature dim: (B, 441, 1536)
  - 2× SelfAttentionBlock (dinov3.layers) + LayerNorm
  - Mean pool → (B, 1536) aligned embedding

Structure reference (direct, no projection):
  - Mean-pool raw backbone tokens per layer → concatenate (B, 3200)
  - Structure loss only compares (B,B) similarity matrices, so dim mismatch is fine

Layer dims @ 336px input (latent 42×42):
  Layer 2: up_blocks[0].resnets[2] → (B, 25, 1280)   5×5
  Layer 5: up_blocks[1].resnets[2] → (B, 100, 1280)  10×10
  Layer 8: up_blocks[2].resnets[2] → (B, 441, 640)   21×21  ← native/common grid
"""

import logging
from functools import partial
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("text_alignment")

# SD 2.1 + CleanDIFT identifiers
SD_MODEL_ID    = "Charles-Elena/stable-diffusion-2-1"
CLEANDIFT_REPO = "CompVis/cleandift"
CLEANDIFT_FILE = "cleandift_sd21_unet.safetensors"

# Layers to hook and their native channel counts
HOOK_LAYERS    = [2, 5, 8]
LAYER_CHANNELS = {2: 1280, 5: 1280, 8: 640}
COMMON_GRID    = 21   # 21×21 = 441 tokens; native for layer 8 @ 336px


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class CleanDIFTEncoderOutput(NamedTuple):
    aligned:          torch.Tensor   # (B, embed_dim)  after head + projection
    original:         torch.Tensor   # (B, 3200)  frozen raw backbone reference (detached)
    patch_tokens:     torch.Tensor   # (B, 441, 1536)  post-attention tokens
    backbone_patches: torch.Tensor   # (B, 441, 1536)  pre-attention projected (detached)


# ---------------------------------------------------------------------------
# Per-layer MLP projection
# ---------------------------------------------------------------------------

class LayerMLP(nn.Module):
    """2-layer MLP: Linear(C_in, proj_dim) → LayerNorm → GELU → Linear(proj_dim, proj_dim)."""

    def __init__(self, in_dim: int, proj_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CleanDIFT Image Encoder
# ---------------------------------------------------------------------------

class CleanDIFTImageEncoder(nn.Module):
    def __init__(
        self,
        proj_dim: int = 512,
        num_head_blocks: int = 2,
        head_blocks_drop_path: float = 0.3,
        common_grid: int = COMMON_GRID,
        use_cls: bool = False,
    ):
        super().__init__()
        from dinov3.layers import SelfAttentionBlock, SwiGLUFFN

        self.proj_dim    = proj_dim
        self.common_grid = common_grid
        self.use_cls     = use_cls
        head_dim         = proj_dim * len(HOOK_LAYERS)   # 512 * 3 = 1536
        self.head_dim    = head_dim
        self.embed_dim   = head_dim * 2 if use_cls else head_dim  # 3072 or 1536

        # ---- Per-layer MLP projections (independent weights) ----
        self.mlp_2 = LayerMLP(LAYER_CHANNELS[2], proj_dim)
        self.mlp_5 = LayerMLP(LAYER_CHANNELS[5], proj_dim)
        self.mlp_8 = LayerMLP(LAYER_CHANNELS[8], proj_dim)

        # ---- Transformer head on concatenated 1536-dim tokens ----
        num_heads = 16               # head_dim per head = 96; 1536 % (4*16) = 0 ✓
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                head_dim,
                num_heads,
                ffn_layer=partial(SwiGLUFFN, align_to=64),
                init_values=1e-5,
                drop_path=head_blocks_drop_path,
            )
            for _ in range(num_head_blocks)
        ])
        self.ln_final = nn.LayerNorm(head_dim)

        # ---- Learnable CLS token (optional) ----
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, head_dim))

        # ---- RoPE for spatial position encoding (21×21 grid after interpolation) ----
        from dinov3.layers import RopePositionEmbedding
        self.head_rope = RopePositionEmbedding(
            embed_dim=head_dim,
            num_heads=num_heads,
            base=100.0,
        )

        # ---- Frozen backbone (loaded separately via load_backbone) ----
        self._pipe = None
        self._empty_text_emb = None
        self._captured = {}   # populated by persistent hooks
        self._hooks = []


    # ------------------------------------------------------------------
    # Backbone loading (called by AlignmentModel after __init__)
    # ------------------------------------------------------------------

    def load_backbone(self, device: torch.device):
        """Load frozen SD 2.1 + CleanDIFT UNet. Register persistent hooks."""
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        logger.info(f"Loading CleanDIFT backbone on {device}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=torch.float16
        ).to(device)

        ckpt = hf_hub_download(repo_id=CLEANDIFT_REPO, filename=CLEANDIFT_FILE)
        pipe.unet.load_state_dict(load_file(ckpt), strict=True)
        pipe.unet.eval()
        pipe.vae.eval()

        for p in pipe.unet.parameters():
            p.requires_grad_(False)
        for p in pipe.vae.parameters():
            p.requires_grad_(False)

        # Precompute empty text embedding once
        with torch.no_grad():
            tok = pipe.tokenizer(
                "", padding="max_length", max_length=77, return_tensors="pt"
            )
            empty_emb = pipe.text_encoder(tok.input_ids.to(device))[0].half()

        # Use object.__setattr__ to bypass nn.Module.__setattr__ so the SD
        # pipeline is stored in __dict__ directly and never registered as a
        # submodule — important because this is called after FSDP wrapping.
        object.__setattr__(self, '_pipe', pipe)
        object.__setattr__(self, '_empty_text_emb', empty_emb)

        # Register persistent hooks
        self._register_hooks()
        logger.info("CleanDIFT backbone loaded and hooks registered.")

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

        def make_hook(layer_idx):
            def fn(module, inp, output):
                # output: (B, C, H, W) — capture as (B, H*W, C) float32 cpu
                B, C, H, W = output.shape
                self._captured[layer_idx] = (
                    output.detach().float().permute(0, 2, 3, 1).reshape(B, H * W, C)
                )
            return fn

        for l in HOOK_LAYERS:
            up_idx  = l // 3
            res_idx = l % 3
            h = self._pipe.unet.up_blocks[up_idx].resnets[res_idx].register_forward_hook(
                make_hook(l)
            )
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _interp(self, tokens: torch.Tensor, grid: int) -> torch.Tensor:
        """Bilinearly interpolate (B, N, C) spatial tokens to grid×grid."""
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        if H == grid:
            return tokens
        x = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        x = F.interpolate(x, size=(grid, grid), mode="bilinear", align_corners=False)
        return x.reshape(B, C, grid * grid).permute(0, 2, 1)  # (B, grid², C)

    # ------------------------------------------------------------------
    # Init weights (trainable heads only)
    # ------------------------------------------------------------------

    def init_weights(self):
        from dinov3.models.vision_transformer import init_weights_vit
        from dinov3.utils import named_apply

        for block in self.blocks:
            named_apply(init_weights_vit, block)
        self.ln_final.reset_parameters()
        self.head_rope._init_weights()
        for mlp in [self.mlp_2, self.mlp_5, self.mlp_8]:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    # Forward: backbone only (for GradCache split if needed)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_backbone_only(self, images: torch.Tensor):
        """Run frozen backbone. Returns raw layer tokens."""
        assert self._pipe is not None, "Call load_backbone() first"
        images = images.to(self._empty_text_emb.device, dtype=torch.float16)
        B = images.shape[0]

        latent = self._pipe.vae.encode(images).latent_dist.sample()
        latent = (latent * self._pipe.vae.config.scaling_factor).half()
        t = torch.zeros(B, dtype=torch.long, device=images.device)
        self._pipe.unet(
            latent, t,
            encoder_hidden_states=self._empty_text_emb.expand(B, -1, -1)
        )
        # Return copies so captured dict can be reused next step
        return (
            self._captured[2].clone(),   # (B, 25, 1280)
            self._captured[5].clone(),   # (B, 100, 1280)
            self._captured[8].clone(),   # (B, 441, 640)
        )

    # ------------------------------------------------------------------
    # Forward: head only (operates on pre-extracted backbone features)
    # ------------------------------------------------------------------

    def forward_head_only(
        self,
        layer2: torch.Tensor,
        layer5: torch.Tensor,
        layer8: torch.Tensor,
    ) -> CleanDIFTEncoderOutput:
        """Run trainable heads on raw backbone features."""
        device = next(self.parameters()).device

        # Move to training device (features arrive as float32 cpu from backbone)
        layer2 = layer2.to(device)
        layer5 = layer5.to(device)
        layer8 = layer8.to(device)

        # Structure reference: mean-pool raw backbone tokens per layer, concatenate.
        # Dims don't need to match aligned — structure loss compares (B,B) similarity matrices.
        with torch.no_grad():
            g2 = layer2.mean(dim=1)   # (B, 1280)
            g5 = layer5.mean(dim=1)   # (B, 1280)
            g8 = layer8.mean(dim=1)   # (B, 640)
            image_original = torch.cat([g2, g5, g8], dim=-1).detach()  # (B, 3200)

        # Per-layer MLP projection
        l2 = self.mlp_2(layer2)   # (B, 25,  512)
        l5 = self.mlp_5(layer5)   # (B, 100, 512)
        l8 = self.mlp_8(layer8)   # (B, 441, 512)

        # Interpolate all to common grid (21×21 = 441)
        l2 = self._interp(l2, self.common_grid)   # (B, 441, 512)
        l5 = self._interp(l5, self.common_grid)   # (B, 441, 512)
        # l8 already 441 tokens — no-op but call for consistency
        l8 = self._interp(l8, self.common_grid)   # (B, 441, 512)

        # Concatenate along feature dim
        tokens = torch.cat([l2, l5, l8], dim=-1)   # (B, 441, 1536)
        backbone_patches = tokens.detach().clone()

        # Prepend learnable CLS token if enabled
        if self.use_cls:
            B = tokens.shape[0]
            cls_expand = self.cls_token.expand(B, -1, -1)  # (B, 1, 1536)
            tokens = torch.cat([cls_expand, tokens], dim=1)  # (B, 442, 1536)

        # RoPE for the common spatial grid (e.g. 21×21 = 441 tokens)
        # Attention layer auto-skips RoPE for prefix tokens (CLS) via prefix = N - rope_len
        rope = self.head_rope(H=self.common_grid, W=self.common_grid)

        # Self-attention blocks with RoPE
        for block in self.blocks:
            tokens = block(tokens, rope_or_rope_list=rope)
        tokens = self.ln_final(tokens)

        if self.use_cls:
            cls_out = tokens[:, 0]                           # (B, 1536)
            patch_mean = tokens[:, 1:].mean(dim=1)           # (B, 1536)
            aligned = torch.cat([cls_out, patch_mean], dim=-1)  # (B, 3072)
            patch_tokens = tokens[:, 1:]                     # (B, 441, 1536)
        else:
            aligned = tokens.mean(dim=1)                     # (B, 1536)
            patch_tokens = tokens                            # (B, 441, 1536)

        return CleanDIFTEncoderOutput(
            aligned=aligned,
            original=image_original,
            patch_tokens=patch_tokens,
            backbone_patches=backbone_patches,
        )

    # ------------------------------------------------------------------
    # Full forward (backbone + head in one call — used in live training)
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> CleanDIFTEncoderOutput:
        layer2, layer5, layer8 = self.forward_backbone_only(images)
        return self.forward_head_only(layer2, layer5, layer8)
