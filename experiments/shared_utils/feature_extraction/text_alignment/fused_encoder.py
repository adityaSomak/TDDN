"""
Fused DINOv3 ViT-H + CleanDIFT image encoder with trainable alignment heads.

Architecture:
  - Frozen DINOv3 ViT-H backbone (module tree, FSDP-wrapped)
    → CLS (B, 1280) + patches (B, 441, 1280)
  - Frozen CleanDIFT SD 2.1 UNet backbone (loaded post-FSDP via object.__setattr__)
    → layers 2, 5, 8 → LayerMLP per layer (C→512) → interp 21×21 → L2 normalize per layer
    → cat → (B, 441, 1536)
  - Fusion: LayerNorm DINOv3 side, cat → (B, 441, 2816)
    → fusion_mlp: Linear(2816,2048)→LN→GELU→Linear(2048,1280)→LN → (B, 441, 1280)
  - Prepend DINOv3 CLS: (B, 442, 1280)
  - 2× SelfAttentionBlock(1280, 16 heads) + LayerNorm + RoPE
  - Output: cat([tokens[:,0], tokens[:,1:].mean(1)]) → (B, 2560)

Structure reference (frozen, semantic, DINOv3 only):
  - cat([dino_cls, F.normalize(dino_patches).mean(1)]).detach() → (B, 2560)
  - Structure loss compares (B,B) similarity matrices only — no dim constraint
"""

import logging
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .image_encoder import ImageEncoderOutput
from .cleandift_encoder import LayerMLP, SD_MODEL_ID, CLEANDIFT_REPO, CLEANDIFT_FILE, HOOK_LAYERS, LAYER_CHANNELS, COMMON_GRID

logger = logging.getLogger("text_alignment")


class FusedImageEncoder(nn.Module):
    """
    Fused DINOv3 ViT-H + CleanDIFT image encoder.

    DINOv3 backbone lives in the module tree (frozen, FSDP-wrapped).
    CleanDIFT SD pipeline loaded post-FSDP via object.__setattr__().
    """

    def __init__(
        self,
        dino_backbone: nn.Module,
        backbone_dim: int = 1280,
        num_heads: int = 16,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        rope_theta: float = 100.0,
        num_head_blocks: int = 2,
        head_blocks_drop_path: float = 0.1,
        cleandift_proj_dim: int = 512,
        common_grid: int = COMMON_GRID,
    ):
        super().__init__()
        from dinov3.layers import SelfAttentionBlock, SwiGLUFFN

        self.backbone_dim = backbone_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_special_tokens = 1 + num_register_tokens  # CLS + register tokens
        self.common_grid = common_grid
        self.proj_dim = cleandift_proj_dim

        # embed_dim = 2 × backbone_dim (CLS + mean(patches))
        self.embed_dim = 2 * backbone_dim  # 2560

        # ---- DINOv3 backbone (frozen, in module tree for FSDP) ----
        # dino_backbone may be None when building a head-only teacher model
        if dino_backbone is not None:
            self.backbone = dino_backbone

        # ImageNet → SD re-normalization constants (applied inside backbone forward)
        # Images arrive as ImageNet-normalized (for DINOv3); CleanDIFT needs [-1,1].
        # x_sd = (x_raw - 0.5) / 0.5,  x_raw = x_imagenet * std + mean
        # Combined: x_sd = x_imagenet * (std / 0.5) + (mean - 0.5) / 0.5
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('_imagenet_mean', _mean)
        self.register_buffer('_imagenet_std',  _std)

        # ---- Per-layer CleanDIFT MLP projections ----
        self.mlp_2 = LayerMLP(LAYER_CHANNELS[2], cleandift_proj_dim)  # 1280 → 512
        self.mlp_5 = LayerMLP(LAYER_CHANNELS[5], cleandift_proj_dim)  # 1280 → 512
        self.mlp_8 = LayerMLP(LAYER_CHANNELS[8], cleandift_proj_dim)  #  640 → 512

        # ---- Patch normalization before fusion ----
        cd_patch_dim = cleandift_proj_dim * len(HOOK_LAYERS)   # 1536
        # CleanDIFT layers are L2-normalized per-layer before concat — no learnable norm needed
        self.dino_patch_norm = nn.LayerNorm(backbone_dim)      # normalize DINOv3 patches (B, 441, 1280)

        # ---- Fusion MLP: 2816 → 2048 → 1280 with LN at each step ----
        fuse_in = backbone_dim + cd_patch_dim  # 1280 + 1536 = 2816
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fuse_in, 2048, bias=False),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, backbone_dim, bias=False),
            nn.LayerNorm(backbone_dim),
        )

        # ---- Transformer head (operates on 1280-dim fused patches + DINOv3 CLS) ----
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                backbone_dim,
                num_heads,
                ffn_layer=partial(SwiGLUFFN, align_to=64),
                init_values=1e-5,
                drop_path=head_blocks_drop_path,
            )
            for _ in range(num_head_blocks)
        ])
        self.ln_final = nn.LayerNorm(backbone_dim)

        # ---- RoPE for common 21×21 spatial grid ----
        from dinov3.layers import RopePositionEmbedding
        self.head_rope = RopePositionEmbedding(
            embed_dim=backbone_dim,
            num_heads=num_heads,
            base=rope_theta,
        )

        # ---- Frozen CleanDIFT backbone (loaded separately post-FSDP) ----
        self._pipe = None
        self._empty_text_emb = None
        self._captured = {}
        self._hooks = []

    # ------------------------------------------------------------------
    # CleanDIFT backbone loading (called after FSDP setup)
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

        with torch.no_grad():
            tok = pipe.tokenizer(
                "", padding="max_length", max_length=77, return_tensors="pt"
            )
            empty_emb = pipe.text_encoder(tok.input_ids.to(device))[0].half()

        # Bypass nn.Module.__setattr__ so SD pipeline is never registered as a
        # submodule — critical because this is called after FSDP wrapping.
        object.__setattr__(self, '_pipe', pipe)
        object.__setattr__(self, '_empty_text_emb', empty_emb)

        self._register_hooks()
        logger.info("CleanDIFT backbone loaded and hooks registered.")

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

        def make_hook(layer_idx):
            def fn(module, inp, output):
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
        return x.reshape(B, C, grid * grid).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Init weights (trainable heads only; backbones already frozen)
    # ------------------------------------------------------------------

    def init_weights(self):
        from dinov3.models.vision_transformer import init_weights_vit
        from dinov3.utils import named_apply

        for block in self.blocks:
            named_apply(init_weights_vit, block)
        self.ln_final.reset_parameters()
        self.head_rope._init_weights()
        self.dino_patch_norm.reset_parameters()

        for mlp in [self.mlp_2, self.mlp_5, self.mlp_8]:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()

    # ------------------------------------------------------------------
    # Forward: backbone only (both DINOv3 and CleanDIFT, for GradCache)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_backbone_only(self, images: torch.Tensor):
        """
        Run both frozen backbones.
        Returns (dino_cls, dino_patches, layer2, layer5, layer8).
        """
        # DINOv3 (in module tree, on training device)
        outputs = self.backbone(images)
        last_hidden = outputs.last_hidden_state   # (B, 446, 1280) for ViT-H+
        dino_cls     = last_hidden[:, 0]                             # (B, 1280)
        dino_patches = last_hidden[:, self.num_special_tokens:]      # (B, 441, 1280)

        # CleanDIFT (SD pipeline on per-rank cuda device)
        assert self._pipe is not None, "Call load_backbone() first"
        # Re-normalize from ImageNet → SD convention ([-1, 1]) before passing to UNet
        cd_device = self._empty_text_emb.device
        images_raw = images.float() * self._imagenet_std.to(images.device) \
                     + self._imagenet_mean.to(images.device)   # undo ImageNet norm
        images_sd  = (images_raw - 0.5) / 0.5                 # SD convention
        images_cd  = images_sd.to(cd_device, dtype=torch.float16)
        B = images_cd.shape[0]
        latent = self._pipe.vae.encode(images_cd).latent_dist.sample()
        latent = (latent * self._pipe.vae.config.scaling_factor).half()
        t = torch.zeros(B, dtype=torch.long, device=images_cd.device)
        self._pipe.unet(
            latent, t,
            encoder_hidden_states=self._empty_text_emb.expand(B, -1, -1)
        )
        layer2 = self._captured[2].clone()   # (B, 25, 1280)
        layer5 = self._captured[5].clone()   # (B, 100, 1280)
        layer8 = self._captured[8].clone()   # (B, 441, 640)

        return dino_cls, dino_patches, layer2, layer5, layer8

    # ------------------------------------------------------------------
    # Forward: head only (operates on pre-extracted backbone features)
    # ------------------------------------------------------------------

    def forward_head_only(
        self,
        dino_cls: torch.Tensor,
        dino_patches: torch.Tensor,
        layer2: torch.Tensor,
        layer5: torch.Tensor,
        layer8: torch.Tensor,
    ) -> ImageEncoderOutput:
        """Run trainable fusion + attention heads on pre-extracted frozen features."""
        device = next(self.parameters()).device

        dino_cls     = dino_cls.to(device)
        dino_patches = dino_patches.to(device)
        layer2 = layer2.to(device)
        layer5 = layer5.to(device)
        layer8 = layer8.to(device)

        # Structure reference: DINOv3 only (frozen semantic anchor).
        with torch.no_grad():
            image_original = torch.cat(
                [dino_cls, F.normalize(dino_patches, dim=-1).mean(dim=1)], dim=-1
            ).detach()   # (B, 2560)

        # ---- CleanDIFT: project → interp → L2 normalize per layer → cat ----
        l2 = F.normalize(self._interp(self.mlp_2(layer2), self.common_grid), dim=-1)   # (B, 441, 512)
        l5 = F.normalize(self._interp(self.mlp_5(layer5), self.common_grid), dim=-1)   # (B, 441, 512)
        l8 = F.normalize(self._interp(self.mlp_8(layer8), self.common_grid), dim=-1)   # (B, 441, 512)

        cd_patches_n = torch.cat([l2, l5, l8], dim=-1)   # (B, 441, 1536) — already normalized

        # ---- Normalize DINOv3 side before fusion ----
        dino_patches_n = self.dino_patch_norm(dino_patches)  # (B, 441, 1280)

        # ---- Fusion: 2816 → 2048 → 1280 ----
        fused = self.fusion_mlp(
            torch.cat([dino_patches_n, cd_patches_n], dim=-1)
        )   # (B, 441, 1280)
        backbone_patches = fused.detach().clone()

        # ---- Prepend DINOv3 CLS → attention ----
        tokens = torch.cat([dino_cls.unsqueeze(1), fused], dim=1)   # (B, 442, 1280)

        rope = self.head_rope(H=self.common_grid, W=self.common_grid)
        for block in self.blocks:
            tokens = block(tokens, rope_or_rope_list=rope)
        tokens = self.ln_final(tokens)

        # ---- Output: cat([CLS, mean(patches)]) ----
        aligned = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)   # (B, 2560)

        return ImageEncoderOutput(
            aligned=aligned,
            original=image_original,
            patch_tokens=tokens[:, 1:],    # (B, 441, 1280)
            backbone_patches=backbone_patches,
        )

    # ------------------------------------------------------------------
    # Full forward (backbone + head — used in live training mode)
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> ImageEncoderOutput:
        dino_cls, dino_patches, layer2, layer5, layer8 = self.forward_backbone_only(images)
        self._last_backbone_feats = (dino_cls, dino_patches, layer2, layer5, layer8)
        return self.forward_head_only(dino_cls, dino_patches, layer2, layer5, layer8)
