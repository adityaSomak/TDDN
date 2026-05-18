"""
Image encoder using HuggingFace DINOv3 backbone.

Architecture:
  - Frozen HF DINOv3 backbone (outputs hidden_states)
  - Discard register tokens, keep CLS + patches
  - 2 trainable SelfAttentionBlocks with RoPE for spatial awareness
  - Final embedding = concat(proj(CLS), proj(mean(patches))) -> embed_dim
  - Returns both aligned (after head) and original (frozen backbone) embeddings
"""

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import NamedTuple

import torch
from torch import nn

logger = logging.getLogger("text_alignment")


class ImageEncoderOutput(NamedTuple):
    aligned: torch.Tensor        # (B, embed_dim) after head + projection
    original: torch.Tensor       # (B, embed_dim) frozen backbone pooled
    patch_tokens: torch.Tensor   # (B, N, D) projected patch tokens from head
    backbone_patches: torch.Tensor  # (B, N, D) frozen backbone patch tokens


class ImageEncoderBase(ABC):
    """ABC so the image backbone can be swapped in the future."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> ImageEncoderOutput:
        ...

    @abstractmethod
    def init_weights(self) -> None:
        ...


class DINOv3ImageEncoder(nn.Module, ImageEncoderBase):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int = 768,
        num_heads: int = 12,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        rope_theta: float = 100.0,
        embed_dim: int = 512,
        num_head_blocks: int = 2,
        head_blocks_drop_path: float = 0.3,
        use_rope_in_head: bool = True,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        from dinov3.layers import SelfAttentionBlock, SwiGLUFFN

        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_special_tokens = 1 + num_register_tokens  # CLS + register tokens
        self.num_head_blocks = num_head_blocks
        self.embed_dim = embed_dim
        self.use_linear_projection = use_linear_projection

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

        # Projection logic matching DINOv3:
        # If use_linear_projection=false AND embed_dim = backbone_dim * 2, use identity
        # Otherwise, project each component to embed_dim // 2
        multiplier = 2  # CLS + patches
        self.cls_projection = nn.Identity()
        self.patch_projection = nn.Identity()
        
        if multiplier * backbone_dim != embed_dim or use_linear_projection:
            assert embed_dim % 2 == 0, "embed_dim must be even for CLS+patch concat"
            logger.info(f"Vision: Using linear projection {backbone_dim} -> {embed_dim // 2} per component")
            self.cls_projection = nn.Linear(backbone_dim, embed_dim // 2, bias=False)
            self.patch_projection = nn.Linear(backbone_dim, embed_dim // 2, bias=False)
        else:
            logger.info(f"Vision: Direct concat (no projection), {backbone_dim} * 2 = {embed_dim}")

        if use_rope_in_head:
            from dinov3.layers import RopePositionEmbedding
            self.head_rope = RopePositionEmbedding(
                embed_dim=backbone_dim,
                num_heads=num_heads,
                base=rope_theta,
            )
        else:
            self.head_rope = None

    def init_weights(self):
        from dinov3.models.vision_transformer import init_weights_vit
        from dinov3.utils import named_apply

        for block in self.blocks:
            named_apply(init_weights_vit, block)
        self.ln_final.reset_parameters()
        if isinstance(self.cls_projection, nn.Linear):
            nn.init.normal_(self.cls_projection.weight, std=self.backbone_dim ** -0.5)
        if isinstance(self.patch_projection, nn.Linear):
            nn.init.normal_(self.patch_projection.weight, std=self.backbone_dim ** -0.5)
        if self.head_rope is not None:
            self.head_rope._init_weights()

    def _pool_original(
        self, cls_token: torch.Tensor, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Pool frozen backbone features into embed_dim for structure reg."""
        cls_proj = self.cls_projection(cls_token)
        patch_proj = self.patch_projection(patch_tokens.mean(dim=1))
        return torch.cat([cls_proj, patch_proj], dim=-1)

    def forward_backbone_only(self, images: torch.Tensor):
        """Run frozen backbone only. Returns (cls_token, backbone_patches)."""
        with torch.no_grad():
            outputs = self.backbone(images)
            last_hidden = outputs.last_hidden_state
            cls_token = last_hidden[:, 0]
            backbone_patches = last_hidden[:, self.num_special_tokens:]
        return cls_token, backbone_patches

    def forward_head_only(self, cls_token: torch.Tensor, backbone_patches: torch.Tensor, img_size: int) -> ImageEncoderOutput:
        """Run trainable heads on pre-extracted backbone features."""
        with torch.no_grad():
            original = self._pool_original(cls_token, backbone_patches)

        image_tokens = torch.cat([cls_token.unsqueeze(1), backbone_patches], dim=1)
        H = W = img_size // self.patch_size

        rope = None
        if self.head_rope is not None:
            rope = self.head_rope(H=H, W=W)

        for block in self.blocks:
            image_tokens = block(image_tokens, rope_or_rope_list=rope)

        image_tokens = self.ln_final(image_tokens)
        head_cls = image_tokens[:, 0]
        head_patches = image_tokens[:, 1:]

        cls_proj = self.cls_projection(head_cls)
        patch_proj = self.patch_projection(head_patches.mean(dim=1))
        aligned = torch.cat([cls_proj, patch_proj], dim=-1)

        return ImageEncoderOutput(
            aligned=aligned,
            original=original.detach(),
            patch_tokens=head_patches,
            backbone_patches=backbone_patches.detach(),
        )

    def forward(self, images: torch.Tensor) -> ImageEncoderOutput:
        with torch.no_grad():
            outputs = self.backbone(images)
            last_hidden = outputs.last_hidden_state  # (B, N_tokens, D)
            cls_token = last_hidden[:, 0]  # (B, D)
            backbone_patches = last_hidden[:, self.num_special_tokens:]  # (B, N_patches, D)

        with torch.no_grad():
            original = self._pool_original(cls_token, backbone_patches)

        # CLS + patches enter head blocks (registers discarded)
        image_tokens = torch.cat([cls_token.unsqueeze(1), backbone_patches], dim=1)

        B, _, h, w = images.shape
        H, W = h // self.patch_size, w // self.patch_size

        rope = None
        if self.head_rope is not None:
            rope = self.head_rope(H=H, W=W)

        for block in self.blocks:
            image_tokens = block(image_tokens, rope_or_rope_list=rope)

        image_tokens = self.ln_final(image_tokens)

        head_cls = image_tokens[:, 0]        # (B, D)
        head_patches = image_tokens[:, 1:]   # (B, N, D)

        cls_proj = self.cls_projection(head_cls)
        patch_proj = self.patch_projection(head_patches.mean(dim=1))
        aligned = torch.cat([cls_proj, patch_proj], dim=-1)  # (B, embed_dim)

        return ImageEncoderOutput(
            aligned=aligned,
            original=original.detach(),
            patch_tokens=head_patches,
            backbone_patches=backbone_patches.detach(),
        )
