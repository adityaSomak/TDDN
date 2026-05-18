"""
CLIP text encoder with trainable head blocks.

Architecture (mirrors image encoder design):
  - Frozen CLIP ViT-B text backbone (token_embedding + positional_embedding +
    transformer + ln_final)
  - 2 trainable SelfAttentionBlocks on the full token sequence
  - EOS-token pooling after head blocks
  - Linear projection to embed_dim

Returns both the original (frozen, pre-head EOS) and aligned (post-head EOS) embeddings.
"""

import logging
from functools import partial
from typing import NamedTuple

import open_clip
import torch
from torch import nn

logger = logging.getLogger("text_alignment")


class TextEncoderOutput(NamedTuple):
    aligned: torch.Tensor    # (B, embed_dim) after head blocks + projection
    original: torch.Tensor   # (B, text_dim) frozen EOS embedding (pre-projection)


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        embed_dim: int = 512,
        num_head_blocks: int = 2,
        head_blocks_drop_path: float = 0.1,
    ):
        super().__init__()
        from dinov3.layers import SelfAttentionBlock, SwiGLUFFN

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Extract text pipeline components from the CLIP model
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask

        self.backbone_dim = self.positional_embedding.shape[-1]
        self.text_dim = self.text_projection.shape[-1] if self.text_projection is not None else self.backbone_dim

        first_block = self.transformer.resblocks[0]
        self.num_heads = first_block.attn.num_heads

        del clip_model

        # Freeze the entire backbone
        for p in self.token_embedding.parameters():
            p.requires_grad = False
        if isinstance(self.positional_embedding, nn.Parameter):
            self.positional_embedding.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.ln_final.parameters():
            p.requires_grad = False
        if isinstance(self.text_projection, nn.Parameter):
            self.text_projection.requires_grad = False
        elif self.text_projection is not None:
            for p in self.text_projection.parameters():
                p.requires_grad = False

        # Trainable head: SelfAttentionBlocks on the full token sequence
        self.num_head_blocks = num_head_blocks
        if num_head_blocks > 0:
            self.blocks = nn.ModuleList([
                SelfAttentionBlock(
                    self.backbone_dim,
                    self.num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=head_blocks_drop_path,
                )
                for _ in range(num_head_blocks)
            ])
            self.head_ln_final = nn.LayerNorm(self.backbone_dim)
        else:
            self.blocks = nn.ModuleList([nn.Identity()])
            self.head_ln_final = nn.Identity()
        
        # Projection: always needed since text_dim (512 for CLIP) != embed_dim (1536)
        self.projection = nn.Linear(self.text_dim, embed_dim, bias=False)

    def init_weights(self):
        from dinov3.models.vision_transformer import init_weights_vit
        from dinov3.utils import named_apply

        if self.num_head_blocks > 0:
            for block in self.blocks:
                named_apply(init_weights_vit, block)
            self.head_ln_final.reset_parameters()
        nn.init.normal_(self.projection.weight, std=self.text_dim ** -0.5)

    def _run_frozen_backbone(self, text_tokens: torch.Tensor) -> tuple:
        """Run the frozen CLIP text pipeline up to ln_final."""
        with torch.no_grad():
            x = self.token_embedding(text_tokens)
            x = x + self.positional_embedding
            attn_mask = self.attn_mask
            if attn_mask is not None:
                attn_mask = attn_mask.to(x.device, dtype=x.dtype)
            x = self.transformer(x, attn_mask=attn_mask)
            x = self.ln_final(x)

            eos_indices = text_tokens.argmax(dim=-1)
            eos_tokens = x[torch.arange(x.shape[0], device=x.device), eos_indices]
            if self.text_projection is not None:
                eos_projected = eos_tokens @ self.text_projection
            else:
                eos_projected = eos_tokens

        return x, eos_indices, eos_projected

    def forward_backbone_only(self, text_tokens: torch.Tensor):
        """Run frozen backbone only. Returns (all_tokens, eos_indices, eos_projected)."""
        return self._run_frozen_backbone(text_tokens)

    def forward_head_only(self, all_tokens: torch.Tensor, eos_indices: torch.Tensor, eos_projected: torch.Tensor) -> TextEncoderOutput:
        """Run trainable heads on pre-extracted backbone features."""
        x = all_tokens
        for block in self.blocks:
            x = block(x)
        x = self.head_ln_final(x)
        eos_features = x[torch.arange(x.shape[0], device=x.device), eos_indices]
        aligned = self.projection(eos_features)

        with torch.no_grad():
            original = self.projection(eos_projected)

        return TextEncoderOutput(aligned=aligned, original=original.detach())

    def forward(self, text_tokens: torch.Tensor) -> TextEncoderOutput:
        all_tokens, eos_indices, eos_projected = self._run_frozen_backbone(text_tokens)

        x = all_tokens
        for block in self.blocks:
            x = block(x)

        x = self.head_ln_final(x)
        eos_features = x[torch.arange(x.shape[0], device=x.device), eos_indices]
        aligned = self.projection(eos_features)
        
        # Project original to same dimension as aligned for structure loss
        with torch.no_grad():
            original = self.projection(eos_projected)

        return TextEncoderOutput(
            aligned=aligned,
            original=original.detach(),
        )
