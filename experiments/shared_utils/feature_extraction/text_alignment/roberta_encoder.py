"""
RoBERTa text encoder with trainable head blocks.

Architecture (mirrors image encoder design):
  - Frozen RoBERTa backbone (sentence-transformers/all-roberta-large-v1)
  - Extract layer 24 (last transformer layer) via output_hidden_states
  - 2 trainable SelfAttentionBlocks on all tokens
  - Average pooling AFTER head blocks (with attention mask)
  - Linear projection to embed_dim

Returns both the original (frozen, pre-head pooled) and aligned (post-head pooled) embeddings.
"""

import logging
from functools import partial
from typing import NamedTuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("text_alignment")


class TextEncoderOutput(NamedTuple):
    aligned: torch.Tensor    # (B, embed_dim) after head blocks + projection
    original: torch.Tensor   # (B, embed_dim) frozen pooled embedding (pre-head)


class RoBERTaTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-roberta-large-v1",
        layer_idx: int = 24,
        embed_dim: int = 1280,
        num_head_blocks: int = 2,
        head_blocks_drop_path: float = 0.1,
        skip_backbone: bool = False,
    ):
        super().__init__()
        from dinov3.layers import SelfAttentionBlock, SwiGLUFFN

        self.layer_idx = layer_idx

        if not skip_backbone:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.backbone.config.output_hidden_states = True
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone_dim = self.backbone.config.hidden_size
            self.num_heads = self.backbone.config.num_attention_heads
        else:
            # Head-only teacher: no backbone loaded, dims hardcoded for roberta-large
            self.backbone = None
            self.tokenizer = None
            self.backbone_dim = 1024  # roberta-large hidden size
            self.num_heads = 16

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
        
        self.projection = nn.Linear(self.backbone_dim, embed_dim, bias=False)

    def init_weights(self):
        from dinov3.models.vision_transformer import init_weights_vit
        from dinov3.utils import named_apply

        if self.num_head_blocks > 0:
            for block in self.blocks:
                named_apply(init_weights_vit, block)
            self.head_ln_final.reset_parameters()
        nn.init.normal_(self.projection.weight, std=self.backbone_dim ** -0.5)

    def forward_backbone_only(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run frozen backbone only. Returns (text_tokens, attention_mask)."""
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            text_tokens = outputs.hidden_states[self.layer_idx]
        return text_tokens, attention_mask

    def forward_head_only(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> TextEncoderOutput:
        """Run trainable heads on pre-extracted backbone features."""
        mask = attention_mask.unsqueeze(-1).to(text_tokens.dtype)  # (B, L, 1)
        x = text_tokens * mask  # zero out padding before head blocks
        for block in self.blocks:
            x = block(x)
            x = x * mask  # re-zero after each block to prevent contamination
        x = self.head_ln_final(x)

        pooled = (x * mask).sum(1) / mask.sum(1)
        aligned = self.projection(pooled)

        with torch.no_grad():
            original_pooled = (text_tokens * mask).sum(1) / mask.sum(1)
            original = self.projection(original_pooled)

        return TextEncoderOutput(aligned=aligned, original=original.detach())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextEncoderOutput:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            text_tokens = outputs.hidden_states[self.layer_idx]

        self._last_backbone_feats = (text_tokens, attention_mask)

        mask = attention_mask.unsqueeze(-1).to(text_tokens.dtype)  # (B, L, 1)
        x = text_tokens * mask  # zero out padding before head blocks
        for block in self.blocks:
            x = block(x)
            x = x * mask  # re-zero after each block to prevent contamination

        x = self.head_ln_final(x)

        pooled = (x * mask).sum(1) / mask.sum(1)
        aligned = self.projection(pooled)

        with torch.no_grad():
            original_pooled = (text_tokens * mask).sum(1) / mask.sum(1)
            original = self.projection(original_pooled)

        return TextEncoderOutput(
            aligned=aligned,
            original=original.detach(),
        )
