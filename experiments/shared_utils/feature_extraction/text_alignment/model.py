"""
AlignmentModel: dual-encoder wrapper that ties image + text encoders and manages logit_scale.

forward() returns everything needed by all three loss functions:
  - normalized image/text aligned embeddings (for CLIP loss)
  - logit_scale (for CLIP loss + structure reg temperature)
  - head patch tokens + backbone patch tokens (for gram loss)
  - original (frozen) image/text embeddings (for structure reg)
"""

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import AlignConfig
from .image_encoder import DINOv3ImageEncoder
from .cleandift_encoder import CleanDIFTImageEncoder
from .fused_encoder import FusedImageEncoder
from .text_encoder import CLIPTextEncoder
from .roberta_encoder import RoBERTaTextEncoder

logger = logging.getLogger("text_alignment")

import os

# Gated DINOv3 / RoBERTa weights require an HF token. Read from the
# environment so it never gets committed.
HF_TOKEN = os.environ.get("HF_TOKEN")


class AlignmentOutput(NamedTuple):
    image_features: torch.Tensor       # (B, E) L2-normalized
    text_features: torch.Tensor        # (B, E) L2-normalized
    logit_scale: torch.Tensor          # scalar
    patch_tokens: torch.Tensor         # (B, N, D) head patches
    backbone_patch_tokens: torch.Tensor  # (B, N, D) frozen patches
    image_original: torch.Tensor       # (B, E) frozen pooled
    text_original: torch.Tensor        # (B, text_dim) frozen EOS


def _build_vision_backbone(config: AlignConfig) -> nn.Module:
    """Load DINOv3 backbone from HuggingFace with pretrained weights."""
    model_id = config.vision_backbone_hf_model_id
    if not model_id:
        raise ValueError("vision_backbone_hf_model_id must be set")

    logger.info(f"Loading DINOv3 backbone: {model_id}")
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


class AlignmentModel(nn.Module):
    def __init__(self, config: AlignConfig, vision_backbone: Optional[nn.Module] = None):
        super().__init__()

        self._use_cleandift = config.use_cleandift
        self._use_fused = getattr(config, 'use_fused_encoder', False)

        if config.use_cleandift:
            # CleanDIFT path — backbone loaded separately after FSDP setup
            self.image_encoder = CleanDIFTImageEncoder(
                proj_dim=config.cleandift_proj_dim,
                num_head_blocks=config.vision_num_head_blocks,
                head_blocks_drop_path=config.head_blocks_drop_path,
                common_grid=config.cleandift_common_grid,
                use_cls=config.cleandift_use_cls,
            )
            logger.info(
                f"CleanDIFT encoder: proj_dim={config.cleandift_proj_dim}, "
                f"embed_dim={self.image_encoder.embed_dim}, "
                f"common_grid={config.cleandift_common_grid}×{config.cleandift_common_grid}"
            )
        elif self._use_fused:
            # Fused DINOv3+CleanDIFT path — DINOv3 backbone in module tree,
            # CleanDIFT SD pipeline loaded separately after FSDP setup
            if vision_backbone is None:
                vision_backbone = _build_vision_backbone(config)
            hf_config = vision_backbone.config
            self.image_encoder = FusedImageEncoder(
                dino_backbone=vision_backbone,
                backbone_dim=hf_config.hidden_size,
                num_heads=hf_config.num_attention_heads,
                patch_size=hf_config.patch_size,
                num_register_tokens=getattr(hf_config, "num_register_tokens", 4),
                rope_theta=getattr(hf_config, "rope_theta", 100.0),
                num_head_blocks=config.vision_num_head_blocks,
                head_blocks_drop_path=config.head_blocks_drop_path,
                cleandift_proj_dim=config.cleandift_proj_dim,
                common_grid=config.cleandift_common_grid,
            )
            logger.info(
                f"Fused DINOv3+CleanDIFT encoder: backbone_dim={hf_config.hidden_size}, "
                f"embed_dim={self.image_encoder.embed_dim}, "
                f"cleandift_proj_dim={config.cleandift_proj_dim}, "
                f"common_grid={config.cleandift_common_grid}×{config.cleandift_common_grid}"
            )
        else:
            if vision_backbone is None:
                vision_backbone = _build_vision_backbone(config)

            hf_config = vision_backbone.config
            self.image_encoder = DINOv3ImageEncoder(
                backbone=vision_backbone,
                backbone_dim=hf_config.hidden_size,
                num_heads=hf_config.num_attention_heads,
                patch_size=hf_config.patch_size,
                num_register_tokens=getattr(hf_config, "num_register_tokens", 4),
                rope_theta=getattr(hf_config, "rope_theta", 100.0),
                embed_dim=config.embed_dim,
                num_head_blocks=config.vision_num_head_blocks,
                head_blocks_drop_path=config.head_blocks_drop_path,
                use_rope_in_head=config.use_rope_in_head,
                use_linear_projection=config.use_linear_projection,
            )

        if config.text_encoder_name.startswith("roberta") or "sentence-transformers" in config.text_encoder_name:
            self.text_encoder = RoBERTaTextEncoder(
                model_name=config.text_encoder_name,
                layer_idx=config.text_layer_idx,
                embed_dim=config.embed_dim,
                num_head_blocks=config.text_num_head_blocks,
                head_blocks_drop_path=config.text_head_blocks_drop_path,
            )
        else:
            self.text_encoder = CLIPTextEncoder(
                model_name=config.text_encoder_name,
                pretrained=config.text_encoder_pretrained,
                embed_dim=config.embed_dim,
                num_head_blocks=config.text_num_head_blocks,
                head_blocks_drop_path=config.text_head_blocks_drop_path,
            )

        self.logit_scale = nn.Parameter(torch.ones(1) * config.init_logit_scale)

        if config.freeze_logit_scale:
            self.logit_scale.requires_grad = False

    def init_weights(self):
        """Initialize only the trainable head blocks (backbones already loaded)."""
        self.image_encoder.init_weights()
        self.text_encoder.init_weights()

    def load_cleandift_backbone(self, device: torch.device):
        """Load frozen CleanDIFT backbone onto device (call after FSDP setup)."""
        assert self._use_cleandift or self._use_fused, \
            "Only valid when use_cleandift=True or use_fused_encoder=True"
        self.image_encoder.load_backbone(device)

    @torch.no_grad()
    def extract_frozen_features(self, images: torch.Tensor, text_inputs):
        """Run frozen backbones only. Returns raw features for GradCache accumulation."""
        if self._use_cleandift:
            # Returns (layer2, layer5, layer8) tuple
            vis_feats = self.image_encoder.forward_backbone_only(images)
        elif self._use_fused:
            # Returns (dino_cls, dino_patches, layer2, layer5, layer8) tuple
            vis_feats = self.image_encoder.forward_backbone_only(images)
        else:
            vis_cls, vis_patches = self.image_encoder.forward_backbone_only(images)
            vis_feats = (vis_cls, vis_patches)

        if isinstance(self.text_encoder, RoBERTaTextEncoder):
            txt_feats = self.text_encoder.forward_backbone_only(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            )
        else:
            txt_feats = self.text_encoder.forward_backbone_only(text_inputs)
        return vis_feats, txt_feats

    def forward_heads(
        self, vis_feats, txt_feats, img_size: int = 336
    ) -> AlignmentOutput:
        """Run trainable alignment heads on pre-extracted frozen features."""
        if self._use_cleandift:
            layer2, layer5, layer8 = vis_feats
            img_out = self.image_encoder.forward_head_only(layer2, layer5, layer8)
        elif self._use_fused:
            dino_cls, dino_patches, layer2, layer5, layer8 = vis_feats
            img_out = self.image_encoder.forward_head_only(
                dino_cls, dino_patches, layer2, layer5, layer8
            )
        else:
            vis_cls, vis_patches = vis_feats
            img_out = self.image_encoder.forward_head_only(vis_cls, vis_patches, img_size)

        if isinstance(self.text_encoder, RoBERTaTextEncoder):
            txt_out = self.text_encoder.forward_head_only(*txt_feats)
        else:
            txt_out = self.text_encoder.forward_head_only(*txt_feats)

        image_features = F.normalize(img_out.aligned, dim=-1)
        text_features = F.normalize(txt_out.aligned, dim=-1)

        return AlignmentOutput(
            image_features=image_features,
            text_features=text_features,
            logit_scale=self.logit_scale.exp(),
            patch_tokens=img_out.patch_tokens,
            backbone_patch_tokens=img_out.backbone_patches,
            image_original=img_out.original,
            text_original=txt_out.original,
        )

    def forward(
        self, images: torch.Tensor, text_inputs
    ) -> AlignmentOutput:
        img_out = self.image_encoder(images)

        if isinstance(self.text_encoder, RoBERTaTextEncoder):
            txt_out = self.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])
        else:
            txt_out = self.text_encoder(text_inputs)

        image_features = F.normalize(img_out.aligned, dim=-1)
        text_features = F.normalize(txt_out.aligned, dim=-1)

        return AlignmentOutput(
            image_features=image_features,
            text_features=text_features,
            logit_scale=self.logit_scale.exp(),
            patch_tokens=img_out.patch_tokens,
            backbone_patch_tokens=img_out.backbone_patches,
            image_original=img_out.original,
            text_original=txt_out.original,
        )
