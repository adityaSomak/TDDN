"""PyTorch Lightning module for frozen-backbone segmentation training.

Trains a small ``SegmentationHead`` on top of frozen features produced
by the backbone configured in ``models.yaml``. Handcraft fusions stack
two extractors' spatial feature maps via the shared ``fuse_concat``
helper before the head. For components with a fitted PCA basis,
per-layer features are projected through the basis before fusion.

Public API
----------
    SegmentationLitModule(model_tag, model_cfg, model_index, training_cfg,
                          n_classes, *, head_kwargs, class_weights,
                          miou_weights, pca_bases=None)
"""
from __future__ import annotations

from typing import Optional

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from shared_utils.feature_extraction import build_extractor, fuse_concat   # noqa: E402

from .loss import CombinedSegLoss
from .metrics import ConfusionMatrixMetric
from .model import SegmentationHead
from .pca import apply_pca_layers


def _spatial_feature_map(
    out: dict, pca_basis: Optional[dict] = None,
) -> torch.Tensor:
    """Reduce an extractor output dict to ``(B, C, H, W)``.

    For per-layer (diffusion) outputs: optionally project through a
    saved PCA basis, then bilinear-align every layer to the largest
    spatial grid and concat along the channel axis.
    """
    if out.get("patch_tokens") is not None:
        return out["patch_tokens"]
    if out.get("per_layer"):
        per_layer = out["per_layer"]
        if pca_basis is not None:
            per_layer = apply_pca_layers(per_layer, pca_basis)
        H = max(f.shape[-2] for f in per_layer.values())
        W = max(f.shape[-1] for f in per_layer.values())
        parts = []
        for _, f in sorted(per_layer.items()):
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            parts.append(f)
        return torch.cat(parts, dim=1)
    raise RuntimeError("Segmentation requires spatial features from the extractor.")


class SegmentationLitModule(pl.LightningModule):
    """Linear-probe segmentation training driver.

    The backbone is loaded eagerly (frozen). The head is also built
    eagerly via a single dummy forward at init time, so the optimizer
    has parameters to attach to before the first training step.

    Args:
        model_tag:     key of this model in ``model_index`` (used to
                       look up the associated PCA basis, if any).
        model_cfg:     entry from ``configs/models.yaml`` (backbone,
                       extractor kwargs, transform, optional components).
        model_index:   full ``tag -> config`` map so fusion components
                       can resolve their sub-configs.
        training_cfg:  parsed ``configs/training.yaml``.
        n_classes:     number of segmentation classes.
        head_kwargs:   forwarded to ``SegmentationHead``.
        class_weights: ``(C,)`` per-class CE weights.
        miou_weights:  ``(C,)`` per-class weights for the validation mIoU.
        pca_bases:     optional ``{component_tag: {layer_idx: basis}}``;
                       applied to per-layer outputs before fusion.
    """

    def __init__(
        self,
        model_tag: str,
        model_cfg: dict,
        model_index: dict,
        training_cfg: dict,
        n_classes: int,
        *,
        head_kwargs: dict,
        class_weights: torch.Tensor,
        miou_weights: torch.Tensor,
        pca_bases: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["class_weights", "miou_weights", "pca_bases"],
        )
        self.model_tag = model_tag
        self.model_cfg = model_cfg
        self.model_index = model_index
        self.training_cfg = training_cfg
        self.n_classes = n_classes
        self.head_kwargs = dict(head_kwargs)
        self.pca_bases = pca_bases or {}
        self.loss = CombinedSegLoss(
            class_weights,
            dice_weight=training_cfg["loss"]["dice_weight"],
            ignore_index=training_cfg["loss"]["ignore_index"],
        )
        self.val_metric = ConfusionMatrixMetric(
            n_classes, weights=miou_weights,
            ignore_index=training_cfg["loss"]["ignore_index"],
        )
        self.head: Optional[SegmentationHead] = None
        self._build_extractors()
        self._build_head_eager()

    def _build_extractors(self) -> None:
        """Materialize the frozen extractor(s) for the configured backbone."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "components" in self.model_cfg:
            self._components = []
            for comp in self.model_cfg["components"]:
                sub = self.model_index[comp["tag"]]
                ex = build_extractor(
                    sub["backbone"], device,
                    extractor_kwargs=sub.get("extractor", {}) or {},
                    loader_kwargs_override=sub.get("loader_kwargs", {}) or {},
                )
                self._components.append({
                    "tag": comp["tag"],
                    "extractor": ex,
                    "weight": comp.get("weight", 1.0),
                    "pca_basis": self.pca_bases.get(comp["tag"]),
                })
        else:
            self.extractor = build_extractor(
                self.model_cfg["backbone"], device,
                extractor_kwargs=self.model_cfg.get("extractor", {}) or {},
                loader_kwargs_override=self.model_cfg.get("loader_kwargs", {}) or {},
            )
            self._single_pca_basis = self.pca_bases.get(self.model_tag)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen extractor (or fused stack) on a batch."""
        if hasattr(self, "_components"):
            maps = [
                _spatial_feature_map(comp["extractor"].extract(images),
                                     comp["pca_basis"])
                for comp in self._components
            ]
            weights = [comp["weight"] for comp in self._components]
            target_h = max(m.shape[-2] for m in maps)
            target_w = max(m.shape[-1] for m in maps)
            return fuse_concat(maps, weights, target_grid=(target_h, target_w))
        return _spatial_feature_map(
            self.extractor.extract(images), self._single_pca_basis,
        )

    def _build_head_eager(self) -> None:
        """Determine the fused feature channel count via a dummy forward.

        The head must exist before ``configure_optimizers`` runs, so we
        push a single zero-image through the extractor stack at init
        time and use the resulting channel count.
        """
        device = next(iter(self._extractor_devices()), "cpu")
        size = int(self.model_cfg["transform"]["input_size"])
        dummy = torch.zeros(1, 3, size, size, device=device)
        with torch.no_grad():
            features = self._features(dummy)
        self.head = SegmentationHead(
            in_channels=features.shape[1],
            n_classes=self.n_classes,
            **{k: v for k, v in self.head_kwargs.items() if k != "kind"},
        ).to(features.device)

    def _extractor_devices(self):
        """Yield the underlying device of each loaded extractor."""
        if hasattr(self, "_components"):
            for comp in self._components:
                yield next(comp["extractor"].model.parameters()).device
        else:
            yield next(self.extractor.model.parameters()).device

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward: frozen feature extraction + trainable head."""
        with torch.no_grad():
            features = self._features(images)
        return self.head(features)

    def training_step(self, batch, batch_idx):  # noqa: D401
        images, masks, _ = batch
        logits = self(images)
        loss = self.loss(logits, masks)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: D401
        images, masks, _ = batch
        logits = self(images)
        loss = self.loss(logits, masks)
        self.val_metric.update(logits, masks)
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:  # noqa: D401
        metrics = self.val_metric.compute()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.val_metric.reset()

    def configure_optimizers(self):  # noqa: D401
        optim_cfg = self.training_cfg["optim"]
        opt = torch.optim.AdamW(
            self.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=optim_cfg["epochs"])
        return [opt], [sched]
