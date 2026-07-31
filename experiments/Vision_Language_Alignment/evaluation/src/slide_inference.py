"""GroupViT/TCL-style sliding-window inference.

Settings are taken from both projects' official mmseg configs -- GroupViT
originated them and TCL reuses them verbatim (`# Modified from GroupViT` in
its own config):

    img_scale=(2048, 448)   -- resize so the short side is 448 (aspect
                               preserved), long side capped at 2048
    crop_size=(448, 448)
    stride=(224, 224)       -- 50% overlap
    aggregation: average overlapping-window logits (mmseg's slide_inference)

This is the de facto standard essentially every training-free open-vocab
segmentation paper since has copied for direct comparability. See
``segmentation/configs/_base_/datasets/ade20k.py`` in both
github.com/NVlabs/GroupViT and github.com/kakaobrain/tcl for the source.

Public API
----------
    resize_shorter_side(img, target=448, max_long_side=2048) -> PIL.Image
    make_windows(image_chw, crop_size, stride) -> list[Window]
    slide_inference(forward_fn, image_chw, n_classes, ...) -> Tensor[K,H,W]
    CanvasAccumulator -- batches windows across MULTIPLE images into one
        forward_fn call (up to max_batch), for full GPU utilization instead
        of one tiny per-image or per-window call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image


def resize_shorter_side(img: Image.Image, target: int = 448, max_long_side: int = 2048) -> Image.Image:
    """mmcv.imrescale semantics for a (max_long_edge, max_short_edge) scale
    tuple: pick the smaller of the two implied scale factors so the result
    fits both constraints. For typical (non-extreme-aspect-ratio) photos
    this reduces to "short side = target".
    """
    w, h = img.size
    scale = min(max_long_side / max(w, h), target / min(w, h))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


@dataclass
class Window:
    crop: torch.Tensor          # (C, crop_size, crop_size), zero-padded if needed
    y1: int
    y2: int
    x1: int
    x2: int
    ch: int                     # valid (unpadded) height
    cw: int                     # valid (unpadded) width


def make_windows(image_chw: torch.Tensor, crop_size: int = 448, stride: int = 224) -> list[Window]:
    """mmseg's ``EncoderDecoder.slide_inference`` window-placement algorithm:
    each window is clipped to the image boundary, then shifted back so it's
    always exactly ``crop_size`` (never padded at the image's far edge --
    padding only happens when the image itself is smaller than ``crop_size``).
    """
    C, H, W = image_chw.shape
    h_grids = max(math.ceil((H - crop_size) / stride) + 1, 1)
    w_grids = max(math.ceil((W - crop_size) / stride) + 1, 1)

    windows = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_size, H)
            x2 = min(x1 + crop_size, W)
            y1 = max(y2 - crop_size, 0)
            x1 = max(x2 - crop_size, 0)

            crop = image_chw[:, y1:y2, x1:x2]
            ch, cw = crop.shape[-2:]
            if ch < crop_size or cw < crop_size:
                padded = torch.zeros(C, crop_size, crop_size, dtype=crop.dtype)
                padded[:, :ch, :cw] = crop
                crop = padded

            windows.append(Window(crop, y1, y2, x1, x2, ch, cw))
    return windows


@torch.no_grad()
def slide_inference(
    forward_fn,
    image_chw: torch.Tensor,
    n_classes: int,
    crop_size: int = 448,
    stride: int = 224,
) -> torch.Tensor:
    """Single-image convenience wrapper: batches all of one image's windows
    into one ``forward_fn`` call. For batching across MULTIPLE images (full
    GPU utilization when each image only has a few windows), use
    ``CanvasAccumulator`` instead -- that's what ``run_eval.py`` actually
    uses; this function exists for the unit tests / simple single-image use.

    Args:
        forward_fn: callable ``(N, C, crop_size, crop_size) -> (N, K, crop_size, crop_size)``.
        image_chw:  ``(C, H, W)`` normalized image tensor, already resized
                    via ``resize_shorter_side``.
        n_classes:  K.

    Returns:
        ``(K, H, W)`` averaged logit canvas at the input's resolution.
    """
    C, H, W = image_chw.shape
    windows = make_windows(image_chw, crop_size, stride)

    logit_sum = torch.zeros(n_classes, H, W)
    count = torch.zeros(1, H, W)

    batch = torch.stack([w.crop for w in windows], dim=0)
    logits_batch = forward_fn(batch)  # (N, K, crop_size, crop_size)

    for i, w in enumerate(windows):
        logit_sum[:, w.y1:w.y2, w.x1:w.x2] += logits_batch[i, :, :w.ch, :w.cw]
        count[:, w.y1:w.y2, w.x1:w.x2] += 1

    return logit_sum / count.clamp(min=1)


class CanvasAccumulator:
    """Batches windows from MULTIPLE images into one ``forward_fn`` call
    (up to ``max_batch`` windows at a time), scattering each window's
    result back into its own image's canvas as soon as the batch is
    forwarded. Call ``add_image(image_chw)`` for each image (returns a
    handle), ``flush()`` periodically or at the end, then ``finalize(handle)``
    once all of that image's windows have been scattered back.

    This is the actual GPU-utilization fix: batch-size-1 (or one-image-worth,
    which is often just 2-6 windows) calls leave most of an A40 idle: this
    accumulates windows across images up to ``max_batch`` so each forward
    call is as full as GPU memory allows.
    """

    def __init__(self, forward_fn, n_classes: int, crop_size: int = 448,
                 stride: int = 224, max_batch: int = 32):
        self.forward_fn = forward_fn
        self.n_classes = n_classes
        self.crop_size = crop_size
        self.stride = stride
        self.max_batch = max_batch
        self._pending_crops: list[torch.Tensor] = []
        self._pending_meta: list[tuple[int, Window]] = []  # (canvas_id, window)
        self._canvases: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # id -> (logit_sum, count)
        self._remaining_windows: dict[int, int] = {}  # id -> windows not yet scattered
        self._next_id = 0

    def add_image(self, image_chw: torch.Tensor) -> int:
        C, H, W = image_chw.shape
        windows = make_windows(image_chw, self.crop_size, self.stride)
        canvas_id = self._next_id
        self._next_id += 1
        self._canvases[canvas_id] = (torch.zeros(self.n_classes, H, W), torch.zeros(1, H, W))
        self._remaining_windows[canvas_id] = len(windows)
        for w in windows:
            self._pending_crops.append(w.crop)
            self._pending_meta.append((canvas_id, w))
        if len(self._pending_crops) >= self.max_batch:
            self._flush()
        return canvas_id

    def _flush(self):
        if not self._pending_crops:
            return
        batch = torch.stack(self._pending_crops, dim=0)
        logits_batch = self.forward_fn(batch)  # (N, K, crop, crop)
        for i, (canvas_id, w) in enumerate(self._pending_meta):
            logit_sum, count = self._canvases[canvas_id]
            logit_sum[:, w.y1:w.y2, w.x1:w.x2] += logits_batch[i, :, :w.ch, :w.cw]
            count[:, w.y1:w.y2, w.x1:w.x2] += 1
        self._pending_crops.clear()
        self._pending_meta.clear()

    def finalize(self, canvas_id: int) -> torch.Tensor:
        """Flushes any pending windows (forcing completion of this canvas)
        and returns the averaged ``(K, H, W)`` logit canvas, freeing it."""
        self._flush()
        logit_sum, count = self._canvases.pop(canvas_id)
        self._remaining_windows.pop(canvas_id, None)
        return logit_sum / count.clamp(min=1)
