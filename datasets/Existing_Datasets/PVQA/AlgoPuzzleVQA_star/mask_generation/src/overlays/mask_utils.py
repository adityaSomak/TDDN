"""Geometry helpers for outline and TDDN mask generation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def parse_grid(text_repr: str) -> list[list[str]]:
    """Parse a semicolon-separated grid string into a 2D list of cells."""
    rows = [r for r in text_repr.strip().splitlines() if r.strip()]
    return [[c.strip() for c in r.split(";")] for r in rows]


def dilated_outline(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Return the boundary band of `thickness` pixels around the mask."""
    if not mask.any():
        return np.zeros_like(mask)
    return (
        binary_dilation(mask, iterations=thickness)
        & ~binary_erosion(mask, iterations=thickness)
    )


def cell_to_patch(r: int, c: int, R: int, C: int, patch_grid: int) -> int:
    """Flat patch index of the centre patch of cell (r, c) on an R x C board."""
    pr = int((r + 0.5) * patch_grid / R)
    pc = int((c + 0.5) * patch_grid / C)
    return min(pr, patch_grid - 1) * patch_grid + min(pc, patch_grid - 1)


def wall_mask_from_grey(rgb: np.ndarray, threshold: int = 80) -> np.ndarray:
    """Binary mask of pixels darker than `threshold` (used for maze walls)."""
    return rgb.mean(axis=-1) < threshold
