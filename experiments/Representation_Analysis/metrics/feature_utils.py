"""Feature loading and matrix building for the metrics pipeline.

Per-image features are stored as ``(H*W, C)`` float16 ``.npy`` files
under ``<features_dir>/<layer>/<split>/<image_stem>.npy``.

Public surface:

    build_global_matrix(...)              (N, C) mean-pooled over patches.
    build_patch_matrix_with_indices(...)  (N*n_subsample, C) sampled patches.
    build_cd_combined(cd2, cd5, cd8)      L2-norm + concat 3 layers.
    build_diffusion_combined_pca(...)     PCA-reduce + L2-norm + concat.
        Fit once on global features (``reducers=None``), pass the returned
        dict to project patch features with the same basis.
    build_fused(dino, cd_combined)        0.5·L2(dino) + 0.5·L2(cd) concat.
    PCAReducer                            TruncatedSVD wrapper.

All heavy arrays are float32.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional


# ── Low-level loading ─────────────────────────────────────────────────────────

def load_feature(path: str) -> np.ndarray:
    """Load a single .npy feature file and return as float32 (H*W, C)."""
    arr = np.load(path, mmap_mode="r")
    return arr.astype(np.float32)


def list_image_stems(feat_dir: str, layer: str, split: str = "val") -> List[str]:
    """Return sorted list of image stems (no extension) available for a layer."""
    layer_dir = Path(feat_dir) / layer / split
    return sorted(p.stem for p in layer_dir.glob("*.npy"))


# ── Interpolation to 32×32 ────────────────────────────────────────────────────

def interpolate_to_32x32(feat: np.ndarray, target: int = 32) -> np.ndarray:
    """Bilinearly interpolate a spatial feature map to target×target.

    Args:
        feat: (H*W, C) float32 array — H*W must be a perfect square
        target: target spatial size (default 32)

    Returns:
        (target*target, C) float32 array
    """
    hw, C = feat.shape
    H = int(round(hw ** 0.5))
    if H * H != hw:
        raise ValueError(f"Feature has {hw} tokens which is not a perfect square")

    if H == target:
        return feat  # already the right size, no copy

    # Use torch for bilinear interpolation (already available in .venv)
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(feat).float()          # (H*W, C)
    x = x.reshape(H, H, C).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, H)
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    x = x.squeeze(0).permute(1, 2, 0).reshape(target * target, C)  # (T*T, C)
    return x.numpy()


# ── Matrix builders ───────────────────────────────────────────────────────────

def build_global_matrix(
    feat_dir: str,
    layer: str,
    image_stems: List[str],
    split: str = "val",
    target: int = 32,
) -> np.ndarray:
    """Build global (mean-pooled) feature matrix.

    Loads each image's feature, interpolates to target×target, then mean-pools
    all patches to yield one vector per image.

    Returns:
        (N, C) float32 array
    """
    layer_dir = Path(feat_dir) / layer / split
    rows = []
    for stem in image_stems:
        feat = load_feature(str(layer_dir / f"{stem}.npy"))   # (H*W, C)
        feat = interpolate_to_32x32(feat, target=target)       # (1024, C)
        rows.append(feat.mean(axis=0))                         # (C,)
    return np.stack(rows, axis=0)  # (N, C)


def build_patch_matrix(
    feat_dir: str,
    layer: str,
    image_stems: List[str],
    n_subsample: int = 100,
    rng: Optional[np.random.Generator] = None,
    split: str = "val",
    target: int = 32,
    patch_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build patch-level feature matrix with random subsampling.

    Loads each image's feature, interpolates to target×target, then samples
    n_subsample patch indices (same indices per image if patch_indices provided,
    otherwise samples fresh per image using rng).

    Args:
        feat_dir: base directory containing layer subdirs
        layer: layer name (e.g. "layer_last", "layer_2")
        image_stems: ordered list of image stems
        n_subsample: number of patches to sample per image
        rng: numpy random Generator for reproducibility
        split: dataset split subfolder
        target: spatial size after interpolation
        patch_indices: if provided, use these fixed indices (shape (n_subsample,))

    Returns:
        (N * n_subsample, C) float32 array
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_patches_total = target * target  # 1024

    layer_dir = Path(feat_dir) / layer / split
    rows = []
    for stem in image_stems:
        feat = load_feature(str(layer_dir / f"{stem}.npy"))    # (H*W, C)
        feat = interpolate_to_32x32(feat, target=target)        # (1024, C)
        if patch_indices is not None:
            idx = patch_indices
        else:
            idx = rng.choice(n_patches_total, size=n_subsample, replace=False)
        rows.append(feat[idx])                                   # (n_subsample, C)
    return np.concatenate(rows, axis=0)  # (N * n_subsample, C)


def make_patch_indices(
    n_images: int,
    n_patches_total: int = 1024,
    n_subsample: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Pre-compute per-image patch indices for consistent cross-layer sampling.

    Returns:
        (n_images, n_subsample) int array of patch indices
    """
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.choice(n_patches_total, size=n_subsample, replace=False)
        for _ in range(n_images)
    ], axis=0)


def build_patch_matrix_with_indices(
    feat_dir: str,
    layer: str,
    image_stems: List[str],
    patch_indices: np.ndarray,
    split: str = "val",
    target: int = 32,
) -> np.ndarray:
    """Build patch matrix using pre-computed per-image indices.

    Ensures all feature types use the exact same spatial locations.

    Args:
        patch_indices: (N, n_subsample) int array

    Returns:
        (N * n_subsample, C) float32 array
    """
    layer_dir = Path(feat_dir) / layer / split
    rows = []
    for i, stem in enumerate(image_stems):
        feat = load_feature(str(layer_dir / f"{stem}.npy"))
        feat = interpolate_to_32x32(feat, target=target)
        rows.append(feat[patch_indices[i]])
    return np.concatenate(rows, axis=0)


# ── Feature combination ───────────────────────────────────────────────────────

def _row_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def build_cd_combined(cd2: np.ndarray, cd5: np.ndarray,
                      cd8: np.ndarray) -> np.ndarray:
    """Raw mode: concat(norm(cd2), norm(cd5), norm(cd8)) → (N, C2+C5+C8)."""
    return np.concatenate([
        _row_normalize(cd2),
        _row_normalize(cd5),
        _row_normalize(cd8),
    ], axis=1)


class PCAReducer:
    """Thin wrapper around TruncatedSVD for per-layer PCA reduction.

    Fit once on global features (N=2000 >> pca_dim=512, so exactly the
    requested number of components is available), then reuse .transform()
    for patch features so both levels use the same projection axes.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.actual_n_components: int = 0
        self._svd = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if n_comp < self.n_components:
            print(f"    [PCAReducer] requested {self.n_components} components "
                  f"but N={X.shape[0]}, D={X.shape[1]} → using {n_comp}")
        self.actual_n_components = n_comp
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        out = self._svd.fit_transform(X)
        return out.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("PCAReducer must be fit before calling transform()")
        return self._svd.transform(X).astype(np.float32)


def build_diffusion_combined_pca(
    layer_a: np.ndarray,
    layer_b: np.ndarray,
    layer_c: np.ndarray,
    pca_dim: int,
    reducers: Optional[dict] = None,
    layer_keys: tuple[str, str, str] = ("a", "b", "c"),
):
    """PCA mode: reduce each diffusion layer to pca_dim, then concat normalised results.

    Backbone-agnostic — works for CD (layers 2/5/8) and SD-2.1 (layers
    picked by the experiment). DINOv3 is never touched by this function.

    Args:
        layer_a, layer_b, layer_c: (N, C_i) raw per-layer matrices.
        pca_dim: target dimension for each layer (exactly achieved when N >> pca_dim).
        reducers: dict keyed by ``layer_keys`` with fitted PCAReducer objects.
            Pass None to fit fresh PCAs; pass the returned dict when projecting
            patch-level features to reuse the same global basis.
        layer_keys: three keys used in the reducers dict; useful when callers
            want stable names (e.g. ``("cd2", "cd5", "cd8")`` or
            ``("sd2.1_a", "sd2.1_b", "sd2.1_c")``).

    Returns:
        combined: (N, 3 * actual_n_components) float32 array.
        reducers: dict of fitted PCAReducer objects — pass to subsequent calls.
    """
    ka, kb, kc = layer_keys
    fit_new = reducers is None
    if fit_new:
        reducers = {
            ka: PCAReducer(pca_dim),
            kb: PCAReducer(pca_dim),
            kc: PCAReducer(pca_dim),
        }

    a_r = reducers[ka].fit_transform(layer_a) if fit_new else reducers[ka].transform(layer_a)
    b_r = reducers[kb].fit_transform(layer_b) if fit_new else reducers[kb].transform(layer_b)
    c_r = reducers[kc].fit_transform(layer_c) if fit_new else reducers[kc].transform(layer_c)

    combined = np.concatenate([
        _row_normalize(a_r),
        _row_normalize(b_r),
        _row_normalize(c_r),
    ], axis=1)

    return combined, reducers


def build_cd_combined_pca(
    cd2: np.ndarray,
    cd5: np.ndarray,
    cd8: np.ndarray,
    pca_dim: int,
    reducers: Optional[dict] = None,
):
    """CleanDIFT-flavored convenience wrapper around ``build_diffusion_combined_pca``.

    Preserves the historical ``("cd2", "cd5", "cd8")`` reducer keys so existing
    saved reducer dicts continue to work.
    """
    return build_diffusion_combined_pca(
        cd2, cd5, cd8, pca_dim,
        reducers=reducers,
        layer_keys=("cd2", "cd5", "cd8"),
    )


def build_fused(dino: np.ndarray, cd_combined: np.ndarray) -> np.ndarray:
    """Fuse DINOv3 and CD combined: concat(0.5*norm(dino), 0.5*norm(cd_combined)).

    Works for both raw and pca cd_combined — dino is never reduced.

    Args:
        dino: (N, 1280) — always raw
        cd_combined: (N, C_cd) — raw (3200) or pca (1536) output

    Returns:
        (N, 1280 + C_cd) float32 array
    """
    return np.concatenate([
        0.5 * _row_normalize(dino),
        0.5 * _row_normalize(cd_combined),
    ], axis=1)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml. Defaults to config.yaml next to this file."""
    import yaml
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)
