"""Representation-similarity metrics.

All metrics return a scalar in [0, 1] and accept ``X``, ``Y`` as ``(N, D)``
numpy arrays (float32 or float64). The two feature widths ``D1``, ``D2``
may differ.

    linear_cka(X, Y)              Kornblith et al. 2019.
    svcca(X, Y, var_threshold)    Raghu et al. 2017 (SVD pre-conditioning).
    pwcca(X, Y, var_threshold)    Morcos et al. 2018.
"""

import numpy as np

from .third_party import cca_core
from .third_party.pwcca import compute_pwcca


# ── Linear CKA ───────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (feature-space formula). Handles D1 ≠ D2.

    CKA(X, Y) = ||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)

    Args:
        X: (N, D1) array
        Y: (N, D2) array

    Returns:
        CKA similarity in [0, 1]
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    num = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    if denom == 0:
        return 0.0
    return float(num / denom)


# ── SVD pre-conditioning helper ───────────────────────────────────────────────

def _svd_reduce(X: np.ndarray, var_threshold: float):
    """Center X and project onto top-k left singular vectors explaining
    var_threshold of variance. Returns (X_red, k) where X_red has shape (k, N).

    cca_core requires (D, N) format and D < N — this function returns that.
    """
    Xc = X - X.mean(axis=0)                  # (N, D) centered
    # Randomized SVD via numpy; for large N use economy SVD
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # U: (N, min(N,D))
    sv_sq = s ** 2
    cumvar = np.cumsum(sv_sq) / (np.sum(sv_sq) + 1e-12)
    k = int(np.searchsorted(cumvar, var_threshold)) + 1
    k = min(k, len(s))
    # Reduced representation: project data onto top-k right singular vectors
    # X_red = (Vt[:k] @ Xc.T) has shape (k, N) — this is what cca_core expects
    X_red = Vt[:k] @ Xc.T          # (k, N)
    return X_red, k


# ── SVCCA ─────────────────────────────────────────────────────────────────────

def svcca(X: np.ndarray, Y: np.ndarray, var_threshold: float = 0.99) -> float:
    """SVCCA — Singular Vector CCA (Raghu et al. 2017).

    Steps:
      1. SVD-reduce X and Y independently to their top-k directions
         explaining var_threshold of variance (makes CCA well-conditioned
         even when D > N).
      2. Run CCA on the reduced (k, N) representations.
      3. Return the mean canonical correlation (trimmed at var_threshold).

    Args:
        X: (N, D1) array
        Y: (N, D2) array
        var_threshold: fraction of variance to retain in SVD step

    Returns:
        SVCCA similarity in [0, 1]
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X_red, _ = _svd_reduce(X, var_threshold)   # (k1, N)
    Y_red, _ = _svd_reduce(Y, var_threshold)   # (k2, N)

    # cca_core requires shape[0] < shape[1] (i.e. k < N) — guaranteed because
    # k ≤ min(N, D) and for N≥50 this always holds after SVD reduction.
    result = cca_core.get_cca_similarity(
        X_red, Y_red,
        epsilon=1e-6,
        threshold=var_threshold,
        compute_coefs=False,
        compute_dirns=False,
        verbose=False,
    )
    mean_val = result["mean"]
    # result["mean"] is a tuple (mean1, mean2); take the first
    if isinstance(mean_val, (tuple, list)):
        return float(mean_val[0])
    return float(mean_val)


# ── PWCCA ─────────────────────────────────────────────────────────────────────

def pwcca(X: np.ndarray, Y: np.ndarray, var_threshold: float = 0.99) -> float:
    """PWCCA — Projection Weighted CCA (Morcos et al. 2018).

    Same SVD pre-conditioning as SVCCA, then weights each canonical
    correlation ρ_i by α_i = ||P.T @ acts||_row_sum (derived from QR
    decomposition of the CCA directions), returning Σ α_i·ρ_i.

    Args:
        X: (N, D1) array
        Y: (N, D2) array
        var_threshold: fraction of variance to retain in SVD step

    Returns:
        PWCCA similarity in [0, 1]
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X_red, _ = _svd_reduce(X, var_threshold)   # (k1, N)
    Y_red, _ = _svd_reduce(Y, var_threshold)   # (k2, N)

    weighted_mean, _, _ = compute_pwcca(X_red, Y_red, epsilon=1e-6)
    return float(weighted_mean)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def all_metrics(X: np.ndarray, Y: np.ndarray,
                var_threshold: float = 0.99) -> dict:
    """Compute all three metrics and return as a dict.

    Args:
        X: (N, D1) array
        Y: (N, D2) array
        var_threshold: SVD variance threshold for SVCCA/PWCCA

    Returns:
        {"linear_cka": float, "svcca": float, "pwcca": float}
    """
    return {
        "linear_cka": linear_cka(X, Y),
        "svcca":       svcca(X, Y, var_threshold),
        "pwcca":       pwcca(X, Y, var_threshold),
    }
