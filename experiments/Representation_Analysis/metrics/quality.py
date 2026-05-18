"""Intrinsic representation-quality metrics.

Both metrics operate on ``(N, D)`` numpy arrays (float32 or float64).

    uniformity(X, n_subsample=10000)   Wang & Isola 2020.
    effective_rank(X)                  Roy & Vetterli 2007.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_DEFAULT_SUBSAMPLE = 10000
_DEFAULT_SEED = 42


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Adds ``1e-8`` to the denominator for safety."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)


def uniformity(
    X: np.ndarray,
    n_subsample: Optional[int] = _DEFAULT_SUBSAMPLE,
    seed: int = _DEFAULT_SEED,
) -> float:
    """Wang & Isola 2020 uniformity (higher = more uniform = better).

    Args:
        X: ``(N, D)`` representation matrix; will be L2-normalized internally.
        n_subsample: if not None, sample this many rows before computing
            the all-pairs term. Default 10000 keeps patch-level runs cheap.
            Pass ``None`` to use all rows (global recipe).
        seed: rng seed for the subsample.

    Returns:
        ``log( mean_{i≠j} exp(-2 ||x̂_i - x̂_j||²) )`` as a Python float.
    """
    if n_subsample is not None and n_subsample < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=n_subsample, replace=False)
        X = X[idx]

    Xn = l2_normalize(X.astype(np.float64))
    sim = Xn @ Xn.T
    sq_dist = 2.0 - 2.0 * sim
    np.fill_diagonal(sq_dist, np.nan)
    exps = np.exp(-2.0 * sq_dist)
    return float(np.log(np.nanmean(exps)))


def effective_rank(X: np.ndarray) -> float:
    """Roy & Vetterli 2007 effective rank via the centered covariance.

    ``eff_rank = exp(H(p))`` where ``p_i = λ_i / Σλ_j`` and ``λ`` are the
    eigenvalues of ``X_c.T @ X_c / N``. Zero eigenvalues (null space)
    contribute 0 to entropy. Returns 1.0 if the covariance is degenerate.

    Cost: O(N·D²) matmul + O(D³) eigendecomposition.
    """
    Xc = X.astype(np.float64) - X.mean(axis=0)
    N = Xc.shape[0]
    C = (Xc.T @ Xc) / N
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return 1.0
    p = eigvals / total
    p_nz = p[p > 1e-12]
    H = -float(np.sum(p_nz * np.log(p_nz)))
    return float(np.exp(H))


__all__ = ["l2_normalize", "uniformity", "effective_rank"]
