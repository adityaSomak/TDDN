"""Disjoint partitioning of a captioned pool into N published versions.

One function, not a class hierarchy -- uniform-disjoint is the only sampler needed
today. A future selection method (MATES/Group-MATES/PreSelect) would plug in as a
different pool ordering fed into the same slice step, not a reason to build an
abstraction now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def partition_disjoint(pool: pd.DataFrame, num_versions: int, n_per_version: int,
                        seed: int) -> list[pd.DataFrame]:
    """One seeded shuffle of `pool`, then sliced into `num_versions` blocks of
    exactly `n_per_version` rows each. `pool` must have at least
    num_versions * n_per_version rows.
    """
    needed = num_versions * n_per_version
    if len(pool) < needed:
        raise ValueError(f"pool has {len(pool)} rows, need >= {needed} for "
                          f"{num_versions} versions of {n_per_version}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))[:needed]
    shuffled = pool.iloc[order].reset_index(drop=True)

    return [shuffled.iloc[i * n_per_version:(i + 1) * n_per_version].reset_index(drop=True)
            for i in range(num_versions)]
