"""Resampling inference for dependent data and data-snooping-robust strategy selection.

* Stationary bootstrap (Politis & Romano 1994): geometric block lengths, preserves
  serial dependence.
* Hansen (2005) Superior Predictive Ability test and White (2000) Reality Check:
  when M strategies are tried, "the best one beat the benchmark" is a statement about
  a *maximum* and must be tested as such. H0: no strategy has positive expected
  performance relative to the benchmark. Three SPA p-values (lower / consistent /
  upper) bracket the answer; the consistent one is the recommended test.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


def stationary_bootstrap_indices(t: int, mean_block: float, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """Politis–Romano resampling indices, shape (t,) or (size, t) — vectorised.

    A new block starts with probability 1/mean_block; within a block indices advance
    by one (circularly).
    """
    b = 1 if size is None else size
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.random((b, t)) < p
    starts[:, 0] = True
    jumps = rng.integers(0, t, (b, t))
    pos = np.arange(t)[None, :]
    last_start = np.maximum.accumulate(np.where(starts, pos, 0), axis=1)
    idx = (np.take_along_axis(jumps, last_start, axis=1) + (pos - last_start)) % t
    return idx[0] if size is None else idx


class SPAResult(BaseModel):
    strategies: list[str]
    best_strategy: str
    best_mean_differential: float
    statistic: float
    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    reality_check_p_value: float
    bootstrap_samples: int
    mean_block: float


def spa_test(differentials: np.ndarray, names: list[str], bootstrap: int = 2000, mean_block: float = 10.0, seed: int | None = None) -> SPAResult:
    """``differentials``: (T, M) performance of each strategy minus the benchmark, per period."""
    d = np.asarray(differentials, dtype=np.float64)
    t, m = d.shape
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    boot_means = np.empty((bootstrap, m))
    chunk = max(1, 2_000_000 // max(t, 1))
    for b0 in range(0, bootstrap, chunk):
        idx = stationary_bootstrap_indices(t, mean_block, rng, min(chunk, bootstrap - b0))
        boot_means[b0 : b0 + idx.shape[0]] = d[idx].mean(axis=1)
    omega = np.sqrt(t) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, 1e-12)
    stat = float(max(0.0, np.max(np.sqrt(t) * dbar / omega)))
    thresh = -np.sqrt(2 * np.log(np.log(t)))
    g_l = np.maximum(dbar, 0.0)
    g_c = dbar * (np.sqrt(t) * dbar / omega >= thresh)
    g_u = dbar
    centred = boot_means - dbar  # d̄* − d̄

    def p_for(g: np.ndarray) -> float:
        z = np.sqrt(t) * (centred + dbar - g) / omega
        tb = np.maximum(0.0, z.max(axis=1))
        return float(np.mean(tb >= stat))

    rc_stat = float(np.max(np.sqrt(t) * dbar))
    rc_boot = np.max(np.sqrt(t) * centred, axis=1)
    best = int(np.argmax(dbar / omega))
    return SPAResult(
        strategies=names,
        best_strategy=names[best],
        best_mean_differential=float(dbar[best]),
        statistic=stat,
        p_value_consistent=p_for(g_c),
        p_value_lower=p_for(g_l),
        p_value_upper=p_for(g_u),
        reality_check_p_value=float(np.mean(rc_boot >= rc_stat)),
        bootstrap_samples=bootstrap,
        mean_block=mean_block,
    )
