"""Multiple-testing control.

* Holm (FWER, any dependence)
* Benjamini–Hochberg (FDR, independence / PRDS)
* Benjamini–Yekutieli (FDR, *arbitrary* dependence — the safe choice for a battery
  of tests computed on the same draws)
* Westfall–Young max-T (FWER, exact under the joint null distribution estimated by
  simulation — as powerful as possible while respecting the dependence between
  per-number statistics, which are negatively correlated because Σ counts = kD)
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.analytics.stats import benjamini_hochberg, simulate_counts
from vietlott_engine.core.history import DrawHistory


def holm(p_values: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def benjamini_yekutieli(p_values: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return np.clip(benjamini_hochberg(p) * c_m, 0, 1)


def westfall_young_maxt(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Step-down max-T adjusted p-values (Westfall & Young 1993, Algorithm 4.1).

    ``observed``: (m,) statistics where larger = more extreme (use |z| for two-sided).
    ``null``: (B, m) statistics simulated under the joint null.
    """
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    m = observed.size
    order = np.argsort(-observed)  # most extreme first
    # successive maxima over the hypotheses not yet rejected (from the least extreme upwards)
    tail_max = np.maximum.accumulate(null[:, order[::-1]], axis=1)[:, ::-1]  # (B, m)
    raw = (1 + np.sum(tail_max >= observed[order][None, :], axis=0)) / (null.shape[0] + 1)
    adj_sorted = np.maximum.accumulate(raw)  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(adj_sorted, 0, 1)
    return out


class NumberDeviation(BaseModel):
    number: int
    count: int
    expected: float
    z: float
    p_raw: float
    p_holm: float
    q_bh: float
    q_by: float
    p_westfall_young: float


class PerNumberReport(BaseModel):
    game: str
    draws: int
    simulations: int
    numbers: list[NumberDeviation]
    any_significant_fwer: bool
    min_westfall_young_p: float


def per_number_deviations(h: DrawHistory, sims: int = 2000, seed: int | None = None) -> PerNumberReport:
    """Which individual numbers (if any) deviate from k/n, with every correction side by side."""
    n, k, d = h.n, h.k, len(h)
    p0 = k / n
    counts = h.counts()
    exp = d * p0
    sd = np.sqrt(d * p0 * (1 - p0))
    z = (counts - exp) / sd
    p_hi = stats.binom.sf(counts - 1, d, p0)
    p_lo = stats.binom.cdf(counts, d, p0)
    p_raw = np.minimum(1.0, 2 * np.minimum(p_hi, p_lo))
    null_counts = simulate_counts(d, n, k, sims, np.random.default_rng(seed))
    null_abs_z = np.abs((null_counts - exp) / sd)
    p_wy = westfall_young_maxt(np.abs(z), null_abs_z)
    ph, qbh, qby = holm(p_raw), benjamini_hochberg(p_raw), benjamini_yekutieli(p_raw)
    rows = [
        NumberDeviation(
            number=i + 1,
            count=int(counts[i]),
            expected=float(exp),
            z=float(z[i]),
            p_raw=float(p_raw[i]),
            p_holm=float(ph[i]),
            q_bh=float(qbh[i]),
            q_by=float(qby[i]),
            p_westfall_young=float(p_wy[i]),
        )
        for i in np.argsort(-np.abs(z))
    ]
    return PerNumberReport(
        game=h.spec.code.value,
        draws=d,
        simulations=sims,
        numbers=rows,
        any_significant_fwer=bool(p_wy.min() < 0.05),
        min_westfall_young_p=float(p_wy.min()),
    )
