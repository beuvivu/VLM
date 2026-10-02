"""Co-occurrence (đồng xuất hiện) analysis.

Under H0 a specific pair {i, j} appears together in a draw with probability
q = k(k−1)/(n(n−1)), so O_ij ~ Binomial(D, q). With n(n−1)/2 pairs (990 for 6/45,
1,485 for 6/55) dozens of pairs will look "hot" by chance alone; this module
reports BH-adjusted q-values and a Monte Carlo test of global over-dispersion so
that chance and signal are separated.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.analytics.randomness import TestResult
from vietlott_engine.analytics.stats import benjamini_hochberg, monte_carlo_p, simulate_incidence
from vietlott_engine.core.history import DrawHistory, decay_weights


class PairStat(BaseModel):
    a: int
    b: int
    observed: int
    expected: float
    lift: float
    z: float
    p_value: float
    q_value: float


class CooccurrenceReport(BaseModel):
    game: str
    draws: int
    pair_probability: float
    top_pairs: list[PairStat]
    bottom_pairs: list[PairStat]
    significant_pairs: int
    dispersion_test: TestResult


def cooccurrence_matrix(h: DrawHistory, decay: float = 1.0) -> np.ndarray:
    """(n, n) matrix of (optionally exponentially decayed) joint appearance counts.

    The diagonal holds the per-number counts.
    """
    inc = h.incidence.astype(np.float64)
    if decay < 1.0:
        inc = inc * decay_weights(len(h), decay)[:, None]
        return inc.T @ h.incidence.astype(np.float64)
    return inc.T @ inc


def _dispersion(c: np.ndarray, expected: float) -> float:
    iu = np.triu_indices(c.shape[0], 1)
    return float(np.sum((c[iu] - expected) ** 2) / expected)


def cooccurrence_report(h: DrawHistory, top: int = 15, mc_sims: int = 300, seed: int | None = None, fdr: float = 0.05) -> CooccurrenceReport:
    h.require(30)
    n, k, d = h.n, h.k, len(h)
    q = k * (k - 1) / (n * (n - 1))
    expected = d * q
    c = cooccurrence_matrix(h)
    iu = np.triu_indices(n, 1)
    obs = c[iu].astype(int)
    sd = np.sqrt(d * q * (1 - q))
    z = (obs - expected) / sd
    # exact two-sided binomial p-values (doubling the smaller tail)
    p_hi = stats.binom.sf(obs - 1, d, q)
    p_lo = stats.binom.cdf(obs, d, q)
    p = np.minimum(1.0, 2 * np.minimum(p_hi, p_lo))
    qv = benjamini_hochberg(p)

    def row(idx: int) -> PairStat:
        return PairStat(
            a=int(iu[0][idx] + 1),
            b=int(iu[1][idx] + 1),
            observed=int(obs[idx]),
            expected=float(expected),
            lift=float(obs[idx] / expected),
            z=float(z[idx]),
            p_value=float(p[idx]),
            q_value=float(qv[idx]),
        )

    order = np.argsort(-z)
    rng = np.random.default_rng(seed)
    obs_disp = _dispersion(c, expected)
    null = np.empty(mc_sims)
    for s in range(mc_sims):
        inc = simulate_incidence(d, n, k, rng).astype(np.float64)
        null[s] = _dispersion(inc.T @ inc, expected)
    disp = TestResult(
        name="pair_overdispersion",
        statistic=obs_disp,
        p_value=monte_carlo_p(obs_disp, null, "greater"),
        method=f"Σ(O−E)²/E over all pairs vs {mc_sims} simulated histories",
        detail={"null_mean": float(null.mean()), "null_sd": float(null.std())},
    )
    return CooccurrenceReport(
        game=h.spec.code.value,
        draws=d,
        pair_probability=q,
        top_pairs=[row(i) for i in order[:top]],
        bottom_pairs=[row(i) for i in order[::-1][:top]],
        significant_pairs=int(np.sum(qv < fdr)),
        dispersion_test=disp,
    )
