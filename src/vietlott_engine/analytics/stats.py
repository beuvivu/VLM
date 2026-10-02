"""Shared statistical primitives: exact null distributions for k-of-n draws,
pooled goodness-of-fit, multiple-testing control and Monte Carlo null simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import stats


# --------------------------------------------------------------------- nulls
def hypergeom_pmf(total: int, successes: int, draws: int) -> np.ndarray:
    """P(X = x), x = 0..draws, for X ~ Hypergeometric(total, successes, draws)."""
    x = np.arange(draws + 1)
    return stats.hypergeom.pmf(x, total, successes, draws)


def subset_sum_distribution(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution of the sum of a uniformly random k-subset of {1..n}.

    Dynamic programming over numbers: ways[j, s] = #j-subsets with sum s.
    """
    max_sum = sum(range(n - k + 1, n + 1))
    ways = np.zeros((k + 1, max_sum + 1), dtype=object)  # exact big integers
    ways[0, 0] = 1
    for x in range(1, n + 1):
        for j in range(min(k, x), 0, -1):
            ways[j, x:] = ways[j, x:] + ways[j - 1, : max_sum + 1 - x]
    support = np.arange(max_sum + 1)
    probs = np.array([int(w) for w in ways[k]], dtype=np.float64) / comb(n, k)
    mask = probs > 0
    return support[mask], probs[mask]


def adjacency_distribution(n: int, k: int) -> np.ndarray:
    """P(r consecutive pairs) in a random k-subset of {1..n}: C(k-1,r)·C(n-k+1,k-r)/C(n,k)."""
    return np.array([comb(k - 1, r) * comb(n - k + 1, k - r) / comb(n, k) for r in range(k)])


# ---------------------------------------------------------- goodness of fit
@dataclass(frozen=True)
class GofResult:
    g_statistic: float
    chi2_statistic: float
    df: int
    p_value: float  # from the G statistic
    observed: list[int]
    expected: list[float]
    labels: list[str]


def pooled_gof(observed: np.ndarray, probs: np.ndarray, labels: list[str] | None = None, min_expected: float = 5.0) -> GofResult:
    """G-test (log-likelihood ratio) of observed category counts against ``probs``.

    Adjacent categories are merged from both tails until every expected count is at
    least ``min_expected`` — the standard remedy for sparse tails (e.g. sums).
    """
    observed = np.asarray(observed, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    total = observed.sum()
    labels = labels or [str(i) for i in range(len(observed))]
    groups: list[list[int]] = [[i] for i in range(len(observed))]

    def exp_of(g: list[int]) -> float:
        return float(total * probs[g].sum())

    # merge left tail
    while len(groups) > 2 and exp_of(groups[0]) < min_expected:
        groups[1] = groups[0] + groups[1]
        groups.pop(0)
    # merge right tail
    while len(groups) > 2 and exp_of(groups[-1]) < min_expected:
        groups[-2] = groups[-2] + groups[-1]
        groups.pop()
    # merge any interior sparse bins into their right neighbour
    i = 0
    while i < len(groups) - 1:
        if exp_of(groups[i]) < min_expected:
            groups[i + 1] = groups[i] + groups[i + 1]
            groups.pop(i)
        else:
            i += 1

    o = np.array([observed[g].sum() for g in groups])
    e = np.array([exp_of(g) for g in groups])
    nz = o > 0
    g_stat = float(2.0 * np.sum(o[nz] * np.log(o[nz] / e[nz])))
    chi2 = float(np.sum((o - e) ** 2 / e))
    df = max(1, len(groups) - 1)
    return GofResult(
        g_statistic=g_stat,
        chi2_statistic=chi2,
        df=df,
        p_value=float(stats.chi2.sf(g_stat, df)),
        observed=[int(x) for x in o],
        expected=[float(x) for x in e],
        labels=[labels[g[0]] if len(g) == 1 else f"{labels[g[0]]}..{labels[g[-1]]}" for g in groups],
    )


# ----------------------------------------------------- multiple testing etc.
def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini–Hochberg adjusted q-values (step-up, monotone)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def wilson_interval(successes: float, trials: float, level: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def monte_carlo_p(observed: float, null_samples: np.ndarray, tail: str = "greater") -> float:
    """Permutation-style p-value with the +1 correction (never exactly zero)."""
    null_samples = np.asarray(null_samples)
    b = null_samples.size
    if tail == "greater":
        hits = np.sum(null_samples >= observed)
    elif tail == "less":
        hits = np.sum(null_samples <= observed)
    else:
        centre = np.mean(null_samples)
        hits = np.sum(np.abs(null_samples - centre) >= abs(observed - centre))
    return float((hits + 1) / (b + 1))


def simulate_incidence(draws: int, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """One simulated history under H0: ``draws`` independent uniform k-subsets, as (D, n) bool."""
    idx = np.argpartition(rng.random((draws, n)), k, axis=1)[:, :k]
    inc = np.zeros((draws, n), dtype=bool)
    inc[np.arange(draws)[:, None], idx] = True
    return inc


def simulate_counts(draws: int, n: int, k: int, sims: int, rng: np.random.Generator) -> np.ndarray:
    """(sims, n) per-number counts under H0."""
    return np.stack([simulate_incidence(draws, n, k, rng).sum(axis=0) for _ in range(sims)])
