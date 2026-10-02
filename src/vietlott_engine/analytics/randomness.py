"""Randomness test battery for k-of-n lottery draws.

H0 throughout: draws are independent, and each draw is a uniformly random k-subset
of {1..n}. Every test uses the *exact* null of that design (hypergeometric,
subset-sum, adjacency) rather than textbook approximations that assume sampling
with replacement.

Note on the frequency χ² test. Within one draw the k numbers are sampled without
replacement, so per-number counts are negatively correlated:
Cov = c·(I − J/n) with c = D·k(n−k)/(n(n−1)). The naive Σ(O−E)²/E is therefore
*not* χ²(n−1); the correctly scaled statistic is Σ(O−E)²·n(n−1)/(D·k(n−k)) =
naive·(n−1)/(n−k). The test suite verifies that its p-values are uniform under H0.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from vietlott_engine.analytics.stats import (
    adjacency_distribution,
    benjamini_hochberg,
    hypergeom_pmf,
    monte_carlo_p,
    pooled_gof,
    simulate_counts,
    subset_sum_distribution,
)
from vietlott_engine.core.history import DrawHistory


class TestResult(BaseModel):
    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    df: float | None = None
    p_value: float
    q_value: float | None = None
    method: str
    detail: dict = Field(default_factory=dict)


class RandomnessReport(BaseModel):
    game: str
    draws: int
    first_date: str | None
    last_date: str | None
    tests: list[TestResult]
    fdr_level: float
    rejected: list[str]
    verdict: str


# ------------------------------------------------------------------ tests
def frequency_chi_square(h: DrawHistory, mc_sims: int = 0, rng: np.random.Generator | None = None) -> TestResult:
    n, k, d = h.n, h.k, len(h)
    obs = h.counts()
    exp = d * k / n
    ss = float(np.sum((obs - exp) ** 2))
    naive = ss / exp
    corrected = ss * n * (n - 1) / (d * k * (n - k))
    p = float(stats.chi2.sf(corrected, n - 1))
    detail = {
        "naive_statistic": naive,
        "naive_p_value_miscalibrated": float(stats.chi2.sf(naive, n - 1)),
        "expected_per_number": exp,
        "min_count": int(obs.min()),
        "max_count": int(obs.max()),
        "most_frequent": [int(i + 1) for i in np.argsort(-obs)[:5]],
        "least_frequent": [int(i + 1) for i in np.argsort(obs)[:5]],
    }
    if mc_sims:
        rng = rng or np.random.default_rng()
        sim = simulate_counts(d, n, k, mc_sims, rng)
        sim_stat = ((sim - exp) ** 2).sum(axis=1) * n * (n - 1) / (d * k * (n - k))
        detail["monte_carlo_p_value"] = monte_carlo_p(corrected, sim_stat, "greater")
    return TestResult(
        name="frequency_chi_square",
        statistic=corrected,
        df=n - 1,
        p_value=p,
        method="Chi-square GOF on per-number counts with without-replacement covariance correction",
        detail=detail,
    )


def entropy_test(h: DrawHistory, mc_sims: int = 1000, rng: np.random.Generator | None = None) -> TestResult:
    """Shannon entropy of the empirical number distribution vs its exact null (Monte Carlo).

    A biased machine concentrates mass → *lower* entropy, so the test is one-sided.
    """
    rng = rng or np.random.default_rng()
    n, k, d = h.n, h.k, len(h)

    def entropy_bits(counts: np.ndarray) -> np.ndarray:
        p = counts / counts.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)

    obs = h.counts()
    h_obs = float(entropy_bits(obs))
    sim = simulate_counts(d, n, k, mc_sims, rng)
    h_sim = entropy_bits(sim.astype(np.float64))
    h_max = float(np.log2(n))
    return TestResult(
        name="shannon_entropy",
        statistic=h_obs,
        p_value=monte_carlo_p(h_obs, h_sim, "less"),
        method=f"Monte Carlo ({mc_sims} simulated histories), lower tail",
        detail={
            "entropy_bits": h_obs,
            "max_entropy_bits": h_max,
            "efficiency": h_obs / h_max,
            "kl_from_uniform_bits": h_max - h_obs,
            "null_mean_bits": float(h_sim.mean()),
            "null_sd_bits": float(h_sim.std()),
            "miller_madow_entropy_bits": h_obs + (np.count_nonzero(obs) - 1) / (2 * obs.sum() * np.log(2)),
        },
    )


def _gof_test(name: str, method: str, observed: np.ndarray, probs: np.ndarray, labels: list[str]) -> TestResult:
    g = pooled_gof(observed, probs, labels)
    return TestResult(
        name=name,
        statistic=g.g_statistic,
        df=g.df,
        p_value=g.p_value,
        method=method,
        detail={"bins": g.labels, "observed": g.observed, "expected": [round(x, 2) for x in g.expected], "chi2": g.chi2_statistic},
    )


def odd_even_test(h: DrawHistory) -> TestResult:
    n, k = h.n, h.k
    odd = (h.numbers % 2 == 1).sum(axis=1)
    n_odd = (n + 1) // 2
    obs = np.bincount(odd, minlength=k + 1)
    return _gof_test("odd_count", "G-test vs Hypergeometric(n, #odd, k)", obs, hypergeom_pmf(n, n_odd, k), [str(i) for i in range(k + 1)])


def low_high_test(h: DrawHistory) -> TestResult:
    n, k = h.n, h.k
    low = (h.numbers <= n // 2).sum(axis=1)
    obs = np.bincount(low, minlength=k + 1)
    return _gof_test("low_count", "G-test vs Hypergeometric(n, n//2, k)", obs, hypergeom_pmf(n, n // 2, k), [str(i) for i in range(k + 1)])


def sum_test(h: DrawHistory) -> TestResult:
    support, probs = subset_sum_distribution(h.n, h.k)
    sums = h.numbers.sum(axis=1).astype(int)
    obs = np.array([np.sum(sums == s) for s in support])
    t = _gof_test("sum_distribution", "G-test vs exact subset-sum distribution (pooled tails)", obs, probs, [str(s) for s in support])
    t.detail.update({"observed_mean": float(sums.mean()), "expected_mean": float((support * probs).sum())})
    t.detail.pop("observed")
    t.detail.pop("expected")
    t.detail.pop("bins")
    return t


def adjacency_test(h: DrawHistory) -> TestResult:
    adj = (np.diff(h.numbers, axis=1) == 1).sum(axis=1)
    obs = np.bincount(adj, minlength=h.k)[: h.k]
    return _gof_test("consecutive_pairs", "G-test vs C(k-1,r)·C(n-k+1,k-r)/C(n,k)", obs, adjacency_distribution(h.n, h.k), [str(r) for r in range(h.k)])


def repeat_test(h: DrawHistory) -> TestResult:
    """Numbers repeated from the previous draw ~ Hypergeometric(n, k, k) under independence."""
    rep = (h.incidence[1:] & h.incidence[:-1]).sum(axis=1)
    obs = np.bincount(rep, minlength=h.k + 1)
    return _gof_test("repeat_from_previous", "G-test vs Hypergeometric(n, k, k) — serial independence", obs, hypergeom_pmf(h.n, h.k, h.k), [str(i) for i in range(h.k + 1)])


def runs_test(series: np.ndarray, name: str = "runs_sum") -> TestResult:
    """Wald–Wolfowitz runs test above/below the median (ties dropped)."""
    med = np.median(series)
    s = series[series != med] > med
    n1, n2 = int(s.sum()), int((~s).sum())
    runs = 1 + int(np.sum(s[1:] != s[:-1])) if s.size else 0
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var)
    return TestResult(name=name, statistic=float(z), p_value=float(2 * stats.norm.sf(abs(z))), method="Wald–Wolfowitz runs test (normal approximation)", detail={"runs": runs, "expected_runs": mu})


def ljung_box(series: np.ndarray, lags: int = 10, name: str = "ljung_box_sum") -> TestResult:
    x = np.asarray(series, dtype=np.float64) - np.mean(series)
    n = x.size
    denom = np.sum(x * x)
    acf = np.array([np.sum(x[j:] * x[: n - j]) / denom for j in range(1, lags + 1)])
    q = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1))))
    return TestResult(name=name, statistic=q, df=lags, p_value=float(stats.chi2.sf(q, lags)), method=f"Ljung–Box Q({lags})", detail={"acf": [round(a, 4) for a in acf]})


# ----------------------------------------------------------------- report
def randomness_report(h: DrawHistory, mc_sims: int = 1000, seed: int | None = None, fdr: float = 0.05) -> RandomnessReport:
    """Run the full battery and control the false discovery rate across tests."""
    h.require(30)
    rng = np.random.default_rng(seed)
    sums = h.numbers.sum(axis=1).astype(np.float64)
    tests = [
        frequency_chi_square(h, mc_sims=mc_sims, rng=rng),
        entropy_test(h, mc_sims=mc_sims, rng=rng),
        odd_even_test(h),
        low_high_test(h),
        sum_test(h),
        adjacency_test(h),
        repeat_test(h),
        runs_test(sums),
        ljung_box(sums),
    ]
    q = benjamini_hochberg([t.p_value for t in tests])
    for t, qv in zip(tests, q):
        t.q_value = float(qv)
    rejected = [t.name for t in tests if t.q_value is not None and t.q_value < fdr]
    verdict = (
        f"No evidence against H0 (independent uniform draws) at FDR {fdr:.0%} across {len(tests)} tests."
        if not rejected
        else f"H0 rejected by {rejected} at FDR {fdr:.0%}; inspect for data errors or a real machine bias."
    )
    return RandomnessReport(
        game=h.spec.code.value,
        draws=len(h),
        first_date=str(h.dates[0]) if len(h) else None,
        last_date=str(h.dates[-1]) if len(h) else None,
        tests=tests,
        fdr_level=fdr,
        rejected=rejected,
        verdict=verdict,
    )
