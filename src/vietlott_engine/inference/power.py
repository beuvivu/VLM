"""Power analysis, minimum detectable effects and equivalence testing.

"No significant result" is only informative if the test *could* have detected a
relevant effect. This module answers two complementary questions:

1. **Power / MDE** — how large a bias would the existing history detect with a given
   probability? For the covariance-corrected frequency χ² (``analytics.randomness``)
   the statistic under an alternative with inclusion probabilities p_i = (k/n)(1+ε_i),
   Σε_i = 0, is approximately non-central χ²(n−1, λ) with

       λ = D·k·(n−1)·Σε_i² / (n·(n−k)).

2. **Equivalence (TOST)** — instead of failing to reject "fair", positively *certify*
   that every number's inclusion probability lies within ±δ of k/n. By the
   intersection–union principle a claim about *all* numbers needs each number's two
   one-sided tests at level α — no multiplicity correction. The smallest δ for which
   the claim holds is the "certified fairness margin".
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats

from vietlott_engine.core.games import DEFAULT_TAX, GameSpec
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.game_theory.popularity import divide_out, elementary_symmetric


# ------------------------------------------------------------------ power
def frequency_noncentrality(n: int, k: int, draws: int, sum_eps_sq: float) -> float:
    return draws * k * (n - 1) * sum_eps_sq / (n * (n - k))


def frequency_test_power(n: int, k: int, draws: int, eps: np.ndarray, alpha: float = 0.05) -> float:
    """Power of the corrected frequency χ² against relative deviations ``eps`` (Σeps = 0)."""
    eps = np.asarray(eps, dtype=np.float64)
    lam = frequency_noncentrality(n, k, draws, float(np.sum(eps**2)))
    crit = stats.chi2.ppf(1 - alpha, n - 1)
    return float(stats.ncx2.sf(crit, n - 1, lam))


def single_number_eps(n: int, eps: float) -> np.ndarray:
    """One number over-represented by ``eps`` (relative), the others compensating equally."""
    v = np.full(n, -eps / (n - 1))
    v[0] = eps
    return v


def mde_frequency(n: int, k: int, draws: int, alpha: float = 0.05, power: float = 0.8, pattern: str = "single") -> float:
    """Smallest relative deviation detectable with ``power``.

    ``pattern='single'``: one hot number (others compensate); ``'spread'``: every number
    off by ±ε (half up, half down).
    """
    crit = stats.chi2.ppf(1 - alpha, n - 1)
    lam = optimize.brentq(lambda x: stats.ncx2.sf(crit, n - 1, x) - power, 1e-6, 1e4)
    sum_sq_needed = lam * n * (n - k) / (draws * k * (n - 1))
    per_unit = n / (n - 1) if pattern == "single" else float(n)  # Σε² for ε = 1
    return float(np.sqrt(sum_sq_needed / per_unit))


def draws_needed(n: int, k: int, eps: float, alpha: float = 0.05, power: float = 0.8, pattern: str = "single") -> int:
    """History length needed to detect a relative bias ``eps`` with ``power``."""
    crit = stats.chi2.ppf(1 - alpha, n - 1)
    lam = optimize.brentq(lambda x: stats.ncx2.sf(crit, n - 1, x) - power, 1e-6, 1e4)
    per_unit = n / (n - 1) if pattern == "single" else float(n)
    return int(np.ceil(lam * n * (n - k) / (k * (n - 1) * per_unit * eps**2)))


def mde_mean(se: float, alpha: float = 0.05, power: float = 0.8, one_sided: bool = True) -> float:
    """Minimum detectable shift of a mean with standard error ``se`` (normal approximation)."""
    za = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha / 2)
    return float((za + stats.norm.ppf(power)) * se)


# ------------------------------------------------------------ equivalence
def tost_binomial_p(count: int, trials: int, p0: float, margin_rel: float) -> float:
    """Exact TOST p-value for H0: |p/p0 − 1| ≥ margin vs H1: |p/p0 − 1| < margin."""
    lo, hi = p0 * (1 - margin_rel), min(1.0, p0 * (1 + margin_rel))
    p_lower = stats.binom.sf(count - 1, trials, lo)  # P(X ≥ x | p = lo): tests p > lo
    p_upper = stats.binom.cdf(count, trials, hi)  # P(X ≤ x | p = hi): tests p < hi
    return float(max(p_lower, p_upper))


def certified_margin(count: int, trials: int, p0: float, alpha: float = 0.05) -> float:
    """Smallest relative margin δ such that TOST rejects non-equivalence at level α.

    Equivalent to the (1−2α) exact Clopper–Pearson interval lying inside p0(1±δ).
    """
    lo = stats.beta.ppf(alpha, count, trials - count + 1) if count > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha, count + 1, trials - count) if count < trials else 1.0
    return float(max(1 - lo / p0, hi / p0 - 1))


def tost_mean(diff: float, se: float, margin: float) -> float:
    """TOST p-value for a mean difference with (approximately) normal estimator."""
    p1 = stats.norm.sf((diff + margin) / se)  # H0: diff ≤ −margin
    p2 = stats.norm.cdf((diff - margin) / se)  # H0: diff ≥ +margin
    return float(max(p1, p2))


def certified_mean_margin(diff: float, se: float, alpha: float = 0.05) -> float:
    """Smallest margin such that TOST certifies |true diff| < margin."""
    return float(abs(diff) + stats.norm.ppf(1 - alpha) * se)


# ------------------------------------------------- money value of a bound
def odds_for_inclusion(target: np.ndarray, k: int, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Odds w of the conditional-Bernoulli draw model q(S) ∝ Π w_i whose marginal
    inclusion probabilities equal ``target`` (Σ target = k). Iterative proportional fitting."""
    target = np.asarray(target, dtype=np.float64)
    w = target / (1 - target)
    for _ in range(iters):
        e = elementary_symmetric(w, k)
        e_wo = divide_out(np.broadcast_to(e, (w.size, k + 1)), w)
        pi = w * e_wo[:, k - 1] / e[k]
        if np.max(np.abs(pi - target)) < tol:
            break
        w = w * target / pi
        w /= np.exp(np.mean(np.log(w)))
    return w


def ticket_match_distribution(w: np.ndarray, ticket: np.ndarray, k: int) -> np.ndarray:
    """Exact P(|S ∩ ticket| = m), m = 0..k, when S ~ q(S) ∝ Π_{i∈S} w_i."""
    mask = np.zeros(w.size, dtype=bool)
    mask[np.asarray(ticket, dtype=int) - 1] = True
    e_all = elementary_symmetric(w, k)[k]
    e_t = elementary_symmetric(w[mask], k)
    e_r = elementary_symmetric(w[~mask], k)
    return np.array([e_t[m] * e_r[k - m] for m in range(k + 1)]) / e_all


def ticket_rtp(spec: GameSpec, inclusion_probs: np.ndarray, ticket: np.ndarray) -> float:
    """Exact return-to-player of ``ticket`` when numbers are drawn with the given marginal
    inclusion probabilities (conditional-Bernoulli draw model; jackpots at their minimum,
    unshared; Power 5-match split into Jackpot 2 / first prize at the fair 1/(n−k))."""
    w = odds_for_inclusion(inclusion_probs, spec.pick)
    dist = ticket_match_distribution(w, ticket, spec.pick)
    ev = 0.0
    for t in spec.tiers:
        amount = float(t.fixed_amount if t.fixed_amount is not None else spec.min_jackpots[t.name])
        p = dist[t.main_matches]
        if spec.has_bonus and t.main_matches == spec.pick - 1:
            share = 1.0 / (spec.pool_size - spec.pick)
            p *= share if t.bonus_required else 1 - share
        ev += p * DEFAULT_TAX.after_tax(amount)
    return ev / spec.ticket_price


# ----------------------------------------------------------------- report
class PowerRow(BaseModel):
    relative_bias: float
    power_single_number: float
    power_spread: float


class NumberEquivalence(BaseModel):
    number: int
    observed_rate: float
    certified_margin: float  # relative, e.g. 0.12 = within ±12% of k/n


class PowerEquivalenceReport(BaseModel):
    game: str
    draws: int
    alpha: float
    mde_single_number_80: float
    mde_spread_80: float
    draws_needed_for_5pct_single: int
    power_curve: list[PowerRow]
    certified_margin_all_numbers: float
    certified_margin_median: float
    worst_numbers: list[NumberEquivalence]
    fair_ticket_rtp: float
    best_case_ticket: list[int]
    best_case_ticket_rtp: float  # the six numbers at their certified upper bounds (most optimistic case)
    interpretation: str


def power_equivalence_report(h: DrawHistory, alpha: float = 0.05) -> PowerEquivalenceReport:
    n, k, d = h.n, h.k, len(h)
    p0 = k / n
    counts = h.counts().astype(int)
    margins = np.array([certified_margin(int(c), d, p0, alpha) for c in counts])
    curve = [
        PowerRow(
            relative_bias=e,
            power_single_number=frequency_test_power(n, k, d, single_number_eps(n, e), alpha),
            power_spread=frequency_test_power(n, k, d, np.resize([e, -e], n) - np.mean(np.resize([e, -e], n)), alpha),
        )
        for e in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    ]
    mde1 = mde_frequency(n, k, d, alpha, 0.8, "single")
    mde_s = mde_frequency(n, k, d, alpha, 0.8, "spread")
    worst = [
        NumberEquivalence(number=int(i + 1), observed_rate=float(counts[i] / d), certified_margin=float(margins[i]))
        for i in np.argsort(-margins)[:5]
    ]
    all_margin = float(margins.max())
    upper = np.array([p0 * (1 + certified_margin(int(c), d, p0, alpha)) if c / d >= p0 else stats.beta.ppf(1 - alpha, c + 1, d - c) for c in counts])
    best6 = np.argsort(-upper)[:k]
    target = np.full(n, (k - upper[best6].sum()) / (n - k))  # the others absorb the difference
    target[best6] = upper[best6]
    fair_rtp = ticket_rtp(h.spec, np.full(n, p0), best6 + 1)
    best_rtp = ticket_rtp(h.spec, target, best6 + 1)
    return PowerEquivalenceReport(
        game=h.spec.code.value,
        draws=d,
        alpha=alpha,
        mde_single_number_80=mde1,
        mde_spread_80=mde_s,
        draws_needed_for_5pct_single=draws_needed(n, k, 0.05, alpha, 0.8, "single"),
        power_curve=curve,
        certified_margin_all_numbers=all_margin,
        certified_margin_median=float(np.median(margins)),
        worst_numbers=worst,
        fair_ticket_rtp=fair_rtp,
        best_case_ticket=sorted(int(i + 1) for i in best6),
        best_case_ticket_rtp=best_rtp,
        interpretation=(
            f"With {d} draws the frequency test detects one number running {mde1:.0%} hot (or every number "
            f"off by ±{mde_s:.1%}) with 80% power. At the {1 - alpha:.0%} level every number's inclusion "
            f"probability is certified to lie within ±{all_margin:.1%} of k/n = {p0:.4f} (intersection–union TOST). "
            f"Even if the six most favourable numbers sat at their upper bounds, a ticket's return-to-player "
            f"(min jackpot, fixed prizes) would rise from {fair_rtp:.1%} to at most {best_rtp:.1%}."
        ),
    )
