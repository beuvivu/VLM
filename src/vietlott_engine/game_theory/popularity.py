"""Model of how *other players* choose numbers (Anti-Popularity / Game Theory).

Every combination has the same probability of being drawn, so the only lever a
player controls is *how many other people hold the same combination* — that
decides how a pari-mutuel jackpot is split. This module models that crowd.

Population model (mixture):
    p(c) = q / C(n,k) + (1 − q) · p_manual(c)
where q is the quick-pick share (uniform random tickets) and

    p_manual(c) ∝ Π_{i∈c} w_i · f(c),      w_i = exp(β_i)

* β_i are per-number log-weights: birthday/calendar bias (1–31, 1–12), culturally
  lucky/unlucky numbers and endings (VN: 8 "phát", 9, 6 "lộc", 4 ≈ "tử", 13),
  and "copying the last result".
* f(c) captures combination-level patterns that are massively over-played in every
  lottery studied (arithmetic progressions such as 1-2-3-4-5-6, straight lines on
  the bet slip, the previous winning combination) and mild avoidance of
  consecutive numbers.

The normaliser Z = e_k(w)·E_π[f(c)] uses the elementary symmetric polynomial
e_k(w) (exact, by dynamic programming) and an exact sampler of the product-form
distribution π(c) ∝ Π w_i for the pattern correction.

The default parameters are *priors* taken from the published behaviour of other
lotteries (Henze & Riedwyl 1998; Simon 1999, UK Lotto; Cook & Clotfelter 1993;
Farrell et al. 2000) plus Vietnamese number folklore. ``calibrate_number_weights``
fits β from per-draw prize-winner counts (Vietlott publishes them) by exact
Poisson maximum likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable
import numpy as np
from scipy import optimize, stats

from vietlott_engine.core.games import GameSpec


# ------------------------------------------------------------ polynomial tools
def elementary_symmetric(w: np.ndarray, kmax: int) -> np.ndarray:
    """e_0..e_kmax of the last axis of ``w`` (supports batched input (..., m))."""
    w = np.asarray(w, dtype=np.float64)
    e = np.zeros(w.shape[:-1] + (kmax + 1,))
    e[..., 0] = 1.0
    for idx in range(w.shape[-1]):
        wi = w[..., idx : idx + 1]
        e[..., 1:] = e[..., 1:] + wi * e[..., :-1]
    return e


def divide_out(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Coefficients of G(x)/(1 + w·x) truncated to the same degree (exact for e_j, j ≤ kmax)."""
    w = np.asarray(w, dtype=np.float64)[..., None]
    h = np.zeros_like(e)
    h[..., 0] = e[..., 0]
    for j in range(1, e.shape[-1]):
        h[..., j] = e[..., j] - w[..., 0] * h[..., j - 1]
    return h


def sample_product_subsets(w: np.ndarray, k: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Exact i.i.d. samples of k-subsets with P(S) ∝ Π_{i∈S} w_i (returns 1-based, sorted)."""
    n = w.size
    prefix = np.zeros((n + 1, k + 1))
    prefix[0, 0] = 1.0
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1]
        prefix[i, 1:] += w[i - 1] * prefix[i - 1, :-1]
    need = np.full(size, k)
    chosen = np.zeros((size, k), dtype=np.int16)
    fill = np.zeros(size, dtype=int)
    for i in range(n, 0, -1):
        active = need > 0
        if not active.any():
            break
        r = need[active]
        p_in = w[i - 1] * prefix[i - 1, r - 1] / prefix[i, r]
        take = rng.random(r.size) < p_in
        idx = np.flatnonzero(active)[take]
        chosen[idx, fill[idx]] = i
        fill[idx] += 1
        need[idx] -= 1
    return np.sort(chosen, axis=1)


def _max_run_length(adjacent: np.ndarray) -> np.ndarray:
    """Length of the longest run of consecutive numbers given (m, k-1) adjacency flags."""
    best = np.ones(adjacent.shape[0], dtype=int)
    cur = np.ones(adjacent.shape[0], dtype=int)
    for j in range(adjacent.shape[1]):
        cur = np.where(adjacent[:, j], cur + 1, 1)
        best = np.maximum(best, cur)
    return best


# --------------------------------------------------------------- parameters
@dataclass(frozen=True)
class PopularityParams:
    quick_pick_share: float = 0.35
    birthday_bonus: float = 0.35  # numbers 1..31
    month_bonus: float = 0.15  # numbers 1..12 (on top of birthday)
    number_bonus: dict[int, float] = field(default_factory=lambda: {7: 0.20, 8: 0.20, 9: 0.15, 6: 0.10, 13: -0.20})
    ending_digit_bonus: dict[int, float] = field(default_factory=lambda: {8: 0.10, 9: 0.08, 6: 0.06, 4: -0.10})
    last_draw_bonus: float = 0.15
    # combination-level log factors
    arithmetic_progression: float = 3.5
    slip_line: float = 2.5
    last_draw_exact: float = 4.0
    consecutive_pair: float = -0.10  # isolated pairs are mildly avoided...
    long_run: float = 1.5  # ...but runs of >= 4 consecutive numbers are a picked pattern
    all_birthday: float = 0.30
    slip_columns: int = 10
    normaliser_samples: int = 40_000


@dataclass
class PopularityModel:
    spec: GameSpec
    params: PopularityParams = field(default_factory=PopularityParams)
    last_draw: tuple[int, ...] | None = None
    beta: np.ndarray | None = None  # overrides the prior per-number weights (e.g. after calibration)
    seed: int = 7
    special_probs: np.ndarray | None = None  # Lotto 5/35: manual-pick distribution of the special number

    def __post_init__(self) -> None:
        n = self.spec.pool_size
        if self.beta is None:
            self.beta = self.prior_beta()
        self.beta = np.asarray(self.beta, dtype=np.float64).reshape(n)
        self.w = np.exp(self.beta)
        self._ek = elementary_symmetric(self.w, self.spec.pick)[self.spec.pick]
        rng = np.random.default_rng(self.seed)
        sample = sample_product_subsets(self.w, self.spec.pick, self.params.normaliser_samples, rng)
        self._pattern_mean = float(np.mean(np.exp(self.pattern_log_factor(sample))))
        self._z = self._ek * self._pattern_mean

    # -------------------------------------------------------- per-number prior
    def prior_beta(self) -> np.ndarray:
        p, n = self.params, self.spec.pool_size
        x = np.arange(1, n + 1)
        beta = np.where(x <= 31, p.birthday_bonus, 0.0) + np.where(x <= 12, p.month_bonus, 0.0)
        for num, b in p.number_bonus.items():
            if 1 <= num <= n:
                beta[num - 1] += b
        for digit, b in p.ending_digit_bonus.items():
            beta[(x % 10) == digit] += b
        if self.last_draw:
            beta[np.asarray(self.last_draw) - 1] += p.last_draw_bonus
        return beta - beta.mean()

    # --------------------------------------------------------- pattern factor
    def pattern_log_factor(self, combos: np.ndarray) -> np.ndarray:
        """log f(c) for combos (m, k) (sorted, 1-based)."""
        p = self.params
        c = np.atleast_2d(np.asarray(combos, dtype=np.int64))
        diffs = np.diff(c, axis=1)
        is_ap = np.all(diffs == diffs[:, :1], axis=1)
        rows = (c - 1) // p.slip_columns
        cols = (c - 1) % p.slip_columns
        is_line = np.all(rows == rows[:, :1], axis=1) | np.all(cols == cols[:, :1], axis=1)
        out = np.where(is_ap, p.arithmetic_progression, np.where(is_line, p.slip_line, 0.0))
        max_run = _max_run_length(diffs == 1)
        out += np.where(~is_ap & (max_run >= 4), p.long_run, 0.0)
        out += np.where(~is_ap & (max_run < 4), p.consecutive_pair * np.sum(diffs == 1, axis=1), 0.0)
        out += np.where(np.all(c <= 31, axis=1), p.all_birthday, 0.0)
        if self.last_draw:
            out += np.where(np.all(c == np.asarray(sorted(self.last_draw)), axis=1), p.last_draw_exact, 0.0)
        return out

    # ------------------------------------------------------------ popularity
    def manual_probability(self, combos: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(np.asarray(combos, dtype=np.int64))
        log_w = self.beta[c - 1].sum(axis=1) + self.pattern_log_factor(c)
        return np.exp(log_w) / self._z

    def ticket_probability_joint(self, combos: np.ndarray, specials: np.ndarray) -> np.ndarray:
        """Separate-drum games: P(ticket = main combo c with special σ)
        = q/(C·B) + (1 − q)·M(c)·π(σ)."""
        b = float(self.spec.bonus_pool_size or 1)
        q = self.params.quick_pick_share
        pi = np.full(int(b), 1.0 / b) if self.special_probs is None else np.asarray(self.special_probs)
        sp = np.asarray(specials, dtype=int).reshape(-1)
        return q / (self.spec.main_combinations * b) + (1 - q) * self.manual_probability(combos) * pi[sp - 1]

    def ticket_probability(self, combos: np.ndarray) -> np.ndarray:
        """Probability that one randomly chosen ticket in the population equals each combo."""
        q = self.params.quick_pick_share
        return q / self.spec.main_combinations + (1 - q) * self.manual_probability(combos)

    def popularity_ratio(self, combos: np.ndarray) -> np.ndarray:
        """1.0 = an average combination; 2.0 = held by twice as many players as average."""
        return self.ticket_probability(combos) * self.spec.main_combinations

    def expected_co_holders(self, combos: np.ndarray, tickets_sold: int) -> np.ndarray:
        return tickets_sold * self.ticket_probability(combos)

    def sample_population(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate tickets bought by the crowd (quick picks + manual picks, pattern-weighted
        by rejection sampling)."""
        q = self.params.quick_pick_share
        n_qp = rng.binomial(size, q)
        n, k = self.spec.pool_size, self.spec.pick
        qp = np.sort(np.argpartition(rng.random((n_qp, n)), k, axis=1)[:, :k] + 1, axis=1)
        manual: list[np.ndarray] = []
        need = size - n_qp
        fmax = np.exp(max(self.params.arithmetic_progression, self.params.slip_line, self.params.long_run) + self.params.last_draw_exact + self.params.all_birthday)
        while need > 0:
            cand = sample_product_subsets(self.w, k, max(need * 2, 1000), rng)
            acc = rng.random(cand.shape[0]) < np.exp(self.pattern_log_factor(cand)) / fmax
            manual.append(cand[acc][:need])
            need -= manual[-1].shape[0]
        return np.vstack([qp, *manual]) if manual else qp

    # ---------------------------------------------------------- calibration
    def with_beta(self, beta: np.ndarray) -> "PopularityModel":
        return PopularityModel(self.spec, self.params, self.last_draw, beta, self.seed, self.special_probs)

    def with_params(self, **kwargs: object) -> "PopularityModel":
        return PopularityModel(self.spec, replace(self.params, **kwargs), self.last_draw, self.beta, self.seed, self.special_probs)


@dataclass(frozen=True)
class CalibrationResult:
    beta: np.ndarray
    log_likelihood: float
    null_log_likelihood: float  # uniform weights
    likelihood_ratio_p_value: float
    converged: bool
    iterations: int


def calibration_objective(
    spec: GameSpec,
    winning_numbers: np.ndarray,
    tier_winners: np.ndarray,
    tickets_sold: np.ndarray,
    matches: int = 3,
    quick_pick_share: float = 0.35,
    ridge: float = 1.0,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Negative penalised Poisson log-likelihood in β and its analytic gradient.

    For draw t with winning set W_t, the expected number of tickets with exactly
    ``matches`` hits is

        μ_t = N_t · [ q·H(m) + (1−q)·e_m(w_W)·e_{k−m}(w_L)/e_k(w) ]

    (product-form manual picks; pattern terms affect a negligible fraction of the
    ~10⁴–10⁵ tier-3 tickets and are ignored here). The gradient uses
    ∂ log e_j(S)/∂β_i = w_i · e_{j−1}(S∖{i}) / e_j(S).
    """
    n, k, m = spec.pool_size, spec.pick, matches
    wn = np.asarray(winning_numbers, dtype=int) - 1  # (T, k)
    y = np.asarray(tier_winners, dtype=np.float64)
    big_n = np.asarray(tickets_sold, dtype=np.float64)
    h_m = spec.match_probability(m)
    q = quick_pick_share
    t_count = wn.shape[0]
    t_idx = np.arange(t_count)
    in_w = np.zeros((t_count, n), dtype=bool)
    in_w[t_idx[:, None], wn] = True

    def terms(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.exp(beta)
        e_all = elementary_symmetric(w, k)  # (k+1,)
        w_win = w[wn]  # (T, k)
        e_w = elementary_symmetric(w_win, k)  # (T, k+1)
        e_l = np.broadcast_to(e_all, (t_count, k + 1)).copy()
        for j in range(k):
            e_l = divide_out(e_l, w_win[:, j])
        r = e_w[:, m] * e_l[:, k - m] / e_all[k]
        g = np.zeros((t_count, n))  # ∂ log r_t / ∂ β_i
        if m >= 1:
            for j in range(k):  # members of W
                e_wo = divide_out(e_w, w_win[:, j])
                g[t_idx, wn[:, j]] += w_win[:, j] * e_wo[:, m - 1] / e_w[:, m]
        if k - m >= 1:  # members of L
            e_l_wo = divide_out(np.broadcast_to(e_l[:, None, :], (t_count, n, k + 1)), np.broadcast_to(w, (t_count, n)))
            g += np.where(~in_w, w[None, :] * e_l_wo[:, :, k - m - 1] / e_l[:, k - m][:, None], 0.0)
        e_all_wo = divide_out(np.broadcast_to(e_all, (n, k + 1)), w)
        g -= (w * e_all_wo[:, k - 1] / e_all[k])[None, :]
        return r, g

    def neg_ll(beta: np.ndarray) -> tuple[float, np.ndarray]:
        r, g = terms(beta)
        mu = big_n * (q * h_m + (1 - q) * r)
        ll = np.sum(y * np.log(mu) - mu) - 0.5 * ridge * np.sum(beta**2)
        dmu = (big_n * (1 - q) * r)[:, None] * g
        grad = ((y / mu - 1)[:, None] * dmu).sum(axis=0) - ridge * beta
        return -float(ll), -grad

    return neg_ll


def calibrate_number_weights(
    spec: GameSpec,
    winning_numbers: np.ndarray,
    tier_winners: np.ndarray,
    tickets_sold: np.ndarray,
    matches: int = 3,
    quick_pick_share: float = 0.35,
    ridge: float = 1.0,
) -> CalibrationResult:
    """Fit per-number log-weights β by penalised Poisson ML from prize-winner counts
    (L-BFGS with the analytic gradient of ``calibration_objective``)."""
    n = spec.pool_size
    neg_ll = calibration_objective(spec, winning_numbers, tier_winners, tickets_sold, matches, quick_pick_share, ridge)
    res = optimize.minimize(neg_ll, np.zeros(n), jac=True, method="L-BFGS-B")
    ll_fit = -res.fun + 0.5 * ridge * float(np.sum(res.x**2))
    ll_null = -neg_ll(np.zeros(n))[0]
    lr = max(0.0, 2 * (ll_fit - ll_null))
    return CalibrationResult(
        beta=res.x - res.x.mean(),
        log_likelihood=float(ll_fit),
        null_log_likelihood=float(ll_null),
        likelihood_ratio_p_value=float(stats.chi2.sf(lr, n - 1)),
        converged=bool(res.success),
        iterations=int(res.nit),
    )


def expected_tier_counts(model: PopularityModel, winning: np.ndarray, tickets_sold: float, matches: int) -> float:
    """Model-implied number of tickets with exactly ``matches`` hits (for diagnostics)."""
    spec = model.spec
    k, m = spec.pick, matches
    w = model.w
    e_all = elementary_symmetric(w, k)
    w_win = w[np.asarray(winning) - 1]
    e_w = elementary_symmetric(w_win, k)
    e_l = e_all.copy()
    for wi in w_win:
        e_l = divide_out(e_l, wi)
    r = e_w[m] * e_l[k - m] / e_all[k]
    q = model.params.quick_pick_share
    return float(tickets_sold * (q * spec.match_probability(m) + (1 - q) * r))


__all__ = [
    "PopularityParams",
    "PopularityModel",
    "calibrate_number_weights",
    "calibration_objective",
    "elementary_symmetric",
    "sample_product_subsets",
    "expected_tier_counts",
]
