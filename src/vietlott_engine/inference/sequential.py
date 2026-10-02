"""Anytime-valid sequential inference with e-processes (testing by betting).

A bettor starts with wealth 1 and, *before* each draw, stakes it on a distribution
q_t over the C(n,k) possible results, built only from past draws. After the draw S_t
the wealth is multiplied by

    E_t = q_t(S_t) / p₀(S_t),     p₀ = 1 / C(n, k)  (fair machine).

Under H0 (independent uniform draws) E₀[E_t | past] = Σ_S q_t(S) = 1, so wealth is a
non-negative martingale and Ville's inequality gives P(sup_t W_t ≥ 1/α) ≤ α. The
test can therefore be monitored after *every* new draw, forever, with no
multiple-looks penalty; 1/max W is an anytime-valid p-value.

q_t is the conditional-Bernoulli ("product") distribution q(S) = Π_{i∈S} w_i / e_k(w),
normalised exactly with the elementary symmetric polynomial e_k. Mixing q_t with the
uniform distribution bounds the per-draw loss, and averaging several bettors (a
mixture of martingales is a martingale) removes the need to tune them.

``ticket_eprocess`` gives the same guarantee for *any* ticket strategy: with
predictable tickets, the mean number of matches per ticket has conditional mean
exactly k²/n under H0, so E_t = 1 + λ(M̄_t − k²/n), 0 < λ ≤ n/k², is a valid e-value.
"""

from __future__ import annotations

from math import comb

import numpy as np
from pydantic import BaseModel
from scipy.special import logsumexp

from vietlott_engine.core.history import DrawHistory
from vietlott_engine.game_theory.popularity import elementary_symmetric


class EProcessResult(BaseModel):
    name: str
    draws: int
    final_log10_wealth: float
    max_log10_wealth: float
    anytime_p_value: float
    rejected: bool
    first_rejection_index: int | None
    first_rejection_date: str | None
    path: list[tuple[int, float]]  # (draw_id, log10 wealth), downsampled


def _result(name: str, log_w: np.ndarray, h: DrawHistory | None, offset: int, alpha: float, ids: np.ndarray | None = None) -> EProcessResult:
    lw10 = log_w / np.log(10)
    running_max = np.maximum.accumulate(lw10) if lw10.size else np.zeros(0)
    thr = np.log10(1 / alpha)
    hit = np.flatnonzero(lw10 >= thr)
    first = int(hit[0]) if hit.size else None
    step = max(1, lw10.size // 200)
    idx = np.arange(0, lw10.size, step)
    if ids is None and h is not None:
        ids = h.draw_ids[offset : offset + lw10.size]
    ids = ids if ids is not None else np.arange(lw10.size)
    max_lw = float(running_max[-1]) if lw10.size else 0.0
    return EProcessResult(
        name=name,
        draws=int(lw10.size),
        final_log10_wealth=float(lw10[-1]) if lw10.size else 0.0,
        max_log10_wealth=max_lw,
        anytime_p_value=float(min(1.0, 10 ** (-max(max_lw, 0.0)))),
        rejected=first is not None,
        first_rejection_index=first,
        first_rejection_date=str(h.dates[offset + first]) if (first is not None and h is not None) else None,
        path=[(int(ids[i]), round(float(lw10[i]), 4)) for i in idx],
    )


def subset_log_prob(weights: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """log q_t(S_t) for per-draw odds ``weights`` (T, n) and drawn sets ``draws`` (T, k, 1-based)."""
    w = np.asarray(weights, dtype=np.float64)
    k = draws.shape[1]
    e_k = elementary_symmetric(w, k)[:, k]
    picked = np.take_along_axis(w, np.asarray(draws, dtype=int) - 1, axis=1)
    return np.log(picked).sum(axis=1) - np.log(e_k)


def mixture_log_wealth(log_q: np.ndarray, n: int, k: int, eps_grid: tuple[float, ...]) -> np.ndarray:
    """Cumulative log wealth of the equal-weight mixture over uniform-mixing levels ε."""
    log_c = np.log(comb(n, k))
    comps = []
    for eps in eps_grid:
        factor = np.log((1 - eps) + eps * np.exp(log_q + log_c))
        comps.append(np.cumsum(factor))
    return logsumexp(np.vstack(comps), axis=0) - np.log(len(eps_grid))


def probabilities_to_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


# ----------------------------------------------------------------- bettors
def frequency_bettor_probs(h: DrawHistory, alpha0: float, decay: float) -> np.ndarray:
    """(D, n) predictable inclusion probabilities from decayed counts of draws < t."""
    inc = h.incidence.astype(np.float64)
    d, n = inc.shape
    p0 = h.k / n
    out = np.empty((d, n))
    c = np.zeros(n)
    total = 0.0
    for t in range(d):
        out[t] = (c + alpha0 * p0) / (total + alpha0)
        c = decay * c + inc[t]
        total = decay * total + 1.0
    return np.clip(out, 1e-4, 1 - 1e-4)


def markov_bettor_probs(h: DrawHistory, smoothing: float) -> np.ndarray:
    """(D, n) predictable P(j ∈ draw t | draw t−1) from transitions observed before t."""
    inc = h.incidence.astype(np.float64)
    d, n = inc.shape
    p0 = h.k / n
    joint = np.zeros((n, n))
    row = np.zeros(n)
    out = np.full((d, n), p0)
    for t in range(1, d):
        prev = inc[t - 1].astype(bool)
        trans = (joint[prev] + smoothing * p0) / (row[prev][:, None] + smoothing)
        p = trans.mean(axis=0)
        out[t] = p * h.k / p.sum()
        joint += np.outer(inc[t - 1], inc[t])
        row += inc[t - 1]
    return np.clip(out, 1e-4, 1 - 1e-4)


def frequency_log_wealth(h: DrawHistory, eps_grid: tuple[float, ...] = (0.1, 0.3, 1.0)) -> np.ndarray:
    """Mixture over prior strength (α₀) and memory (λ) of frequency-bias bettors."""
    log_ws = []
    for a0 in (30.0, 300.0, 3000.0):
        for lam in (1.0, 0.995):
            w = probabilities_to_odds(frequency_bettor_probs(h, a0, lam))
            log_ws.append(mixture_log_wealth(subset_log_prob(w, h.numbers), h.n, h.k, eps_grid))
    return logsumexp(np.vstack(log_ws), axis=0) - np.log(len(log_ws))


def markov_log_wealth(h: DrawHistory, eps_grid: tuple[float, ...] = (0.1, 0.3, 1.0)) -> np.ndarray:
    """Mixture of bettors that use the previous draw to predict the next one."""
    log_ws = []
    for sm in (10.0, 100.0):
        w = probabilities_to_odds(markov_bettor_probs(h, sm))
        log_ws.append(mixture_log_wealth(subset_log_prob(w, h.numbers), h.n, h.k, eps_grid))
    return logsumexp(np.vstack(log_ws), axis=0) - np.log(len(log_ws))


def frequency_eprocess(h: DrawHistory, alpha: float = 0.05) -> EProcessResult:
    return _result("frequency_bias", frequency_log_wealth(h), h, 0, alpha)


def markov_eprocess(h: DrawHistory, alpha: float = 0.05) -> EProcessResult:
    return _result("serial_dependence", markov_log_wealth(h), h, 0, alpha)


def predictive_eprocess(name: str, probs: np.ndarray, draws: np.ndarray, n: int, alpha: float = 0.05, ids: np.ndarray | None = None, eps_grid: tuple[float, ...] = (0.1, 0.3, 1.0)) -> EProcessResult:
    """E-process for an arbitrary *predictable* forecaster (e.g. the GCN).

    ``probs`` (T, n) are inclusion probabilities issued before each of the T draws in
    ``draws`` (T, k, 1-based numbers).
    """
    k = draws.shape[1]
    log_w = mixture_log_wealth(subset_log_prob(probabilities_to_odds(probs), draws), n, k, eps_grid)
    return _result(name, log_w, None, 0, alpha, ids)


def ticket_eprocess(name: str, mean_matches: np.ndarray, mu: float, alpha: float = 0.05, ids: np.ndarray | None = None, fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)) -> EProcessResult:
    """One-sided e-process "this ticket strategy matches more than chance" (per-draw mean matches)."""
    m = np.asarray(mean_matches, dtype=np.float64)
    comps = [np.cumsum(np.log1p((f / mu) * (m - mu))) for f in fractions]
    log_w = logsumexp(np.vstack(comps), axis=0) - np.log(len(fractions))
    return _result(name, log_w, None, 0, alpha, ids)


class SequentialReport(BaseModel):
    game: str
    draws: int
    alpha: float
    threshold_log10: float
    frequency: EProcessResult
    serial: EProcessResult
    combined: EProcessResult
    interpretation: str


def sequential_report(h: DrawHistory, alpha: float = 0.05) -> SequentialReport:
    lf, ls = frequency_log_wealth(h), markov_log_wealth(h)
    f = _result("frequency_bias", lf, h, 0, alpha)
    s = _result("serial_dependence", ls, h, 0, alpha)
    c = _result("combined", np.logaddexp(lf, ls) - np.log(2), h, 0, alpha)  # average of e-processes
    thr = float(np.log10(1 / alpha))
    if c.rejected:
        text = f"The combined bettor crossed 1/α = {1 / alpha:.0f}× on {c.first_rejection_date}: evidence against a fair, memoryless machine — investigate."
    else:
        text = (
            f"Betting on history never came close to the {1 / alpha:.0f}× rejection threshold (best: "
            f"10^{c.max_log10_wealth:.2f}×). Final wealth: frequency bettor 10^{f.final_log10_wealth:+.2f}, serial "
            f"bettor 10^{s.final_log10_wealth:+.2f}. Anytime-valid p = {c.anytime_p_value:.2f}; this test stays "
            "valid when re-run after every new draw."
        )
    return SequentialReport(game=h.spec.code.value, draws=len(h), alpha=alpha, threshold_log10=thr, frequency=f, serial=s, combined=c, interpretation=text)
