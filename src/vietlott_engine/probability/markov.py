"""Markov-chain models of draw-to-draw dependence.

1. ``NumberTransitionModel`` — an n×n matrix P(j ∈ draw t+1 | i ∈ draw t).
   Under independence every entry equals k/n. Significance is assessed with a
   permutation test that shuffles the *order* of draws: it preserves each draw
   exactly and destroys only temporal structure, which is precisely the null we
   care about ("the machine has no memory").

2. ``StateMarkovChain`` — a first-order chain over a discrete summary of each draw
   (sum tercile, odd count, repeats). Reports the transition matrix, stationary
   distribution, a G-test of first order vs order 0, and the mutual information
   between consecutive states (bits). MI ≈ 0 ⇔ no exploitable memory.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.analytics.randomness import TestResult
from vietlott_engine.analytics.stats import monte_carlo_p, subset_sum_distribution
from vietlott_engine.core.history import DrawHistory


class NumberTransitionModel:
    def __init__(self, n: int, k: int, smoothing: float = 10.0) -> None:
        self.n, self.k, self.smoothing = n, k, smoothing
        self.joint = np.zeros((n, n))
        self.row_totals = np.zeros(n)

    def fit(self, h: DrawHistory) -> "NumberTransitionModel":
        inc = h.incidence.astype(np.float64)
        self.joint = inc[:-1].T @ inc[1:]
        self.row_totals = inc[:-1].sum(axis=0)
        return self

    @property
    def transition(self) -> np.ndarray:
        """Smoothed P(j next | i now), shrunk toward k/n with ``smoothing`` pseudo-draws."""
        prior = self.k / self.n
        return (self.joint + self.smoothing * prior) / (self.row_totals[:, None] + self.smoothing)

    def predict(self, last_numbers: np.ndarray | list[int]) -> np.ndarray:
        """Inclusion probabilities for the next draw given the last draw (sums to k)."""
        rows = self.transition[np.asarray(last_numbers, dtype=int) - 1]
        p = rows.mean(axis=0)
        return p * self.k / p.sum()

    @staticmethod
    def dependence_statistic(inc: np.ndarray) -> float:
        prev, nxt = inc[:-1], inc[1:]
        joint = prev.T @ nxt
        expected = np.outer(prev.sum(axis=0), nxt.sum(axis=0)) / prev.shape[0]
        return float(np.sum((joint - expected) ** 2 / np.maximum(expected, 1e-12)))

    def permutation_test(self, h: DrawHistory, sims: int = 500, rng: np.random.Generator | None = None) -> TestResult:
        rng = rng or np.random.default_rng()
        inc = h.incidence.astype(np.float64)
        obs = self.dependence_statistic(inc)
        null = np.array([self.dependence_statistic(inc[rng.permutation(len(h))]) for _ in range(sims)])
        return TestResult(
            name="number_transition_dependence",
            statistic=obs,
            p_value=monte_carlo_p(obs, null, "greater"),
            method=f"Permutation test on draw order ({sims} shuffles)",
            detail={"null_mean": float(null.mean()), "null_sd": float(null.std())},
        )


class ChainSummary(BaseModel):
    name: str
    states: list[str]
    transition_matrix: list[list[float]]
    stationary: list[float]
    g_statistic: float
    df: int
    p_value: float
    mutual_information_bits: float


class StateMarkovChain:
    def __init__(self, n_states: int, smoothing: float = 0.5) -> None:
        self.s = n_states
        self.smoothing = smoothing
        self.counts = np.zeros((n_states, n_states))

    def fit(self, states: np.ndarray) -> "StateMarkovChain":
        states = np.asarray(states, dtype=int)
        self.counts = np.zeros((self.s, self.s))
        np.add.at(self.counts, (states[:-1], states[1:]), 1)
        return self

    @property
    def transition(self) -> np.ndarray:
        c = self.counts + self.smoothing
        return c / c.sum(axis=1, keepdims=True)

    def stationary(self) -> np.ndarray:
        vals, vecs = np.linalg.eig(self.transition.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        return v / v.sum()

    def independence_g_test(self) -> tuple[float, int, float]:
        o = self.counts
        e = np.outer(o.sum(axis=1), o.sum(axis=0)) / max(o.sum(), 1)
        nz = o > 0
        g = float(2 * np.sum(o[nz] * np.log(o[nz] / e[nz])))
        used_r = int(np.sum(o.sum(axis=1) > 0))
        used_c = int(np.sum(o.sum(axis=0) > 0))
        df = max(1, (used_r - 1) * (used_c - 1))
        return g, df, float(stats.chi2.sf(g, df))

    def mutual_information_bits(self) -> float:
        p = self.counts / max(self.counts.sum(), 1)
        pr, pc = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
        nz = p > 0
        return float(np.sum(p[nz] * np.log2(p[nz] / (pr @ pc)[nz])))

    def summary(self, name: str, labels: list[str]) -> ChainSummary:
        g, df, p = self.independence_g_test()
        return ChainSummary(
            name=name,
            states=labels,
            transition_matrix=[[round(float(x), 4) for x in row] for row in self.transition],
            stationary=[round(float(x), 4) for x in self.stationary()],
            g_statistic=g,
            df=df,
            p_value=p,
            mutual_information_bits=self.mutual_information_bits(),
        )


def draw_state_features(h: DrawHistory) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Discrete per-draw states used by the feature chains."""
    support, probs = subset_sum_distribution(h.n, h.k)
    cdf = np.cumsum(probs)
    t1, t2 = support[np.searchsorted(cdf, 1 / 3)], support[np.searchsorted(cdf, 2 / 3)]
    sums = h.numbers.sum(axis=1)
    sum_state = np.digitize(sums, [t1 + 0.5, t2 + 0.5])
    odd = (h.numbers % 2 == 1).sum(axis=1)
    odd_state = np.clip(odd - 2, 0, 2)  # ≤2, 3, ≥4
    rep = np.concatenate([[0], (h.incidence[1:] & h.incidence[:-1]).sum(axis=1)])
    rep_state = np.clip(rep, 0, 2)  # 0, 1, ≥2
    return {
        "sum_tercile": (sum_state, [f"≤{t1}", f"{t1 + 1}–{t2}", f">{t2}"]),
        "odd_count": (odd_state, ["≤2", "3", "≥4"]),
        "repeats_from_previous": (rep_state[1:], ["0", "1", "≥2"]),
    }


class MarkovReport(BaseModel):
    game: str
    draws: int
    number_dependence: TestResult
    feature_chains: list[ChainSummary]
    next_draw_top: list[dict]


def markov_report(h: DrawHistory, sims: int = 300, seed: int | None = None, smoothing: float = 10.0) -> MarkovReport:
    h.require(30)
    rng = np.random.default_rng(seed)
    model = NumberTransitionModel(h.n, h.k, smoothing).fit(h)
    test = model.permutation_test(h, sims, rng)
    chains = [StateMarkovChain(len(lbl)).fit(st).summary(name, lbl) for name, (st, lbl) in draw_state_features(h).items()]
    pred = model.predict(h.numbers[-1])
    base = h.k / h.n
    top = [{"number": int(i + 1), "probability": float(pred[i]), "lift": float(pred[i] / base)} for i in np.argsort(-pred)[:10]]
    return MarkovReport(game=h.spec.code.value, draws=len(h), number_dependence=test, feature_chains=chains, next_draw_top=top)
