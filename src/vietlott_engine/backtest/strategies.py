"""Ticket-selection strategies evaluated by the walk-forward backtester.

Contract: ``fit`` and ``generate`` receive a detached, read-only prefix of the
history containing only draws strictly before the draw being predicted. A strategy
may keep internal state between calls but never receives future data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vietlott_engine.core.history import DrawHistory
from vietlott_engine.game_theory.ev import EVCalculator, optimize_tickets
from vietlott_engine.game_theory.popularity import PopularityModel
from vietlott_engine.ml_models.features import SnapshotBuilder
from vietlott_engine.ml_models.gcn import GCNConfig, GCNPredictor
from vietlott_engine.probability.bayesian import DirichletMultinomialModel
from vietlott_engine.probability.markov import NumberTransitionModel
from vietlott_engine.wheeling.cover import CoveringDesigner


def gumbel_topk(logits: np.ndarray, k: int, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_tickets`` k-subsets without replacement with P ∝ exp(logits) (Plackett–Luce)."""
    g = logits[None, :] + rng.gumbel(size=(n_tickets, logits.size))
    return np.sort(np.argpartition(-g, k, axis=1)[:, :k] + 1, axis=1)


class Strategy(ABC):
    name: str = "abstract"
    refit_every: int = 1
    min_history: int = 1

    def fit(self, history: DrawHistory) -> None:  # noqa: B027 - optional hook
        """Called by the engine every ``refit_every`` draws."""

    @abstractmethod
    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        """Return an (n_tickets, k) int array of 1-based numbers."""


class RandomStrategy(Strategy):
    """Quick pick — the null benchmark every other strategy must beat."""

    name = "random"

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        n, k = history.n, history.k
        return np.sort(np.argpartition(rng.random((n_tickets, n)), k, axis=1)[:, :k] + 1, axis=1)


class FrequencyStrategy(Strategy):
    """Hot (most frequent) or cold (least frequent) numbers over a rolling window."""

    def __init__(self, mode: str = "hot", window: int = 100, pool_factor: float = 2.0) -> None:
        if mode not in {"hot", "cold"}:
            raise ValueError("mode must be 'hot' or 'cold'")
        self.mode, self.window, self.pool_factor = mode, window, pool_factor
        self.name = f"{mode}_{window}"

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        counts = history.tail(self.window).counts()
        order = np.argsort(-counts if self.mode == "hot" else counts, kind="stable")
        pool = order[: int(history.k * self.pool_factor)] + 1
        return np.sort(np.array([rng.choice(pool, history.k, replace=False) for _ in range(n_tickets)]), axis=1)


class OverdueStrategy(Strategy):
    """Numbers with the longest current absence ("gan"/overdue)."""

    name = "overdue"

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        from vietlott_engine.analytics.gaps import current_gaps

        gaps = current_gaps(history)
        pool = np.argsort(-gaps, kind="stable")[: 2 * history.k] + 1
        return np.sort(np.array([rng.choice(pool, history.k, replace=False) for _ in range(n_tickets)]), axis=1)


class BayesianStrategy(Strategy):
    def __init__(self, alpha0: float = 1.0, decay: float = 0.99, temperature: float = 1.0) -> None:
        self.alpha0, self.decay, self.temperature = alpha0, decay, temperature
        self.name = f"bayes_decay{decay}"
        self.model: DirichletMultinomialModel | None = None

    def fit(self, history: DrawHistory) -> None:
        self.model = DirichletMultinomialModel(history.n, history.k, self.alpha0, self.decay).fit(history)

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        assert self.model is not None
        return gumbel_topk(np.log(self.model.posterior_mean) / self.temperature, history.k, n_tickets, rng)


class MarkovStrategy(Strategy):
    name = "markov"

    def __init__(self, smoothing: float = 10.0, sharpness: float = 5.0) -> None:
        self.smoothing, self.sharpness = smoothing, sharpness
        self.model: NumberTransitionModel | None = None

    def fit(self, history: DrawHistory) -> None:
        self.model = NumberTransitionModel(history.n, history.k, self.smoothing).fit(history)

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        assert self.model is not None
        p = self.model.predict(history.numbers[-1])
        return gumbel_topk(self.sharpness * np.log(p), history.k, n_tickets, rng)


class GCNStrategy(Strategy):
    """Samples tickets from the GCN's predicted inclusion probabilities."""

    name = "gcn"
    min_history = 120

    def __init__(self, refit_every: int = 100, sharpness: float = 5.0, config: GCNConfig | None = None) -> None:
        self.refit_every, self.sharpness = refit_every, sharpness
        self.config = config or GCNConfig(epochs=30)
        self.builder: SnapshotBuilder | None = None
        self.model: GCNPredictor | None = None

    def _sync(self, history: DrawHistory) -> None:
        if self.builder is None:
            self.builder = SnapshotBuilder(history.n, history.k)
        for row in history.incidence[self.builder.draws_seen :]:
            self.builder.update(row)

    def fit(self, history: DrawHistory) -> None:
        self._sync(history)
        assert self.builder is not None
        snaps = self.builder.snapshots()
        warmup = min(30, len(history) // 4)
        self.model = GCNPredictor(snaps.x.shape[2], self.config, history.k / history.n).fit(snaps, warmup, len(history))

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        self._sync(history)
        assert self.model is not None and self.builder is not None
        snaps = self.builder.snapshots()
        p = self.model.predict_proba(snaps, len(history))[0]
        return gumbel_topk(self.sharpness * np.log(p), history.k, n_tickets, rng)


class AntiPopularityStrategy(Strategy):
    """Plays the least popular combinations (EV / jackpot-share optimiser)."""

    name = "anti_popularity"

    def __init__(self, refit_every: int = 50, pool_size: int = 12, iterations: int = 3000) -> None:
        self.refit_every, self.pool_size, self.iterations = refit_every, pool_size, iterations
        self.pool: np.ndarray | None = None

    def fit(self, history: DrawHistory) -> None:
        model = PopularityModel(history.spec, last_draw=tuple(int(x) for x in history.numbers[-1]))
        res = optimize_tickets(EVCalculator(history.spec, model), n_tickets=self.pool_size, max_overlap=3, iterations=self.iterations, seed=len(history))
        self.pool = np.array([t.numbers for t in res.tickets])

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        assert self.pool is not None
        return self.pool[rng.choice(len(self.pool), n_tickets, replace=n_tickets > len(self.pool))]


class WheelStrategy(Strategy):
    """Covering-design wheel over the ``pool_size`` hottest numbers (ignores n_tickets)."""

    def __init__(self, pool_size: int = 10, guarantee: int = 3, condition: int = 4, window: int = 100, seed: int = 0) -> None:
        self.pool_size, self.window = pool_size, window
        self.name = f"wheel_{pool_size}_{condition}if{guarantee}"
        self.pattern: list[list[int]] | None = None
        self._design_args = (pool_size, guarantee, condition, seed)
        self.pool: np.ndarray | None = None

    def fit(self, history: DrawHistory) -> None:
        if self.pattern is None:
            v, t, m, seed = self._design_args
            designer = CoveringDesigner(v, history.k, t, m, seed=seed)
            masks = designer.solve()
            self.pattern = [[i for i in range(v) if mk >> i & 1] for mk in masks]
        counts = history.tail(self.window).counts()
        self.pool = np.argsort(-counts, kind="stable")[: self.pool_size] + 1

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        assert self.pattern is not None and self.pool is not None
        return np.sort(self.pool[np.asarray(self.pattern)], axis=1)


def default_strategies(include_ml: bool = True) -> list[Strategy]:
    s: list[Strategy] = [
        RandomStrategy(),
        FrequencyStrategy("hot", 50),
        FrequencyStrategy("cold", 50),
        OverdueStrategy(),
        BayesianStrategy(decay=0.99),
        MarkovStrategy(),
        AntiPopularityStrategy(),
    ]
    if include_ml:
        s.append(GCNStrategy())
    return s


STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "hot": lambda: FrequencyStrategy("hot", 50),
    "cold": lambda: FrequencyStrategy("cold", 50),
    "overdue": OverdueStrategy,
    "bayes": lambda: BayesianStrategy(decay=0.99),
    "markov": MarkovStrategy,
    "gcn": GCNStrategy,
    "anti_popularity": AntiPopularityStrategy,
    "wheel": WheelStrategy,
}
