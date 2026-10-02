from __future__ import annotations

import numpy as np
import pytest

from vietlott_engine.backtest.engine import BacktestConfig, WalkForwardBacktester
from vietlott_engine.backtest.strategies import BayesianStrategy, FrequencyStrategy, RandomStrategy, Strategy, WheelStrategy
from vietlott_engine.core.exceptions import LookAheadError
from vietlott_engine.core.games import POWER_655
from vietlott_engine.core.history import DrawHistory


class SpyStrategy(Strategy):
    """Records what it is shown and tries to cheat."""

    name = "spy"

    def __init__(self) -> None:
        self.lengths: list[int] = []
        self.full: DrawHistory | None = None

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        self.lengths.append(len(history))
        base = history.incidence.base
        assert base is None or base.shape[0] == len(history), "prefix exposes future rows via .base"
        with pytest.raises(ValueError):
            history.incidence[0, 0] = not history.incidence[0, 0]
        return np.tile(np.arange(1, 7), (n_tickets, 1))


class RepeatLastDrawStrategy(Strategy):
    """Replays the most recent *known* result — a popular real-world 'system'."""

    name = "repeat_last"

    def generate(self, history: DrawHistory, n_tickets: int, rng: np.random.Generator) -> np.ndarray:
        return np.tile(history.numbers[-1], (n_tickets, 1))


def test_strategy_only_sees_the_past(mega_history: DrawHistory) -> None:
    spy = SpyStrategy()
    WalkForwardBacktester(mega_history, BacktestConfig(start=100, end=160, bootstrap_samples=10)).run_strategy(spy, 100, 160)
    assert spy.lengths == list(range(100, 160))


def test_scoring_matches_manual_computation(power_history: DrawHistory) -> None:
    bt = WalkForwardBacktester(power_history, BacktestConfig(start=100, end=200, bootstrap_samples=10, apply_tax=False))
    res, matches = bt.run_strategy(SpyStrategy(), 100, 200)
    manual = [len(set(range(1, 7)) & set(power_history.numbers[t].tolist())) for t in range(100, 200)]
    assert matches.tolist() == manual
    prize = {3: 50_000, 4: 500_000, 5: 40_000_000}
    expected_payout = sum(prize.get(m, 0) for m in manual)  # no 5+bonus/6 in a 100-draw synthetic sample
    if res.jackpot_hits == 0:
        assert res.payout == expected_payout


def test_invalid_tickets_are_rejected(mega_history: DrawHistory) -> None:
    class Bad(Strategy):
        name = "bad"

        def generate(self, history, n_tickets, rng):  # noqa: ANN001
            return np.array([[1, 1, 2, 3, 4, 5]])

    with pytest.raises(LookAheadError):
        WalkForwardBacktester(mega_history).run_strategy(Bad(), 100, 110)


def test_random_strategy_is_consistent_with_null(mega_history: DrawHistory) -> None:
    rep = WalkForwardBacktester(mega_history, BacktestConfig(start=100, tickets_per_draw=10, seed=3, bootstrap_samples=200)).run(
        [RandomStrategy(), FrequencyStrategy("hot", 50), BayesianStrategy(decay=0.99), WheelStrategy(9, 3, 3)]
    )
    by = {r.strategy: r for r in rep.results}
    assert abs(by["random"].z_mean_matches) < 4
    assert all(r.q_value_vs_null is not None for r in rep.results)
    assert "No strategy matched more numbers than chance" in rep.conclusion
    assert rep.spa is not None and rep.spa.p_value_consistent > 0.05
    assert by["hot_50"].design_effect > 1.2  # overlapping tickets inflate the variance…
    assert abs(by["random"].design_effect - 1) < 0.3  # …independent quick picks do not
    assert all(r.certified_edge_bound > abs(r.mean_matches - r.expected_mean_matches) for r in rep.results)
    assert by["wheel_9_3if3"].tickets % rep.draws_evaluated == 0
    assert rep.to_markdown().startswith("# Walk-forward backtest")


def test_repeating_last_draw_has_no_edge(mega_history: DrawHistory) -> None:
    rep = WalkForwardBacktester(mega_history, BacktestConfig(start=100, bootstrap_samples=50)).run([RepeatLastDrawStrategy()])
    assert abs(rep.results[0].z_mean_matches) < 4  # worth exactly k²/n matches, like any ticket


def test_power_bonus_tier(power_history: DrawHistory) -> None:
    class JackpotTwo(Strategy):
        name = "jp2"

        def __init__(self, target: int, full: DrawHistory) -> None:
            self.target, self.full = target, full

        def generate(self, history, n_tickets, rng):  # noqa: ANN001
            nums = list(self.full.numbers[self.target][:5]) + [int(self.full.bonus[self.target])]
            return np.array([nums])

    bt = WalkForwardBacktester(power_history, BacktestConfig(bootstrap_samples=10))
    res, _ = bt.run_strategy(JackpotTwo(150, power_history), 150, 151)
    assert res.tier_counts["jackpot2"] == 1 and res.jackpot_hits == 1
    assert POWER_655.classify(5, True).name == "jackpot2"
