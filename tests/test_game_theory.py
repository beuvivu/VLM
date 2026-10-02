from __future__ import annotations

import itertools

import numpy as np
import pytest

from vietlott_engine.core.exceptions import DataValidationError
from vietlott_engine.core.games import MEGA_645, POWER_655, TaxRule
from vietlott_engine.game_theory.ev import EVCalculator, expected_net_share, optimize_tickets
from vietlott_engine.game_theory.popularity import (
    PopularityModel,
    PopularityParams,
    calibrate_number_weights,
    calibration_objective,
    elementary_symmetric,
    expected_tier_counts,
    sample_product_subsets,
)


def test_elementary_symmetric_and_sampler_exact() -> None:
    rng = np.random.default_rng(0)
    w = rng.random(8) + 0.3
    e = elementary_symmetric(w, 4)
    brute = [sum(np.prod(w[list(c)]) for c in itertools.combinations(range(8), j)) for j in range(5)]
    assert np.allclose(e, brute)

    combos = list(itertools.combinations(range(1, 8), 3))
    ww = np.array([1, 2, 0.5, 1.5, 1, 3, 0.7])
    p = np.array([np.prod(ww[np.array(c) - 1]) for c in combos])
    p /= p.sum()
    samples = sample_product_subsets(ww, 3, 100_000, rng)
    index = {c: i for i, c in enumerate(combos)}
    freq = np.bincount([index[tuple(s)] for s in samples], minlength=len(combos)) / len(samples)
    assert np.abs(freq - p).max() < 0.005


def test_population_probabilities_normalised() -> None:
    model = PopularityModel(MEGA_645)
    rng = np.random.default_rng(1)
    uniform = np.sort(np.argpartition(rng.random((200_000, 45)), 6, axis=1)[:, :6] + 1, axis=1)
    # E_uniform[p(c)]·C(n,k) must be 1 when p is a normalised distribution over combinations
    assert model.popularity_ratio(uniform).mean() == pytest.approx(1.0, rel=0.02)


def test_patterns_are_popular_high_numbers_are_not() -> None:
    model = PopularityModel(POWER_655, last_draw=(14, 18, 21, 38, 48, 52))
    ratio = lambda c: float(model.popularity_ratio(np.array([c]))[0])  # noqa: E731
    assert ratio((1, 2, 3, 4, 5, 6)) > 50
    assert ratio((14, 18, 21, 38, 48, 52)) > 20  # copying the last result
    assert ratio((34, 37, 40, 43, 47, 53)) < 0.6


def test_expected_share_matches_simulation() -> None:
    rng = np.random.default_rng(2)
    tax = TaxRule()
    pot, lam = 50e9, 0.8
    sim = np.mean([tax.after_tax(pot / (1 + k)) for k in rng.poisson(lam, 200_000)])
    assert expected_net_share(pot, lam, tax) == pytest.approx(sim, rel=0.01)
    assert expected_net_share(pot, 0.0, tax) == pytest.approx(pot - 0.1 * (pot - 20e6))


def test_ev_with_no_crowd_equals_textbook_value() -> None:
    params = PopularityParams(quick_pick_share=1.0)
    calc = EVCalculator(MEGA_645, PopularityModel(MEGA_645, params), tax=TaxRule(rate=0.0))
    b = calc.evaluate((1, 2, 3, 4, 5, 6), jackpot1=12e9, tickets_sold=0)
    p = MEGA_645.tier_probabilities
    expected = p["jackpot1"] * 12e9 + p["first"] * 10e6 + p["second"] * 300e3 + p["third"] * 30e3
    assert b.expected_value == pytest.approx(expected)
    assert b.popularity_ratio == pytest.approx(1.0)


def test_break_even_and_validation() -> None:
    calc = EVCalculator(POWER_655)
    b = calc.evaluate((34, 37, 40, 43, 47, 53), jackpot1=100e9)
    assert b.return_to_player < 1
    assert b.break_even_jackpot1 is not None and b.break_even_jackpot1 > 100e9
    assert calc.evaluate_ev_only((34, 37, 40, 43, 47, 53), b.break_even_jackpot1, None, 1_500_000) == pytest.approx(10_000, rel=1e-3)
    with pytest.raises(DataValidationError):
        calc.evaluate((1, 2, 3, 4, 5, 56))


def test_optimizer_beats_random_popularity_and_respects_overlap() -> None:
    calc = EVCalculator(MEGA_645)
    res = optimize_tickets(calc, n_tickets=4, max_overlap=2, iterations=3000, seed=0)
    assert res.max_pairwise_overlap <= 2
    assert all(t.popularity_ratio < 0.6 for t in res.tickets)
    assert res.average_ticket_ev > res.random_ticket_ev


def test_calibration_recovers_planted_weights() -> None:
    rng = np.random.default_rng(3)
    true_beta = rng.normal(0, 0.3, 45)
    true_beta -= true_beta.mean()
    model = PopularityModel(MEGA_645, beta=true_beta)
    draws = np.array([np.sort(rng.choice(np.arange(1, 46), 6, replace=False)) for _ in range(700)])
    sold = rng.integers(800_000, 1_500_000, len(draws)).astype(float)
    mu = np.array([expected_tier_counts(model, d, n, 3) for d, n in zip(draws, sold)])
    y = rng.poisson(mu)
    fit = calibrate_number_weights(MEGA_645, draws, y, sold, matches=3, quick_pick_share=model.params.quick_pick_share, ridge=0.1)
    assert fit.converged
    assert np.corrcoef(fit.beta, true_beta)[0, 1] > 0.9
    assert fit.likelihood_ratio_p_value < 1e-6


def test_calibration_gradient_is_exact() -> None:
    from scipy import optimize as sopt

    rng = np.random.default_rng(4)
    draws = np.array([np.sort(rng.choice(np.arange(1, 46), 6, replace=False)) for _ in range(20)])
    sold = np.full(20, 1e6)
    y = rng.poisson(20_000, 20).astype(float)
    f = calibration_objective(MEGA_645, draws, y, sold, matches=3, ridge=0.1)
    beta = rng.normal(0, 0.2, 45)
    grad = f(beta)[1]
    err = sopt.check_grad(lambda b: f(b)[0], lambda b: f(b)[1], beta, epsilon=1e-6)
    assert err / np.linalg.norm(grad) < 1e-4
