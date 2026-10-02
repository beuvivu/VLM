"""Calibration (size) and power checks for every v2 inference procedure."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from vietlott_engine.analytics.randomness import frequency_chi_square
from vietlott_engine.core.games import MEGA_645, POWER_655
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.game_theory.decision import kelly, payout_distribution, payout_profile
from vietlott_engine.game_theory.ev import EVCalculator
from vietlott_engine.inference.bootstrap import spa_test, stationary_bootstrap_indices
from vietlott_engine.inference.changepoint import changepoint_report
from vietlott_engine.inference.hierarchical import hierarchical_report
from vietlott_engine.inference.multiple_testing import benjamini_yekutieli, holm, per_number_deviations, westfall_young_maxt
from vietlott_engine.inference.power import (
    certified_margin,
    certified_mean_margin,
    frequency_test_power,
    mde_frequency,
    odds_for_inclusion,
    ticket_match_distribution,
    tost_binomial_p,
    tost_mean,
)
from vietlott_engine.inference.predictive import diebold_mariano, spiegelhalter_z
from vietlott_engine.inference.sequential import frequency_log_wealth, markov_log_wealth, sequential_report, subset_log_prob, ticket_eprocess
from vietlott_engine.wheeling.cover import WheelRequest, build_wheel
from tests.conftest import make_history


def _ar1(rng: np.random.Generator, t: int, phi: float, cols: int = 1) -> np.ndarray:
    e = rng.normal(size=(t, cols))
    for i in range(1, t):
        e[i] += phi * e[i - 1]
    return e


# ------------------------------------------------------- multiple testing
def test_holm_by_and_westfall_young_fwer() -> None:
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    assert benjamini_yekutieli([0.01, 0.04, 0.03, 0.2])[0] == pytest.approx(0.04 * (1 + 1 / 2 + 1 / 3 + 1 / 4))
    rng = np.random.default_rng(0)
    rejections = sum(westfall_young_maxt(np.abs(rng.normal(size=15)), np.abs(rng.normal(size=(500, 15)))).min() < 0.05 for _ in range(200))
    assert rejections / 200 <= 0.09


def test_per_number_finds_planted_hot_ball() -> None:
    w = np.ones(45)
    w[7] = 1.6
    rep = per_number_deviations(make_history(MEGA_645, 1500, seed=4, weights=w), sims=500, seed=0)
    assert rep.numbers[0].number == 8 and rep.any_significant_fwer
    fair = per_number_deviations(make_history(MEGA_645, 1500, seed=4), sims=500, seed=0)
    assert fair.min_westfall_young_p > 0.01


# ------------------------------------------------------ power & equivalence
def test_analytic_power_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    n, k, d = 45, 6, 1000
    target = np.full(n, k / n)
    target[:2] *= 1.3
    target[2:] = (k - target[:2].sum()) / (n - 2)
    w = odds_for_inclusion(target, k)
    rej, realized = 0, np.zeros(n)
    for _ in range(200):
        g = np.log(w)[None, :] + rng.gumbel(size=(d, n))
        # Gumbel top-k ≠ conditional Bernoulli exactly, so measure the realised deviation
        h = DrawHistory.from_arrays(MEGA_645, np.argpartition(-g, k, axis=1)[:, :k] + 1)
        realized += h.counts() / d
        rej += frequency_chi_square(h).p_value < 0.05
    eps = realized / 200 / (k / n) - 1
    assert abs(frequency_test_power(n, k, d, eps - eps.mean()) - rej / 200) < 0.1
    assert mde_frequency(45, 6, 1371) == pytest.approx(0.37, abs=0.02)


def test_tost_margin_is_the_boundary() -> None:
    p0, d, x = 6 / 45, 1371, 190
    m = certified_margin(x, d, p0, 0.05)
    assert tost_binomial_p(x, d, p0, m) == pytest.approx(0.05, abs=1e-6)
    assert tost_binomial_p(x, d, p0, m * 1.2) < 0.05 < tost_binomial_p(x, d, p0, m * 0.8)
    b = certified_mean_margin(0.01, 0.02)
    assert tost_mean(0.01, 0.02, b) == pytest.approx(0.05, abs=1e-9)


def test_exact_ticket_distribution_under_fair_and_tilted_draws() -> None:
    w = odds_for_inclusion(np.full(45, 6 / 45), 6)
    assert np.allclose(ticket_match_distribution(w, np.arange(1, 7), 6), MEGA_645.match_distribution())
    target = np.full(10, 3 / 10)
    target[:2], target[2:] = 0.5, (3 - 1.0) / 8
    w = odds_for_inclusion(target, 3)
    subsets = list(itertools.combinations(range(10), 3))
    q = np.array([np.prod(w[list(s)]) for s in subsets])
    q /= q.sum()
    incl = np.array([sum(q[j] for j, s in enumerate(subsets) if i in s) for i in range(10)])
    assert np.allclose(incl, target, atol=1e-8)


# ------------------------------------------------------------- hierarchical
def test_hierarchical_fair_vs_heterogeneous() -> None:
    fair = hierarchical_report(make_history(MEGA_645, 1371, seed=21))
    assert fair.log10_bayes_factor_material_vs_fair < 0 and fair.tau_relative_upper95 < 0.06
    rng = np.random.default_rng(3)
    w = np.exp(rng.normal(0, 0.2, 45))
    het = hierarchical_report(make_history(MEGA_645, 1371, seed=21, weights=w))
    assert het.log10_bayes_factor_material_vs_fair > 2
    assert het.pooled_best_ticket_rtp > het.fair_ticket_rtp


# ---------------------------------------------------------------- e-values
def test_subset_distribution_is_normalised() -> None:
    w = np.random.default_rng(0).random(9) + 0.2
    subs = np.array(list(itertools.combinations(range(1, 10), 4)))
    q = np.exp(subset_log_prob(np.tile(w, (len(subs), 1)), subs))
    assert len(subs) == comb(9, 4) and q.sum() == pytest.approx(1.0)


def test_eprocess_ville_bound_and_power() -> None:
    crossings = 0
    for s in range(60):
        h = make_history(MEGA_645, 500, seed=2000 + s)
        lw = np.logaddexp(frequency_log_wealth(h), markov_log_wealth(h)) - np.log(2)
        crossings += lw.max() >= np.log(20)
    assert crossings / 60 <= 0.1
    w = np.ones(45)
    w[:4] = 1.5
    assert sequential_report(make_history(MEGA_645, 1371, seed=7, weights=w)).combined.rejected


def test_ticket_eprocess_is_valid_for_any_predictable_tickets() -> None:
    rng = np.random.default_rng(5)
    mu = 36 / 45
    crossings = 0
    for _ in range(200):
        draws = np.argpartition(rng.random((400, 45)), 6, axis=1)[:, :6]
        inc = np.zeros((400, 45), bool)
        inc[np.arange(400)[:, None], draws] = True
        ticket = np.arange(6)  # the same fixed ticket every draw: maximally "correlated"
        crossings += ticket_eprocess("x", inc[:, ticket].sum(axis=1).astype(float), mu).max_log10_wealth >= np.log10(20)
    assert crossings / 200 <= 0.08


# ------------------------------------------------------------ change points
def test_changepoint_detects_local_regime_only() -> None:
    w = np.ones(45)
    w[:3] = 2.5
    parts = [make_history(MEGA_645, 500, seed=1).numbers, make_history(MEGA_645, 200, seed=2, weights=w).numbers, make_history(MEGA_645, 400, seed=3).numbers]
    rep = changepoint_report(DrawHistory.from_arrays(MEGA_645, np.vstack(parts)), sims=150, seed=0)
    assert rep.window_scan.p_value < 0.05 and set(rep.window_scan.hottest_numbers) == {1, 2, 3}
    fair = changepoint_report(make_history(POWER_655, 1100, seed=9), sims=150, seed=0)
    assert fair.window_scan.p_value > 0.01


# ---------------------------------------------------- forecast comparison
def test_diebold_mariano_size_with_autocorrelation() -> None:
    rng = np.random.default_rng(6)
    rej = sum(diebold_mariano(_ar1(rng, 500, 0.5)[:, 0]).p_value_two_sided < 0.05 for _ in range(300))
    assert rej / 300 < 0.1
    z, p = spiegelhalter_z(np.full(20_000, 0.2), rng.random(20_000) < 0.2)
    assert p > 0.001


def test_spa_size_and_power() -> None:
    rng = np.random.default_rng(7)
    idx = stationary_bootstrap_indices(50, 5, rng, 3)
    assert idx.shape == (3, 50) and idx.min() >= 0 and idx.max() < 50
    rej = sum(spa_test(_ar1(rng, 600, 0.2, 6), [str(i) for i in range(6)], bootstrap=300, seed=s).p_value_consistent < 0.05 for s in range(80))
    assert rej / 80 <= 0.12
    d = rng.normal(size=(600, 6))
    d[:, 2] += 0.2
    res = spa_test(d, [str(i) for i in range(6)], bootstrap=500, seed=0)
    assert res.best_strategy == "2" and res.p_value_consistent < 0.01


# --------------------------------------------------------------- decisions
def test_payout_distribution_and_kelly() -> None:
    calc = EVCalculator(MEGA_645)
    ticket = (33, 35, 37, 40, 42, 45)
    for j1, sold, positive in ((30e9, 1_500_000, False), (100e9, 4_000_000, True)):
        v, p = payout_distribution(calc, ticket, j1, None, sold)
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
        prof = payout_profile(v, p, 10_000)
        assert prof.expected_value == pytest.approx(calc.evaluate(ticket, j1, None, sold).expected_value, rel=1e-9)
        k = kelly(v, p, 10_000, 1e9)
        assert (k.optimal_fraction > 0) == positive
        if positive:
            f = k.optimal_fraction
            x = v / 10_000
            deriv = np.sum(p * (x - 1) / (1 + f * (x - 1)))
            assert abs(deriv) < 1e-3 and k.min_bankroll_for_one_ticket > 1e10


def test_wheel_ilp_proves_optimality() -> None:
    res = build_wheel(WheelRequest(pool=list(range(1, 10)), ticket_size=6, guarantee=4, condition=4), seed=1, time_limit=30)
    assert res.verified and res.n_tickets == 12 and res.optimality == "proven_optimal"
    quick = build_wheel(WheelRequest(pool=list(range(1, 11)), ticket_size=6, guarantee=3, condition=4), seed=1, exact=False)
    assert quick.lower_bound <= quick.n_tickets and quick.verified
