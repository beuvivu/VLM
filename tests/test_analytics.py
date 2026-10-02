from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from vietlott_engine.analytics.cooccurrence import cooccurrence_matrix, cooccurrence_report
from vietlott_engine.analytics.gaps import current_gaps, gap_report
from vietlott_engine.analytics.randomness import frequency_chi_square, randomness_report
from vietlott_engine.analytics.stats import adjacency_distribution, benjamini_hochberg, pooled_gof, subset_sum_distribution
from vietlott_engine.core.games import MEGA_645
from vietlott_engine.core.history import DrawHistory
from tests.conftest import make_history


def test_subset_sum_and_adjacency_exact_small_case() -> None:
    n, k = 9, 3
    subsets = list(combinations(range(1, n + 1), k))
    support, probs = subset_sum_distribution(n, k)
    brute = np.bincount([sum(s) for s in subsets]) / len(subsets)
    assert np.allclose(probs, brute[support])
    adj = adjacency_distribution(n, k)
    brute_adj = np.bincount([sum(b - a == 1 for a, b in zip(s, s[1:])) for s in subsets], minlength=k) / len(subsets)
    assert np.allclose(adj, brute_adj)


def test_corrected_chi_square_is_calibrated_under_h0() -> None:
    """p-values of the covariance-corrected statistic are Uniform(0,1) under H0;
    the naive Σ(O−E)²/E is not (it is too conservative)."""
    p_corr, p_naive = [], []
    for seed in range(300):
        h = make_history(MEGA_645, 200, seed=seed)
        r = frequency_chi_square(h)
        p_corr.append(r.p_value)
        p_naive.append(r.detail["naive_p_value_miscalibrated"])
    assert stats.kstest(p_corr, "uniform").pvalue > 0.01
    assert stats.kstest(p_naive, "uniform").pvalue < 1e-6


def test_battery_passes_fair_and_detects_biased_machine() -> None:
    fair = make_history(MEGA_645, 1200, seed=3)
    assert randomness_report(fair, mc_sims=200, seed=0).rejected == []
    w = np.ones(45)
    w[:5] = 1.6  # five "heavy" balls
    biased = make_history(MEGA_645, 1200, seed=3, weights=w)
    rep = randomness_report(biased, mc_sims=200, seed=0)
    assert "frequency_chi_square" in rep.rejected


def test_bh_and_pooled_gof() -> None:
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert np.all(np.diff(q[np.argsort([0.01, 0.04, 0.03, 0.2])]) >= -1e-12)
    assert q[0] == pytest.approx(0.04)
    g = pooled_gof(np.array([0, 1, 50, 49, 1]), np.array([0.001, 0.01, 0.49, 0.49, 0.009]))
    assert all(e >= 5 for e in g.expected)


def test_gaps_and_hazard(mega_history: DrawHistory) -> None:
    cur = current_gaps(mega_history)
    last = mega_history.numbers[-1] - 1
    assert np.all(cur[last] == 0)
    rep = gap_report(mega_history)
    p = MEGA_645.inclusion_probability
    assert rep.geometric_fit.p_value > 0.001
    for row in rep.hazard[:10]:
        assert row.ci_low - 0.02 <= p <= row.ci_high + 0.02


def test_cooccurrence_matrix_and_report(mega_history: DrawHistory) -> None:
    c = cooccurrence_matrix(mega_history)
    assert np.allclose(np.diag(c), mega_history.counts())
    assert np.allclose(c, c.T)
    assert c.sum() == pytest.approx(len(mega_history) * 36)
    rep = cooccurrence_report(mega_history, mc_sims=100, seed=0)
    assert rep.significant_pairs == 0
    assert rep.pair_probability == pytest.approx(30 / (45 * 44))
