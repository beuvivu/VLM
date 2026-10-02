from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln

from vietlott_engine.core.games import MEGA_645
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.probability.bayesian import DirichletMultinomialModel, bayesian_report, tune_decay
from vietlott_engine.probability.markov import NumberTransitionModel, StateMarkovChain, markov_report
from tests.conftest import make_history


def test_decayed_posterior_matches_manual(mega_history: DrawHistory) -> None:
    lam, a0 = 0.97, 2.0
    m = DirichletMultinomialModel(45, 6, a0, lam).fit(mega_history)
    d = len(mega_history)
    manual = a0 + sum(lam ** (d - 1 - t) * mega_history.incidence[t] for t in range(d))
    assert np.allclose(m.alpha, manual)
    assert m.inclusion_probabilities.sum() == pytest.approx(6.0)


def test_marginal_likelihood_closed_form_small() -> None:
    counts = np.array([3.0, 1.0, 0.0])
    a0 = 0.7
    # sequential predictive (Pólya urn) product must equal the closed form
    alpha = np.full(3, a0)
    logp = 0.0
    for cat, c in enumerate(counts):
        for _ in range(int(c)):
            logp += np.log(alpha[cat] / alpha.sum())
            alpha[cat] += 1
    assert DirichletMultinomialModel.log_marginal_likelihood(counts, a0) == pytest.approx(logp)
    assert gammaln(1.0) == 0.0


def test_bayes_factor_favours_uniform_on_fair_data_and_not_on_biased() -> None:
    fair = make_history(MEGA_645, 1500, seed=5)
    assert bayesian_report(fair).log_bayes_factor_vs_uniform < 0
    w = np.ones(45)
    w[:3] = 2.0
    biased = make_history(MEGA_645, 1500, seed=5, weights=w)
    rep = bayesian_report(biased, alpha0=5.0)
    assert rep.log_bayes_factor_vs_uniform > 0
    assert {x.number for x in rep.numbers[:3]} == {1, 2, 3}


def test_predictive_score_has_no_skill_on_fair_data() -> None:
    fair = make_history(MEGA_645, 600, seed=6)
    scores = tune_decay(fair, grid=(0.95, 1.0), alpha0=50.0, start=100)
    assert all(s.t_statistic < 3 for s in scores)


def test_markov_transition_permutation_test_power() -> None:
    fair = make_history(MEGA_645, 500, seed=7)
    model = NumberTransitionModel(45, 6).fit(fair)
    assert model.permutation_test(fair, sims=150, rng=np.random.default_rng(0)).p_value > 0.01
    assert model.predict(fair.numbers[-1]).sum() == pytest.approx(6.0)

    # planted memory: each draw repeats 3 numbers of the previous one
    rng = np.random.default_rng(8)
    rows = [np.sort(rng.choice(np.arange(1, 46), 6, replace=False))]
    for _ in range(499):
        keep = rng.choice(rows[-1], 3, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(1, 46), keep), 3, replace=False)
        rows.append(np.sort(np.concatenate([keep, rest])))
    sticky = DrawHistory.from_arrays(MEGA_645, np.array(rows))
    assert NumberTransitionModel(45, 6).fit(sticky).permutation_test(sticky, sims=150, rng=np.random.default_rng(0)).p_value < 0.01
    rep = markov_report(sticky, sims=100, seed=0)
    repeats = next(c for c in rep.feature_chains if c.name == "repeats_from_previous")
    assert repeats.mutual_information_bits >= 0


def test_state_chain_stationary_distribution() -> None:
    states = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1] * 50)
    ch = StateMarkovChain(2, smoothing=0.0).fit(states)
    pi = ch.stationary()
    assert pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi @ ch.transition, pi)
