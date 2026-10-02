from __future__ import annotations

import numpy as np

from vietlott_engine.core.games import MEGA_645
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.ml_models.features import build_snapshots
from vietlott_engine.ml_models.gcn import GCNConfig, GCNPredictor, walk_forward_skill
from tests.conftest import make_history


def test_snapshots_are_causal(mega_history: DrawHistory) -> None:
    full = build_snapshots(mega_history)
    prefix = build_snapshots(mega_history.upto(400))
    assert np.allclose(full.x[:401], prefix.x)
    assert np.allclose(full.a_hat[:401], prefix.a_hat)
    assert full.x.shape == (len(mega_history) + 1, 45, 6)
    assert np.isfinite(full.x).all() and np.isfinite(full.a_hat).all()


def test_gcn_gradients_match_finite_differences(mega_history: DrawHistory) -> None:
    snaps = build_snapshots(mega_history)
    model = GCNPredictor(snaps.x.shape[2], GCNConfig(hidden=4, l2=1e-2), 6 / 45)
    idx = np.arange(100, 104)
    a, x, y = snaps.a_hat[idx].astype(float), snaps.x[idx].astype(float), snaps.y[idx].astype(float)
    _, grads = model.loss_and_grads(model.params, a, x, y)
    eps = 1e-6
    for p, g in zip(model.params.as_list(), grads):
        for i in list(np.ndindex(p.shape))[:5]:
            old = p[i]
            p[i] = old + eps
            lp, _ = model.loss_and_grads(model.params, a, x, y)
            p[i] = old - eps
            lm, _ = model.loss_and_grads(model.params, a, x, y)
            p[i] = old
            assert abs((lp - lm) / (2 * eps) - g[i]) < 1e-6


def test_gcn_learns_planted_structure_but_not_noise() -> None:
    """Sanity check of the null result: the model *can* find real signal."""
    rng = np.random.default_rng(0)
    rows = [np.sort(rng.choice(np.arange(1, 46), 6, replace=False))]
    for _ in range(599):  # strong memory: 3 numbers carried over from the previous draw
        keep = rng.choice(rows[-1], 3, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(1, 46), keep), 3, replace=False)
        rows.append(np.sort(np.concatenate([keep, rest])))
    sticky = DrawHistory.from_arrays(MEGA_645, np.array(rows))
    cfg = GCNConfig(epochs=25, seed=0)
    signal = walk_forward_skill(build_snapshots(sticky), 6, first_test=400, refit_every=200, config=cfg)
    assert signal.t_statistic > 5 and signal.mean_hits_top_k > signal.expected_hits_random + 0.5

    noise = walk_forward_skill(build_snapshots(make_history(MEGA_645, 600, seed=9)), 6, first_test=400, refit_every=200, config=cfg)
    assert noise.t_statistic < 3
