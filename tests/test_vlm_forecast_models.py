import numpy as np
import pytest

from vlm.forecast.models import GRU, OnlineLogistic, TreeExpert


def test_gru_bptt_gradients_match_finite_differences():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 4, 3)) * .3
    y = np.array([1, 0, 1, 0], float)
    model = GRU(3, hidden=2, base=.4, seed=1)
    _, gradients = model.loss_and_gradients(x, y)
    eps = 1e-5
    for name, weights in model.params.items():
        for index in list(np.ndindex(weights.shape))[:4]:
            old = weights[index]
            weights[index] = old + eps
            positive, _ = model.loss_and_gradients(x, y)
            weights[index] = old - eps
            negative, _ = model.loss_and_gradients(x, y)
            weights[index] = old
            assert gradients[name][index] == pytest.approx((positive-negative)/(2*eps), abs=2e-7)


def test_online_models_learn_planted_signal_and_reload_optimizer():
    x = np.array([[-1., .2], [1., .2], [-1., -.2], [1., -.2]])
    y = np.array([0., 1., 0., 1.])
    logistic = OnlineLogistic(2, base=.5, l2=.001)
    gru = GRU(2, hidden=4, base=.5, seed=3)
    seq = np.stack([x, x])
    initial, _ = gru.loss_and_gradients(seq, y)
    for _ in range(180):
        logistic.learn(x, y)
        gru.learn(seq, y)
    after, _ = gru.loss_and_gradients(seq, y)
    assert after < initial * .4
    assert logistic.predict(x)[1] > .75 and logistic.predict(x)[0] < .25
    restored = GRU.from_dict(gru.to_dict())
    gru.learn(seq, y)
    restored.learn(seq, y)
    np.testing.assert_array_equal(gru.predict(seq), restored.predict(seq))


@pytest.mark.parametrize('backend', ['rf', 'xgb', 'lgb'])
def test_tree_backends_handle_soft_and_degenerate_labels_and_reload(backend):
    if backend != 'rf':
        pytest.importorskip({'xgb':'xgboost', 'lgb':'lightgbm'}[backend])
    x = np.arange(120, dtype=float).reshape(40, 3) / 120
    y = np.clip(x[:, 0] * .8 + .1, 0, 1)
    tree = TreeExpert(backend, base=.5, seed=1, depth=3)
    tree.fit(x, y)
    before = tree.predict(x)
    assert np.isfinite(before).all() and ((before > 0) & (before < 1)).all()
    restored = TreeExpert.from_dict(tree.to_dict())
    np.testing.assert_allclose(restored.predict(x), before, atol=1e-7)
    tree.fit(x, np.ones(40))
    assert np.isfinite(tree.predict(x)).all()
    tree.fit(x, np.zeros(40))
    assert np.isfinite(tree.predict(x)).all()


def test_requested_missing_backend_is_an_explicit_error(monkeypatch):
    import vlm.forecast.models as module
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name:None)
    with pytest.raises(ImportError, match='ml'):
        TreeExpert('xgb', base=.5)
