from itertools import combinations, product

import numpy as np
import pytest

from vietlott_engine.forecast.data import DigitSpec, SetSpec
from vlm.forecast.features import FeatureState
from vlm.forecast.distribution import Law


def test_features_are_causal_and_gaps_are_censored():
    a, b = FeatureState(SetSpec(5, 2)), FeatureState(SetSpec(5, 2))
    rows = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]], float)
    snapshots = []
    for row, did in zip(rows, [1, 3]):
        snapshots.append(a.snapshot(did, '2026-01-01'))
        a.observe(row, did, '2026-01-01')
    np.testing.assert_array_equal(snapshots[0], b.snapshot(1, '2026-01-01'))
    b.observe(rows[0], 1, '2026-01-01')
    np.testing.assert_array_equal(snapshots[1], b.snapshot(3, '2026-01-01'))
    names = a.names
    assert snapshots[1][0, names.index('id_gap')] > 0
    assert snapshots[1][0, names.index('gap_censored')] == 1
    out = a.snapshot(4, '2026-01-02')
    assert out.shape == (5, len(names)) and np.isfinite(out).all()
    assert out[2, names.index('freq_all')] == pytest.approx(.7)  # smoothing (2 + 2*.4)/(2+2)
    assert out[3, names.index('lag1')] == 0
    with pytest.raises(ValueError):
        a.observe(rows[0], 3, '2026-01-02')


def test_feature_state_reload_preserves_pairs_triples_and_time():
    a = FeatureState(SetSpec(6, 3))
    a.observe(np.array([1, 1, 1, 0, 0, 0]), 2, '2026-01-01')
    a.observe(np.array([1, 1, 0, 0, 0, 1]), 3, '2026-01-02')
    b = FeatureState.from_dict(a.to_dict())
    np.testing.assert_array_equal(a.snapshot(4, '2026-01-03'), b.snapshot(4, '2026-01-03'))
    assert a.snapshot(4, '2026-01-03')[0, a.names.index('triple_link')] > 0
    with pytest.raises(ValueError):
        a.observe(np.array([1, 1, 0, 0, 0, 0]), 4, '2026-01-03')
    with pytest.raises(ValueError):
        a.snapshot(4, '2025-01-01')


def test_set_law_is_normalized_and_mixture_not_product_of_marginals():
    law = Law(SetSpec(4, 2), np.array([[1, 2, 3, 4], [4, 3, 2, 1]]), np.array([.75, .25]))
    probs = [law.probability(list(c)) for c in combinations(range(1, 5), 2)]
    assert sum(probs) == pytest.approx(1)
    assert law.probability([1, 2]) == pytest.approx((.75 * 2 + .25 * 12) / 35)
    assert law.marginals().sum() == pytest.approx(2)
    top, exact = law.top(3)
    assert exact and [t['numbers'] for t in top] == [[3, 4], [2, 4], [2, 3]]
    assert top[0]['p_model'] == pytest.approx(9.5 / 35)
    assert sum(t['p_model'] for t in top) < 1


def test_power_bonus_comes_from_remaining_drum():
    law = Law(SetSpec(4, 2, True), np.array([[1, 2, 3, 4]]), np.array([1.]))
    total = 0
    for c in combinations(range(1, 5), 2):
        for bonus in set(range(1, 5)) - set(c):
            total += law.probability(list(c), bonus=bonus)
    assert total == pytest.approx(1)
    assert law.probability([1, 2], bonus=3) == pytest.approx(2 / 35 * 3 / 7)
    with pytest.raises(ValueError):
        law.probability([1, 2], bonus=1)


def test_digit_law_handles_repeated_symbols_and_enumerates_exactly():
    spec = DigitSpec(2, 2, 1, ('a', 'b'))
    law = Law(spec, np.array([[[.8, .2], [.6, .4]]]), np.array([1.]))
    assert sum(law.probability(list(c)) for c in product(range(2), repeat=2)) == pytest.approx(1)
    assert law.probability([0, 0]) == pytest.approx(.48)
    top, exact = law.top(3)
    assert exact and top[0]['numbers'] == [0, 0]
    assert sum(t['p_model'] for t in top) == pytest.approx(.92)


@pytest.mark.parametrize('values', [[[0, 1, 2, 3]], [[1, float('nan'), 2, 3]], [[-1, 2, 3, 4]]])
def test_invalid_law_weights_do_not_silently_produce_a_forecast(values):
    with pytest.raises(ValueError):
        Law(SetSpec(4, 2), np.array(values), np.array([1.]))
