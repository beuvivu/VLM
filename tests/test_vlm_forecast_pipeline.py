from datetime import datetime, timezone

import numpy as np
import pytest

from vietlott_engine.core.products import ProductCode
from vietlott_engine.forecast.data import Series
from vlm.forecast.pipeline import MLConfig, MLForecaster


def mega_series(size=90, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros((size, 45), dtype=bool)
    for row in x:
        row[rng.choice(45, 6, replace=False)] = True
    return Series(ProductCode.MEGA_645, np.arange(1, size + 1), np.full(size, '2026-01-02'),
                  {'main':{'X':x, 'bonus':np.zeros(size, dtype=int)}})


def config(**kw):
    return MLConfig(bootstrap=100, warmup=8, tree_every=20, tree_buffer=24, search_nodes=50, **kw)


def test_learning_is_incremental_reload_and_chunk_invariant(tmp_path):
    series = mega_series()
    whole = MLForecaster('mega645', config())
    assert whole.update(series) == 90
    split = MLForecaster('mega645', config())
    assert split.update(series.head(45)) == 45
    path = tmp_path / 'state.json.gz'
    split.save(path)
    split = MLForecaster.load(path)
    assert split.update(series) == 45
    assert split.update(series) == 0
    np.testing.assert_array_equal(whole.components['main'].law().marginals(), split.components['main'].law().marginals())
    assert whole.report()['components'][0]['metrics'] == split.report()['components'][0]['metrics']
    assert split.report()['confidence']['scored_live'] == 0
    assert not split.report()['confidence']['validated']


def test_corrected_prefix_resets_weights_and_live_evidence():
    series = mega_series()
    f = MLForecaster('mega645', config())
    f.update(series)
    f.live_log_e = 100
    corrected = mega_series()
    corrected.obs['main']['X'][0] = corrected.obs['main']['X'][1]
    assert f.update(corrected) == 90
    assert f.live_log_e == 0
    assert 'history_revised' in f.warnings
    reference = MLForecaster('mega645', config())
    reference.update(corrected)
    np.testing.assert_array_equal(f.components['main'].law().marginals(), reference.components['main'].law().marginals())


def test_bootstrap_is_bounded_and_never_certifies_retrospective_success():
    f = MLForecaster('mega645', MLConfig(bootstrap=40, warmup=8, tree_buffer=16, search_nodes=20))
    assert f.update(mega_series(90)) == 40
    f.components['main'].log_gain = 1000
    report = f.report(3)
    assert report['draws_learned'] == 40 and report['historical_draws_available'] == 90
    assert not report['confidence']['validated']
    assert all(t['p_deployed'] == t['p_fair'] for t in report['components'][0]['top'])
    assert sum(t['p_model'] for t in report['components'][0]['top']) < 1


def test_live_score_uses_the_issued_distribution_before_any_learning(tmp_path):
    f = MLForecaster('mega645', config())
    series = mega_series(40)
    series.dates[-1] = '2026-01-04'  # next scheduled Mega draw, Sunday
    f.update(series.head(39))
    law = f.components['main'].law('2026-01-04')
    expected = law.log_likelihood(series.obs['main']['X'][39]) - law.null_log_likelihood(series.obs['main']['X'][39])
    assert f.issue('2026-01-04T18:00:00+07:00', '2026-01-04T17:00:00+07:00')
    path = tmp_path/'issued.json.gz'
    f.save(path)
    f = MLForecaster.load(path)
    f.update(series)
    assert f.live_scored == 1 and f.live_log_e == pytest.approx(expected)
    assert f.update(series) == 0 and f.live_scored == 1
    assert not f.issue('2026-01-04T18:00:00+07:00', '2026-01-04T19:00:00+07:00')
    assert not f.issue('2026-01-02T18:30:00', '2026-01-02T17:00:00')


def test_invalid_and_gapped_data_cannot_be_called_complete():
    series = mega_series(20)
    series.draw_ids[10:] += 1
    f = MLForecaster('mega645', config())
    f.update(series)
    assert f.report()['missing_ids_inside_range'] == 1
    bad = mega_series(20)
    bad.draw_ids[2] = bad.draw_ids[1]
    with pytest.raises(ValueError):
        f.update(bad)
    bad = mega_series(20)
    bad.obs['main']['X'][0] = False
    with pytest.raises(ValueError):
        f.update(bad)
    bad = mega_series(20)
    bad.dates[:] = '2099-01-01'
    with pytest.raises(ValueError):
        f.update(bad, now=datetime(2026, 10, 3, tzinfo=timezone.utc))


def test_corrupted_checkpoint_does_not_preserve_fake_evidence(tmp_path):
    path = tmp_path / 'state.json.gz'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises((ValueError, OSError)):
        MLForecaster.load(path)


def test_vietnam_draw_date_is_not_rejected_when_utc_is_previous_day():
    series = mega_series(2)
    series.dates[:] = '2026-01-02'
    f = MLForecaster('mega645', config())
    assert f.update(series, now=datetime(2026, 1, 1, 23, tzinfo=timezone.utc)) == 2


def test_lost_or_mismatched_live_target_blocks_old_certification():
    series = mega_series(40)
    f = MLForecaster('mega645', config())
    f.update(series.head(39))
    f.live_scored = 100
    f.live_log_e = f.max_live_log_e = 20
    f.live_recent = [1.] * 100
    assert f.issue('2026-01-04T18:00:00+07:00', '2026-01-02T20:00:00+07:00')
    f.update(series)  # target actually has date Jan 2, contradicts registered Sunday
    assert not f.confidence()['validated']


def test_noise_never_gets_certified_by_retrospective_model_selection():
    f = MLForecaster('mega645', config())
    f.update(mega_series(100, seed=84))
    assert np.isfinite(f.components['main'].weights).all()
    assert f.components['main'].weights.sum() == pytest.approx(1)
    assert f.confidence()['confidence_score'] is None
    assert not f.confidence()['validated']


def test_old_date_anomalies_are_audited_without_poisoning_clean_bootstrap_window():
    series = mega_series(40)
    series.dates[1] = '2026-01-01'  # old date reversal, outside the last 16 draws
    f = MLForecaster('mega645', MLConfig(bootstrap=16, warmup=4, tree_every=16, search_nodes=1))
    assert f.update(series) == 16
    assert f.report()['date_anomalies'] == 1
    assert 'history_date_anomalies' in f.warnings
    assert not f.confidence()['validated']
    bad = mega_series(40)
    bad.dates[-1] = '2026-01-01'  # reversal in the actual fitting window
    with pytest.raises(ValueError):
        MLForecaster('mega645', MLConfig(bootstrap=16)).update(bad)


@pytest.mark.parametrize('target,made', [
    ('2026-01-03T18:00:00+07:00', '2026-01-03T17:00:00+07:00'),
    ('2026-01-04T18:30:00+07:00', '2026-01-04T17:00:00+07:00'),
    ('2026-01-04T18:00:00+07:00', '2026-01-04T17:58:00+07:00'),
])
def test_core_rejects_unverified_slot_or_near_draw_issue(target, made):
    f = MLForecaster('mega645', config())
    f.update(mega_series(2))
    assert not f.issue(target, made)
    assert f.pending is None


def test_core_uses_vietnam_target_date_for_a_utc_timestamp():
    f = MLForecaster('mega645', config())
    f.update(mega_series(2))
    assert f.issue('2026-01-04T11:00:00+00:00', '2026-01-04T10:00:00+00:00')
    assert f.pending['target_date'] == '2026-01-04'


def keno_series():
    x = np.zeros((2, 80), bool)
    x[:, :20] = True
    return Series(ProductCode.KENO, np.array([1, 2]), np.array(['2026-01-02', '2026-01-03']),
                  {'main':{'X':x, 'bonus':np.zeros(2, int)}})


def test_core_date_only_fast_issue_cannot_manufacture_live_evidence():
    f = MLForecaster('keno', config())
    f.update(keno_series().head(1))
    assert not f.issue('2026-01-03T06:00:00+07:00', '2026-01-02T23:00:00+07:00')
    f.live_scored, f.live_log_e, f.live_recent = 100, 20., [1.] * 100
    assert not f.confidence()['validated']


def test_unverified_pending_is_rejected_on_load_and_never_scored(tmp_path):
    f = MLForecaster('keno', config())
    series = keno_series()
    f.update(series.head(1))
    f.pending = {'target_id':2, 'target_date':'2026-01-03', 'target_time':'2026-01-03T06:00:00+07:00',
                 'made_at':'2026-01-02T23:00:00+07:00',
                 'laws':{name:comp.law('2026-01-03').to_dict() for name, comp in f.components.items()}}
    path = tmp_path/'state.json.gz'
    f.save(path)
    with pytest.raises(ValueError):
        MLForecaster.load(path)
    f.update(series)
    assert f.live_scored == 0 and not f.last_live_scores


def test_corrupted_replay_shape_is_rejected_during_restore(tmp_path):
    import gzip
    import json
    f = MLForecaster('mega645', config())
    f.update(mega_series(2))
    data = f.to_dict()
    data['components']['main']['frames'][0] = [[]]
    path = tmp_path/'state.json.gz'
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        MLForecaster.load(path)


@pytest.mark.parametrize('on_day,target', [(1, '2026-01-02T21:00:00+07:00'),
                                         (2, '2026-01-03T13:00:00+07:00')])
def test_core_lotto_uses_the_actual_next_daily_slot(on_day, target):
    x = np.zeros((on_day, 35), bool)
    x[:, :5] = True
    c = np.zeros((on_day, 1, 12), int)
    c[:, :, 0] = 1
    series = Series(ProductCode.LOTTO_535, np.arange(1, on_day+1), np.full(on_day, '2026-01-02'),
                    {'main':{'X':x, 'bonus':np.zeros(on_day, int)}, 'special':{'C':c}})
    f = MLForecaster('lotto535', config())
    f.update(series)
    made = '2026-01-02T14:00:00+07:00' if on_day == 1 else '2026-01-02T22:00:00+07:00'
    assert not f.issue('2026-01-03T21:00:00+07:00', made)
    assert f.issue(target, made)
