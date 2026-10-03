import asyncio
import json
from datetime import date

from fastapi.testclient import TestClient
import pytest

from vietlott_engine.api.deps import AppState
from vietlott_engine.api.main import create_app
from vietlott_engine.core.config import Settings
from vietlott_engine.core.games import MEGA_645
from vietlott_engine.core.models import Draw
from vietlott_engine.crawler.storage import InMemoryRepository
from vlm.forecast.pipeline import MLConfig, MLForecaster
from vlm.forecast.service import forecast_guard, refresh_models, refresh_snapshot
from tests.test_vlm_forecast_pipeline import mega_series, config


def test_snapshot_service_uses_first_issued_forecast_and_recovers_corruption(tmp_path):
    series = mega_series(40)
    f, learned, summary = refresh_snapshot('mega645', series, tmp_path, config=config())
    assert learned == 40 and not summary['issued']  # January history is stale in October
    path = tmp_path / 'ml' / 'mega645.json.gz'
    assert path.exists()
    path.write_bytes(b'torn state')
    f, learned, summary = refresh_snapshot('mega645', series, tmp_path, config=config())
    assert learned == 40 and 'checkpoint_recovered' in f.warnings
    assert not f.confidence()['validated']


def test_file_lock_prevents_a_second_writer(tmp_path):
    from vlm.updates.lease import WriterLease
    with WriterLease(tmp_path / 'forecast.lock'):
        with pytest.raises(RuntimeError):
            with forecast_guard(tmp_path):
                pass


def test_legacy_forecast_writer_shares_the_same_cross_process_guard(tmp_path):
    from vlm.updates.lease import WriterLease
    from vietlott_engine.forecast.engine import refresh
    with WriterLease(tmp_path / 'forecast.lock'):
        with pytest.raises(RuntimeError):
            refresh('mega645', mega_series(2), tmp_path)


def test_async_updater_learns_real_persisted_results_and_exposes_model_errors(tmp_path, monkeypatch):
    import vlm.updates.service as updater
    repo = InMemoryRepository()
    repo.upsert([Draw(game='mega645', draw_id=1, draw_date=date(2026, 10, 2), numbers=(1, 2, 3, 4, 5, 6))])
    settings = Settings(data_dir=tmp_path, product_seed_dir=tmp_path,
                        ml_auto_update_enabled=True, ml_bootstrap=16, ml_tree_every=16)
    state = AppState(settings, repo)
    async def empty_sync(state, spec, **kwargs):
        return {'inserted':1, 'source':'fixture'}
    async def no_finance(*args, **kw):
        raise RuntimeError('offline prize source')
    monkeypatch.setattr(updater, 'sync_matrix', empty_sync)
    monkeypatch.setattr(updater, 'sync_canonical_prizes', no_finance)
    result = asyncio.run(updater.update_product(state, 'mega645'))
    assert result['learning']['learned_draws'] == 1
    path = tmp_path / 'forecast' / 'ml' / 'mega645.json.gz'
    assert MLForecaster.load(path).last_id == 1
    async def broken_model(*args, **kwargs):
        raise RuntimeError('model failed')
    monkeypatch.setattr('vlm.forecast.service.refresh_models', broken_model)
    result = asyncio.run(updater.update_product(state, 'mega645'))
    assert result['last_draw_id'] == 1 and repo.count(MEGA_645.code) == 1
    assert result['learning']['error'] == 'RuntimeError'


def test_ml_api_and_budget_portfolio_are_valid_unique_tickets(tmp_path):
    repo = InMemoryRepository()
    repo.upsert([Draw(game='mega645', draw_id=i, draw_date=date(2026, 10, 2),
                      numbers=(1, 2, 3, 4, 5, 6)) for i in range(1, 21)])
    settings = Settings(data_dir=tmp_path, product_seed_dir=tmp_path, ml_bootstrap=16,
                        ml_tree_every=16, auto_update_enabled=False)
    with TestClient(create_app(settings, repo)) as client:
        response = client.get('/ml/forecast/mega645', params={'top_n':5, 'budget':35000})
        assert response.status_code == 200
        report = response.json()
        assert report['target_id'] == 21
        tickets = report['portfolio']['tickets']
        assert len(tickets) == 3 and report['portfolio']['spent_vnd'] == 30000
        assert len({tuple(t['numbers']) for t in tickets}) == 3
        assert all(len(t['numbers']) == 6 and len(set(t['numbers'])) == 6 for t in tickets)
        assert not report['confidence']['validated']
        assert client.get('/ml/forecast/nope').status_code == 404
        assert client.get('/ml/forecast/mega645', params={'top_n':101}).status_code == 422


def test_cli_input_is_validated_and_forecast_can_be_reused(tmp_path, capsys):
    from vlm.forecast.cli import main
    path = tmp_path / 'history.jsonl'
    rows = [{'id':i, 'date':'2026-10-02', 'result':[1, 2, 3, 4, 5, 6]} for i in range(1, 13)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    args = ['next', '--product','mega645', '--input',str(path), '--dir',str(tmp_path/'state'), '--top-n','3']
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report['components'][0]['top']) == 3 and report['last_id'] == 12
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['learning']['learned_draws'] == 0
    rows[0]['result'] = [1, 2, 3, 4, 5, 99]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)['error']


def test_explicit_cli_backend_change_cannot_silently_reuse_old_model(tmp_path, capsys):
    from vlm.forecast.cli import main
    path = tmp_path/'history.jsonl'
    path.write_text(json.dumps({'id':1, 'date':'2026-10-02', 'result':[1, 2, 3, 4, 5, 6]})+'\n')
    args = ['next', '--product','mega645', '--input',str(path), '--dir',str(tmp_path/'state')]
    assert main(args) == 0
    capsys.readouterr()
    assert main([*args, '--bootstrap','16']) == 1
    assert json.loads(capsys.readouterr().out)['note'].endswith('fit để đổi cấu hình đã lưu.')


def test_new_results_keep_learning_after_process_restart(tmp_path):
    series = mega_series(41)
    refresh_snapshot('mega645', series.head(40), tmp_path, config=config())
    f, learned, _ = refresh_snapshot('mega645', series, tmp_path)
    assert learned == 1 and f.last_id == 41
    assert f.config.bootstrap == 100  # persisted training configuration, not a silent reset


def test_knowing_target_is_not_enough_when_latest_fast_day_is_incomplete(tmp_path):
    # Existing schedule verifier refuses a partial last day; model state must not bypass it.
    from vietlott_engine.core.products import ProductCode
    from vietlott_engine.forecast.data import Series
    import numpy as np
    x = np.zeros((101, 80), bool)
    x[:, :20] = True
    series = Series(ProductCode.KENO, np.arange(1, 102),
                    np.array(['2026-10-01']*100 + ['2026-10-02']),
                    {'main':{'X':x, 'bonus':np.zeros(101, int)}})
    _, _, summary = refresh_snapshot('keno', series, tmp_path, config=MLConfig(bootstrap=8, search_nodes=5))
    assert not summary['issued']


def test_async_learning_does_not_block_event_loop(tmp_path):
    repo = InMemoryRepository()
    repo.upsert([Draw(game='mega645', draw_id=i, draw_date=date(2026, 10, 2),
                     numbers=(1, 2, 3, 4, 5, 6)) for i in range(1, 41)])
    state = AppState(Settings(data_dir=tmp_path, product_seed_dir=tmp_path,
                             ml_bootstrap=40, ml_tree_every=16), repo)
    async def run():
        progress = 0
        finished = False
        async def pulse():
            nonlocal progress
            for _ in range(5):
                await asyncio.sleep(0)
                if not finished:
                    progress += 1
        async def learn():
            nonlocal finished
            result = await refresh_models(state, 'mega645')
            finished = True
            return result
        result, _ = await asyncio.gather(learn(), pulse())
        assert progress == 5 and result['learned_draws'] == 40
    asyncio.run(run())


def test_even_complete_date_only_fast_data_cannot_register_a_verified_target(tmp_path, monkeypatch):
    from vietlott_engine.core.products import ProductCode
    from vietlott_engine.forecast.data import Series
    import numpy as np
    import vlm.forecast.service as service
    from datetime import datetime
    monkeypatch.setattr(service, 'record_window', lambda *a:(True, 'night window', datetime.fromisoformat('2026-10-04T06:00:00+07:00')))
    x = np.zeros((4, 80), bool)
    x[:, :20] = True
    series = Series(ProductCode.KENO, np.arange(1, 5), np.full(4, '2026-10-03'),
                    {'main':{'X':x, 'bonus':np.zeros(4, int)}})
    _, _, summary = refresh_snapshot('keno', series, tmp_path, config=MLConfig(bootstrap=4, search_nodes=1))
    assert not summary['issued']
    assert summary['note'].startswith('date_only')


@pytest.mark.parametrize('tail', [b'{"event":"issue"', b'{"note":"\xe1\xbb', b''])
def test_ml_event_ledger_recovers_a_crash_tail_and_preserves_first_issue(tmp_path, tail):
    from vlm.forecast.service import _event
    first = {'event':'issue', 'product':'mega645', 'target_id':2, 'history_sha256':'hash', 'marker':'first'}
    path = tmp_path/'ml-ledger.jsonl'
    # A complete final object can also lose just its newline at a crash.
    path.write_bytes(json.dumps(first).encode() + (b'\n'+tail if tail else b''))
    second = {**first, 'target_id':3, 'marker':'second'}
    assert _event(tmp_path, second) == second
    assert _event(tmp_path, {**first, 'marker':'replacement'}) == first
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [first, second]
    if tail:
        quarantines = list(tmp_path.glob('ml-ledger.jsonl.*.corrupt'))
        assert len(quarantines) == 1 and quarantines[0].read_bytes() == tail


def test_slow_learning_has_separate_budget_and_preserves_successful_result_status(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from datetime import datetime
    from vlm.updates.service import journal_path, make_runner
    from vietlott_engine.crawler.product_store import ProductSyncPipeline
    settings = Settings(data_dir=tmp_path, product_seed_dir=tmp_path/'empty',
                        ml_auto_update_enabled=True, auto_update_timeout_s=5,
                        fallback_order=['mirror']).model_copy(update={'ml_learning_timeout_s':.02})
    state = AppState(settings, InMemoryRepository())
    async def fetch(*args, **kwargs):
        return [{'id':1, 'date':'2026-10-03', 'result':[1, 2, 3]}]
    monkeypatch.setattr(ProductSyncPipeline, 'fetch', fetch)
    @asynccontextmanager
    async def offline_client(settings):
        yield None  # fetch is stubbed; constructing a real TLS client is unrelated
    monkeypatch.setattr('vlm.updates.service.build_http_client', offline_client)
    async def run():
        release = asyncio.Event()
        completed = []
        source_status_at_learning_start = []
        async def slow_model(*args):
            source_status_at_learning_start.append(json.loads(runner.path.read_text())['products']['bingo18'])
            await release.wait()
            completed.append(True)
            return {'enabled':True, 'learned_draws':1, 'error':None}
        monkeypatch.setattr('vlm.forecast.service.refresh_models', slow_model)
        runner = make_runner(state)
        runner.products = ('bingo18',)
        runner.clock = lambda:datetime.fromisoformat('2026-10-03T19:00:00+07:00')
        result = (await runner.tick())['products']['bingo18']
        assert result['error'] is None and result['last_draw_id'] == 1
        assert result['learning']['error'] == 'TimeoutError'
        # Source success must already be durable while the model is still blocked.
        saved = source_status_at_learning_start[0]
        assert saved['last_draw_id'] == 1 and saved['error'] is None and saved['last_success']
        assert not completed
        assert json.loads(journal_path(state).read_text())['draw_id'] == 1
        release.set()
        from vlm.updates.service import finish_learning
        await finish_learning(state)
        assert completed == [True]
    asyncio.run(run())


def test_bootstrap_limit_matches_the_thousand_draw_design():
    with pytest.raises(ValueError):
        MLConfig(bootstrap=1001)
    with pytest.raises(ValueError):
        Settings(ml_bootstrap=1001)
