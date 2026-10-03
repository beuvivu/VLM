import asyncio
import json
from datetime import datetime, timedelta

import pytest


@pytest.mark.parametrize('product,now,want', [
    ('mega645','2026-10-03T19:00:00+07:00','2026-10-02T18:30:00+07:00'),
    ('power655','2026-10-03T18:29:59+07:00','2026-10-01T18:30:00+07:00'),
    ('power655','2026-10-03T11:30:00+00:00','2026-10-03T18:30:00+07:00'),
    ('max3d','2026-10-05T18:30:00+07:00','2026-10-05T18:30:00+07:00'),
    ('max3dpro','2026-10-04T01:00:00+07:00','2026-10-03T18:30:00+07:00'),
    ('lotto535','2026-10-03T13:09:59+07:00','2026-10-02T21:10:00+07:00'),
    ('lotto535','2026-10-03T21:10:00+07:00','2026-10-03T21:10:00+07:00'),
])
def test_completed_slots_respect_draw_day_delay_and_vietnam_time(product,now,want):
    from vlm.updates.schedule import latest_completed_slot
    assert latest_completed_slot(product,datetime.fromisoformat(now)).isoformat()==want


@pytest.mark.parametrize('product,now,want', [
    ('keno','2026-10-03T06:00:00+07:00',120),
    ('bingo18','2026-10-03T22:14:59+07:00',120),
    ('keno','2026-10-03T22:15:00+07:00',3600),
    ('power655','2026-10-03T18:00:00+07:00',120),
    ('mega645','2026-10-03T18:30:00+07:00',3600),
    ('lotto535','2026-10-03T21:30:00+07:00',120),
])
def test_poll_only_uses_each_products_live_window(product,now,want):
    from vlm.updates.schedule import poll_interval
    assert poll_interval(product,datetime.fromisoformat(now))==want


def test_date_only_fast_results_and_one_lotto_draw_cannot_certify_current_results():
    from vlm.updates.schedule import freshness
    now=datetime.fromisoformat('2026-10-03T22:00:00+07:00')
    assert freshness('keno',now,'2026-10-03',100)['verified'] is False
    assert freshness('lotto535',now,'2026-10-03',1)['status']=='behind_schedule'
    assert freshness('lotto535',now,'2026-10-03',2)['verified'] is True
    assert freshness('power655',now,'2026-10-01',1)['status']=='behind_schedule'


def test_runner_failure_isolated_persisted_and_retried_after_restart(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        now=datetime.fromisoformat('2026-10-03T18:35:00+07:00')
        async def update(product):
            if product=='power655': raise RuntimeError('secret proxy password')
            return {'last_draw_id':1,'last_draw_date':'2026-10-03','draws_on_last_date':4,'inserted':1,'source':'test'}
        path=tmp_path/'updates.json'
        r=UpdateRunner(path,update,products=('power655','bingo18'),clock=lambda:now)
        await r.tick()
        status=json.loads(path.read_text())['products']
        assert status['power655']['failures']==1
        assert status['power655']['error']=='RuntimeError'
        assert status['bingo18']['last_draw_id']==1
        assert status['bingo18']['freshness']['verified'] is False
        assert 'secret' not in path.read_text()
        now+=timedelta(seconds=30)
        restarted=UpdateRunner(path,update,products=('power655','bingo18'),clock=lambda:now)
        await restarted.tick()
        assert restarted.status()['products']['power655']['failures']==2
        assert not list(tmp_path.glob('*.tmp'))
    asyncio.run(run())


def test_stalled_source_cannot_starve_another_product_and_overlapping_ticks_skip(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        async def update(product):
            if product=='keno': await asyncio.Event().wait()
            return {'last_draw_id':99,'last_draw_date':'2026-10-03','draws_on_last_date':2,'inserted':1}
        r=UpdateRunner(tmp_path/'status.json',update,products=('keno','lotto535'),timeout_s=0.02,
                       clock=lambda:datetime.fromisoformat('2026-10-03T21:30:00+07:00'))
        await asyncio.gather(r.tick(),r.tick())
        status=r.status()['products']
        assert status['keno']['failures']==1
        assert status['keno']['error']=='TimeoutError'
        assert status['lotto535']['last_draw_id']==99
        assert status['lotto535']['freshness']['verified'] is True
    asyncio.run(run())


def test_empty_success_is_stale_and_new_due_slot_overrides_old_next_attempt(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        now=datetime.fromisoformat('2026-10-03T13:09:00+07:00')
        async def update(product): return {'last_draw_id':9,'last_draw_date':'2026-10-02','draws_on_last_date':2,'inserted':0}
        r=UpdateRunner(tmp_path/'status.json',update,products=('lotto535',),clock=lambda:now)
        await r.tick()
        before=r.status()['products']['lotto535']['last_attempt']
        now+=timedelta(minutes=1)
        await r.tick()
        entry=r.status()['products']['lotto535']
        assert entry['last_attempt']!=before
        assert entry['freshness']['status']=='behind_schedule'
    asyncio.run(run())


def test_overdue_weekly_result_is_retried_within_five_minutes_after_live_window(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        now = datetime.fromisoformat('2026-10-03T20:00:00+07:00')
        async def update(product):
            return {'last_draw_id':1, 'last_draw_date':'2026-10-01', 'draws_on_last_date':1, 'inserted':0}
        r = UpdateRunner(tmp_path / 'status.json', update, products=('power655',), clock=lambda:now)
        entry = (await r.tick())['products']['power655']
        assert entry['freshness']['status'] == 'behind_schedule'
        assert datetime.fromisoformat(entry['next_attempt']) - now <= timedelta(minutes=5)
    asyncio.run(run())


def test_api_starts_periodic_updater_and_cancels_it_before_closing_storage(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from vietlott_engine.api.main import create_app
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.updates.runner import UpdateRunner
    stopped=[]
    async def run_forever(self):
        try: await asyncio.Event().wait()
        finally: stopped.append(True)
    monkeypatch.setattr(UpdateRunner,'run_forever',run_forever)
    app=create_app(Settings(storage_backend='memory',data_dir=tmp_path,auto_update_enabled=True),repository=InMemoryRepository())
    with TestClient(app) as client:
        status=client.get('/updates/status')
        assert status.status_code==200
        assert status.json()['enabled'] is True
        assert status.json()['timezone']=='Asia/Ho_Chi_Minh'
    assert stopped==[True]


def test_sync_service_saves_results_and_invalidates_api_cache(tmp_path,monkeypatch):
    from contextlib import asynccontextmanager
    from fastapi.testclient import TestClient
    from vietlott_engine.api.main import create_app
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.updates.service import update_product
    import vlm.updates.service as service
    import httpx
    def handler(request):
        if request.url.path.endswith('power645.jsonl'):
            return httpx.Response(200,text='{"id":1,"date":"2026-10-02","result":[1,2,3,4,5,6]}\n')
        return httpx.Response(404)
    @asynccontextmanager
    async def client(_settings):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c: yield c
    monkeypatch.setattr(service,'build_http_client',client)
    settings=Settings(storage_backend='memory',data_dir=tmp_path,source='github_mirror',fallback_order=['github_mirror'],
                      auto_update_enabled=False)
    app=create_app(settings,repository=InMemoryRepository())
    with TestClient(app) as c:
        assert c.get('/games/mega645/draws').status_code==409
        result=c.portal.call(update_product,app.state.vqe,'mega645')
        assert result['last_draw_id']==1
        assert result['inserted']==1
        assert c.get('/games/mega645/draws').json()['total']==1
    journal=tmp_path/'updates'/'results.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[0]['game_type']=='mega645'
    assert rows[0]['draw_id']==1
    assert rows[0]['winning_numbers']==[1,2,3,4,5,6]


def test_matrix_empty_or_timed_out_source_falls_through_to_real_new_result():
    from vietlott_engine.crawler.sources.fallback import FallbackDrawSource
    from vietlott_engine.crawler.sources.base import FetchResult
    from vietlott_engine.core.games import MEGA_645
    from vietlott_engine.core.models import Draw
    async def run():
        class Empty:
            name='stale'
            async def fetch(self,*args): return FetchResult()
        class Blocked:
            name='blocked'
            async def fetch(self,*args): await asyncio.Event().wait()
        class Fresh:
            name='fresh'
            async def fetch(self,*args):
                return FetchResult(draws=[Draw(game='mega645',draw_id=2,draw_date='2026-10-02',numbers=(1,2,3,4,5,6))])
        source=FallbackDrawSource([Empty(),Blocked(),Fresh()],source_timeout_s=0.02)
        result=await source.fetch(MEGA_645,1)
        assert result.draws[0].draw_id==2
        assert source.log.used=='fresh'
    asyncio.run(run())


def test_product_stale_source_does_not_hide_newer_fallback(tmp_path):
    from vietlott_engine.crawler.product_store import ProductStore, ProductSyncPipeline
    from vietlott_engine.core.products import ProductCode
    async def run():
        store=ProductStore(tmp_path,None)
        store.upsert(ProductCode.BINGO18,[{'id':1,'date':'2026-10-03','result':[1,2,3]}],'seed')
        class Sources(ProductSyncPipeline):
            async def fetch(self,product,source,max_pages,full):
                return [{'id':1 if source=='nhanaz' else 2,'date':'2026-10-03','result':[1,2,3]}]
        pipe=Sources(None,store,vietlott_base_url='',mirror_base_url='',canonical_base_url='',
                     fallback_order=['nhanaz','mirror'])
        report=await pipe.run(ProductCode.BINGO18)
        assert report.last_id==2
        assert report.source_used=='mirror'
    asyncio.run(run())


def test_results_journal_recovers_a_lost_database_without_duplicate_rows(tmp_path):
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.database.schema import DrawRecord
    from vlm.updates.service import _append,replay_results
    settings=Settings(data_dir=tmp_path)
    writer=AppState(settings,InMemoryRepository())
    draw=DrawRecord.from_legacy('mega645',{'id':1570,'date':'2026-10-02','result':[1,6,20,27,31,41]})
    _append(writer,[draw])
    _append(writer,[draw])
    restored=AppState(settings,InMemoryRepository())
    assert replay_results(restored)==1
    assert restored.repository.load('mega645')[0].numbers==(1,6,20,27,31,41)
    assert replay_results(restored)==1
    assert restored.repository.count('mega645')==1


def test_exclusive_writer_lease_rejects_second_process_owner(tmp_path):
    from vlm.updates.lease import WriterLease
    with WriterLease(tmp_path/'writer.lock'):
        with pytest.raises(RuntimeError,match='already running'):
            with WriterLease(tmp_path/'writer.lock'): pass
    with WriterLease(tmp_path/'writer.lock'): pass


def test_cli_status_does_not_open_or_write_a_database(tmp_path,capsys):
    from vlm.updates.cli import main
    path=tmp_path/'state'
    path.mkdir()
    (path/'status.json').write_text(json.dumps({'version':1,'products':{'power655':{'error':'TimeoutError'}}}))
    assert main(['--status','--state-dir',str(path)])==0
    assert json.loads(capsys.readouterr().out)['products']['power655']['error']=='TimeoutError'
    assert sorted(p.name for p in path.iterdir())==['status.json']


def test_one_hung_game_does_not_suppress_next_healthy_poll(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        now=datetime.fromisoformat('2026-10-03T18:00:00+07:00')
        async def update(product):
            if product=='keno': await asyncio.Event().wait()
            return {'last_draw_id':int((now.hour*3600+now.minute*60)/120),
                    'last_draw_date':'2026-10-03','draws_on_last_date':1,'inserted':1}
        r=UpdateRunner(tmp_path/'status.json',update,products=('keno','bingo18'))
        r.clock=lambda:now
        await r.tick(wait=False)
        await asyncio.sleep(0.01)
        first=r.status()['products']['bingo18']['last_draw_id']
        now+=timedelta(seconds=120)
        await r.tick(wait=False)
        await asyncio.sleep(0.01)
        assert r.status()['products']['bingo18']['last_draw_id']==first+1
        await r.stop()
    asyncio.run(run())


def test_replay_preserves_current_corrections_and_journals_the_correction(tmp_path):
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.core.config import Settings
    from vietlott_engine.core.models import Draw
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.database.schema import DrawRecord
    from vlm.updates.service import _append,replay_results
    state=AppState(Settings(data_dir=tmp_path),InMemoryRepository())
    old=DrawRecord.from_legacy('mega645',{'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6]})
    corrected=old.model_copy(update={'winning_numbers':(1,2,3,4,5,7)})
    _append(state,[old])
    state.repository.upsert([Draw(game='mega645',draw_id=1,draw_date='2026-10-02',numbers=(1,2,3,4,5,7))])
    replay_results(state)
    assert state.repository.load('mega645')[0].numbers==(1,2,3,4,5,7)
    _append(state,[corrected])
    restored=AppState(state.settings,InMemoryRepository())
    replay_results(restored)
    assert restored.repository.load('mega645')[0].numbers==(1,2,3,4,5,7)


def test_torn_final_journal_record_retains_valid_prefix_and_can_append_again(tmp_path):
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.database.schema import DrawRecord
    from vlm.updates.service import _append,replay_results,journal_path
    state=AppState(Settings(data_dir=tmp_path),InMemoryRepository())
    draw=DrawRecord.from_legacy('mega645',{'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6]})
    _append(state,[draw])
    with journal_path(state).open('a') as f: f.write('{"game_type":')
    restored=AppState(state.settings,InMemoryRepository())
    replay_results(restored)
    assert restored.repository.count('mega645')==1
    _append(restored,[draw.model_copy(update={'draw_id':2})])
    again=AppState(state.settings,InMemoryRepository())
    replay_results(again)
    assert again.repository.count('mega645')==2
    assert list(journal_path(state).parent.glob('*.corrupt'))


def test_valid_unterminated_journal_tail_is_separated_before_next_append(tmp_path):
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.database.schema import DrawRecord
    from vlm.updates.service import _append, replay_results, journal_path
    state = AppState(Settings(data_dir=tmp_path), InMemoryRepository())
    draw = DrawRecord.from_legacy('mega645', {'id':1, 'date':'2026-10-02', 'result':[1,2,3,4,5,6]})
    _append(state, [draw])
    path = journal_path(state)
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    restored = AppState(state.settings, InMemoryRepository())
    assert replay_results(restored) == 1
    _append(restored, [draw.model_copy(update={'draw_id':2})])
    again = AppState(state.settings, InMemoryRepository())
    assert replay_results(again) == 2
    assert [d.draw_id for d in again.repository.load('mega645')] == [1,2]


@pytest.mark.parametrize('product', ['mega645', 'bingo18'])
def test_sync_service_journals_corrections_to_older_existing_draws(tmp_path,monkeypatch,product):
    from contextlib import asynccontextmanager
    import httpx
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.core.config import Settings
    from vietlott_engine.core.games import get_game
    from vietlott_engine.core.models import Draw
    from vietlott_engine.core.products import ProductCode
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.database.schema import DrawRecord
    import vlm.updates.service as service
    numbers = [1,2,3,4,5,6] if product == 'mega645' else [1,2,3]
    old = {'id':1, 'date':'2026-09-30', 'result':numbers}
    latest = old | {'id':2, 'date':'2026-10-02'}
    corrected = old | {'result':numbers[:-1] + [7 if product == 'mega645' else 4]}
    settings = Settings(data_dir=tmp_path, product_seed_dir=tmp_path / 'empty-seed')
    state = AppState(settings, InMemoryRepository())
    if product == 'mega645':
        state.repository.upsert([Draw(game=product,draw_id=r['id'],draw_date=r['date'],numbers=r['result']) for r in (old,latest)])
    else:
        state.product_store().upsert(ProductCode(product), [old,latest], 'seed')
    service._append(state, [DrawRecord.from_legacy(product,r) for r in (old,latest)])
    @asynccontextmanager
    async def client(_settings):
        transport = httpx.MockTransport(lambda request:httpx.Response(200,text=''.join(json.dumps(r)+'\n' for r in (corrected,latest))))
        async with httpx.AsyncClient(transport=transport) as c: yield c
    monkeypatch.setattr(service, 'build_http_client', client)
    if product == 'mega645':
        asyncio.run(service.sync_matrix(state, get_game(product), source='github_mirror', full_refresh=True))
    else:
        asyncio.run(service.sync_product(state, ProductCode(product), source='mirror', full=True))
    restored = AppState(settings, InMemoryRepository())
    # Simulate cache/database loss for the product store as well.
    if product == 'bingo18':
        for path in (tmp_path / 'products').iterdir(): path.unlink()
    service.replay_results(restored)
    values = list(restored.repository.load(product)[0].numbers) if product == 'mega645' else restored.product_store().load(product).values[0].tolist()
    assert values == corrected['result']


def test_off_schedule_weekly_result_is_never_certified_fresh():
    from vlm.updates.schedule import freshness
    r=freshness('mega645',datetime.fromisoformat('2026-10-03T19:00:00+07:00'),'2026-10-03',1)
    assert r['verified'] is False
    assert r['status']=='invalid_schedule'


def test_fast_game_no_id_advance_for_twenty_minutes_is_reported(tmp_path):
    from vlm.updates.runner import UpdateRunner
    async def run():
        now=datetime.fromisoformat('2026-10-03T08:00:00+07:00')
        async def update(product):
            return {'last_draw_id':77,'last_draw_date':'2026-10-03','draws_on_last_date':1,'inserted':0}
        r=UpdateRunner(tmp_path/'status.json',update,products=('keno',),clock=lambda:now)
        await r.tick()
        now+=timedelta(minutes=20)
        await r.tick()
        entry=r.status()['products']['keno']
        assert entry['freshness']['status']=='not_advancing'
        assert entry['freshness']['verified'] is False
    asyncio.run(run())


@pytest.mark.parametrize('filename', ['draws.jsonl', 'draws.jsonl.gz'])
def test_product_atomic_write_failure_keeps_existing_results(tmp_path,monkeypatch,filename):
    from vietlott_engine.core.products import write_jsonl,read_jsonl
    import os
    path=tmp_path/filename
    old={'id':1,'date':'2026-10-03','result':[1,2,3]}
    write_jsonl(path,[old])
    def blocked(*args): raise OSError('disk unavailable')
    monkeypatch.setattr(os,'replace',blocked)
    with pytest.raises(OSError): write_jsonl(path,[old,old|{'id':2}])
    assert read_jsonl(path)==[old]
    assert sorted(p.name for p in tmp_path.iterdir())==[filename]


def test_api_failed_startup_releases_storage_and_writer_lease(tmp_path):
    from fastapi.testclient import TestClient
    from vietlott_engine.api.main import create_app
    from vietlott_engine.core.config import Settings
    from vietlott_engine.crawler.storage import InMemoryRepository
    from vlm.updates.lease import WriterLease
    class Repository(InMemoryRepository):
        closed = False
        def close(self): self.closed = True
    repo = Repository()
    directory = tmp_path / 'updates'
    directory.mkdir()
    (directory / 'results.jsonl').write_text('{bad}\n')
    app = create_app(Settings(storage_backend='memory', data_dir=tmp_path, auto_update_enabled=True), repo)
    with pytest.raises(ValueError):
        with TestClient(app): pass
    with WriterLease(directory / 'writer.lock'): pass
    assert repo.closed


@pytest.mark.parametrize('worker_error', [False, True])
def test_cli_once_returns_failure_for_stale_results_or_failed_checkpoint(tmp_path,monkeypatch,capsys,worker_error):
    from vietlott_engine.core.config import Settings
    from vlm.updates.cli import main
    from vlm.updates.runner import UpdateRunner
    import vlm.updates.cli as cli
    import vlm.updates.service as service
    settings = Settings(storage_backend='memory', data_dir=tmp_path, seed_file_dir=None)
    monkeypatch.setattr(cli, 'get_settings', lambda: settings)
    async def update(product):
        return {'last_draw_id':1, 'last_draw_date':'2026-10-03' if worker_error else '2026-10-01',
                'draws_on_last_date':1, 'inserted':0}
    def make(state):
        runner = UpdateRunner(tmp_path / 'updates' / 'status.json', update, products=('power655',),
                              clock=lambda:datetime.fromisoformat('2026-10-03T19:00:00+07:00'))
        if worker_error:
            def save(): raise OSError('secret disk path')
            monkeypatch.setattr(runner, '_save', save)
        return runner
    monkeypatch.setattr(service, 'make_runner', make)
    assert main(['--once', '--product', 'power655']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['products']['power655']['freshness']['status'] == ('caught_up' if worker_error else 'behind_schedule')
    if worker_error: assert result['worker_error'] == 'OSError'
    else: assert (tmp_path / 'updates' / 'status.json').exists()
    assert 'secret' not in json.dumps(result)
