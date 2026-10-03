"""Shared API/CLI update operations; the API remains the storage owner."""
from __future__ import annotations

import asyncio
import os
import tempfile
from collections import defaultdict
from datetime import datetime

from vietlott_engine.core.games import GAMES, get_game
from vietlott_engine.core.models import Draw
from vietlott_engine.core.products import ProductCode, history_to_rows
from vietlott_engine.crawler.pipeline import SyncPipeline, build_http_client, build_source
from vietlott_engine.crawler.product_store import ProductSyncPipeline, sync_canonical_prizes
from vietlott_engine.forecast.schedule import VN
from vlm.database.schema import DrawRecord
from pydantic import ValidationError


def journal_path(state):  # type: ignore[no-untyped-def]
    return (state.settings.auto_update_dir or state.settings.data_dir / 'updates') / 'results.jsonl'


def _read_journal(path) -> list[DrawRecord]:  # type: ignore[no-untyped-def]
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, prefix = [], bytearray()
    for index, line in enumerate(lines):
        if not line.strip():
            prefix.extend(line)
            continue
        try:
            records.append(DrawRecord.model_validate_json(line))
        except ValidationError as exc:
            # Only a torn unterminated JSON tail is recoverable automatically.
            # A malformed complete/middle record remains an explicit error.
            if index != len(lines)-1 or line.endswith(b'\n') or not any(e['type']=='json_invalid' for e in exc.errors()):
                raise
            quarantine = path.with_name(path.name + '.' + datetime.now(VN).strftime('%Y%m%dT%H%M%S%f') + '.corrupt')
            quarantine.write_bytes(line)
            with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as out:
                name = out.name
                out.write(prefix)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            return records
        prefix.extend(line)
    if raw and not raw.endswith(b'\n'):
        # A crash may leave complete JSON with only the delimiter missing.
        # Separate it before a later append can concatenate two JSON objects.
        with path.open('ab') as out:
            out.write(b'\n')
            out.flush()
            os.fsync(out.fileno())
    return records


def _identity(record: DrawRecord) -> tuple:
    return (record.draw_date.date(), record.winning_numbers, record.bonus_number)


def _journal_index(state) -> dict:  # type: ignore[no-untyped-def]
    if state._journal_seen is None:
        state._journal_seen = {(r.game_type,r.draw_id):_identity(r) for r in _read_journal(journal_path(state))}
    return state._journal_seen


def _product_baseline(history) -> dict:  # type: ignore[no-untyped-def]
    # History has already been validated. Compare its compact values without
    # rebuilding hundreds of thousands of Pydantic models on every fast poll.
    return {int(i):(str(day),tuple(values.tolist()))
            for i,day,values in zip(history.draw_ids,history.dates,history.values)}


def _product_records(code, history, baseline: dict, journal_ids: set[int], source: str) -> list[DrawRecord]:  # type: ignore[no-untyped-def]
    newest = str(history.dates.max()) if len(history) else None
    records = []
    for i, day, values in zip(history.draw_ids,history.dates,history.values):
        identity = (str(day),tuple(values.tolist()))
        if baseline.get(int(i)) != identity or identity[0] == newest or int(i) in journal_ids:
            records.append(DrawRecord.from_legacy(code.value, {'id':int(i), 'date':str(day),
                                                              'result':values.tolist(), 'source':source}))
    return records


def _append(state, records: list[DrawRecord], *, baseline: dict | None = None) -> None:  # type: ignore[no-untyped-def]
    if not records:
        return
    path = journal_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    _journal_index(state)
    if baseline is not None:
        newest = max(r.draw_date.date() for r in records)
        # Track old-result corrections as well as new IDs. Known journal IDs are
        # rechecked after failed appends; unchanged bundled history stays in seed.
        records = [r for r in records if baseline.get((r.game_type,r.draw_id)) != _identity(r)
                   or r.draw_date.date() == newest or (r.game_type,r.draw_id) in state._journal_seen]
    records = [r for r in records if state._journal_seen.get((r.game_type,r.draw_id)) != _identity(r)]
    if not records:
        return
    payload = ''.join(r.model_dump_json() + '\n' for r in records)
    # The event loop owns these writes. Append + fsync protects completed results
    # separately from replaceable checkpoints/caches.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, 'a', encoding='utf-8') as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
    except OSError:
        state._journal_seen = None  # re-read/repair a possible torn tail on retry
        raise
    state._journal_seen.update(((r.game_type,r.draw_id),_identity(r)) for r in records)


def _matrix_record(draw: Draw) -> DrawRecord:
    return DrawRecord.from_legacy(draw.game.value, {'id':draw.draw_id, 'date':draw.draw_date.isoformat(),
        'result':list(draw.numbers), 'bonus_number':draw.bonus, 'source':draw.source,
        'jackpot1_value':draw.jackpot1_value, 'jackpot2_value':draw.jackpot2_value})


def replay_results(state) -> int:  # type: ignore[no-untyped-def]
    """Recover committed draw results when a database/cache was lost."""
    path = journal_path(state)
    if not path.exists():
        return 0
    records = {(r.game_type,r.draw_id):r for r in _read_journal(path)}
    products = defaultdict(list)
    matrix = []
    existing = {spec.code.value:{d.draw_id:d for d in state.repository.load(spec.code)} for spec in GAMES.values()}
    for r in records.values():
        if r.game_type in GAMES:
            current = existing[r.game_type].get(r.draw_id)
            if current is not None:
                if _identity(_matrix_record(current)) != _identity(r):
                    state.replay_conflicts.append({'game_type':r.game_type, 'draw_id':r.draw_id})
                continue
            matrix.append(Draw(game=r.game_type, draw_id=r.draw_id, draw_date=r.draw_date.date(),
                               numbers=tuple(r.winning_numbers), bonus=r.bonus_number, source=r.source,
                               jackpot1_value=r.jackpot1_value, jackpot2_value=r.jackpot2_value))
        else:
            products[ProductCode(r.game_type)].append({'id':r.draw_id, 'date':r.draw_date.date().isoformat(),
                                                      'result':list(r.winning_numbers)})
    if matrix:
        state.repository.upsert(matrix)
        for product in {d.game for d in matrix}:
            state.invalidate(get_game(product))
    for product, rows in products.items():
        store = state.product_store()
        current = {row['id']:row for row in history_to_rows(store.load(product, include_unconfirmed=True))}
        missing = []
        for row in rows:
            if row['id'] not in current:
                missing.append(row)
            elif current[row['id']]['date'] != row['date'] or [int(n) for n in current[row['id']]['result']] != [int(n) for n in row['result']]:
                state.replay_conflicts.append({'game_type':product.value,'draw_id':row['id']})
        store.upsert(product, missing, 'updates-journal')
        state.invalidate_product(product)
    return len(records)


async def sync_matrix(state, spec, *, source: str | None = None, full_refresh: bool = False,
                      learn: bool = True) -> dict:  # type: ignore[no-untyped-def]
    async with state.sync_lock(spec.code.value):
        before = {(d.game.value,d.draw_id):_identity(_matrix_record(d)) for d in state.repository.load(spec.code)}
        async with build_http_client(state.settings) as client:
            report = await SyncPipeline(build_source(state.settings, client, source), state.repository).run(spec, full_refresh)
        draws = state.repository.load(spec.code)
        _append(state, [_matrix_record(d) for d in draws], baseline=before)
        state.invalidate(spec)
    return {**report.to_dict(), 'learning':await _learn(state, spec.code.value) if learn else None}


async def sync_product(state, code, *, source: str = 'auto', max_pages: int | None = None, full: bool = False,
                       learn: bool = True) -> dict:  # type: ignore[no-untyped-def]
    async with state.sync_lock(code.value):
        s, store = state.settings, state.product_store()
        history = await asyncio.to_thread(store.load, code, include_unconfirmed=True)
        before = await asyncio.to_thread(_product_baseline, history)
        async with build_http_client(s) as client:
            pipe = ProductSyncPipeline(client, store, vietlott_base_url=s.vietlott_base_url,
                mirror_base_url=s.product_mirror_base_url, canonical_base_url=s.canonical_base_url,
                bootstrap_cookie=s.vietlott_cookie_bootstrap, pages_per_batch=s.max_concurrency,
                nhanaz_base_url=s.nhanaz_base_url, nhanaz_dir=s.nhanaz_dir, v130_dir=s.v130_dir,
                fallback_order=s.fallback_order, source_timeout_s=s.source_timeout_s)
            report = await pipe.run(code, source, max_pages or s.product_sync_max_pages, full)
        h = await asyncio.to_thread(store.load, code)
        journal_ids = {i for (product,i) in _journal_index(state) if product == code.value}
        records = await asyncio.to_thread(_product_records, code, h, before, journal_ids, report.source_used or source)
        _append(state, records)
        state.invalidate_product(code)
    return {**report.to_dict(), 'learning':await _learn(state, code.value) if learn else None}


async def _learn(state, product: str) -> dict:  # type: ignore[no-untyped-def]
    if not state.settings.ml_auto_update_enabled:
        return {'enabled':False, 'error':None}
    from vlm.forecast.service import refresh_models
    job = state._ml_jobs.get(product)
    if job is None:
        async def guarded_refresh():
            async with state.sync_lock(product):
                return await refresh_models(state, product)
        job = asyncio.create_task(guarded_refresh(), name=f'vlm-learn-{product}')
        state._ml_jobs[product] = job
        def done(task):
            if state._ml_jobs.get(product) is task:
                state._ml_jobs.pop(product, None)
            if not task.cancelled():
                task.exception()  # consume an error even after the caller timed out
        job.add_done_callback(done)
    try:
        return await asyncio.wait_for(asyncio.shield(job), state.settings.ml_learning_timeout_s)
    except TimeoutError:
        # A Python fitting thread cannot be killed safely. Keep it tracked and
        # drain it before closing storage; no duplicate job is queued on retry.
        return {'enabled':True, 'error':'TimeoutError', 'pending':True}
    except Exception as exc:
        return {'enabled':True, 'error':type(exc).__name__}


async def finish_learning(state) -> None:  # type: ignore[no-untyped-def]
    """Do not close storage while an uncancellable fitting thread still uses it."""
    if state._ml_jobs:
        await asyncio.gather(*list(state._ml_jobs.values()), return_exceptions=True)


async def update_product(state, product: str, *, learn: bool = True) -> dict:  # type: ignore[no-untyped-def]
    """Persist results first; prize-table unavailability never discards results."""
    prize_error = None
    if product in GAMES:
        spec = get_game(product)
        report = await sync_matrix(state, spec, source='auto', learn=False)
        draws = state.repository.load(spec.code)
        last = max(draws, key=lambda d:d.draw_id) if draws else None
        last_date = last.draw_date.isoformat() if last else None
        count = sum(d.draw_date == last.draw_date for d in draws) if last else 0
        # A few latest details are enough for live settlement; bulk finance is a
        # separate audit. Give this optional step a small budget.
        try:
            async with state.sync_lock(product):
                async with build_http_client(state.settings) as client:
                    s = state.settings
                    finance = await asyncio.wait_for(sync_canonical_prizes(client, state.repository, spec,
                        source='auto', last=3, canonical_base_url=s.canonical_base_url,
                        vietlott_base_url=s.vietlott_base_url, bootstrap_cookie=s.vietlott_cookie_bootstrap,
                        nhanaz_base_url=s.nhanaz_base_url, nhanaz_dir=s.nhanaz_dir, v130_dir=s.v130_dir), min(45, s.auto_update_timeout_s / 4))
                    prize_error = 'SourceError' if finance.error else None
        except Exception as exc:
            prize_error = type(exc).__name__
        return {'last_draw_id':last.draw_id if last else None, 'last_draw_date':last_date,
                'draws_on_last_date':count, 'inserted':report['inserted'], 'source':report['source'], 'prize_error':prize_error,
                'learning':await _learn(state, product) if learn else None}
    code = ProductCode(product)
    report = await sync_product(state, code, learn=False)
    h = state.product_store().load(code)
    count = int((h.dates == h.dates[-1]).sum()) if len(h) else 0
    return {'last_draw_id':report['last_id'], 'last_draw_date':report['last_date'], 'draws_on_last_date':count,
            'inserted':report['inserted'], 'source':report['source_used'], 'error':report['error'],
            'learning':await _learn(state, product) if learn else None}


def make_runner(state):  # type: ignore[no-untyped-def]
    from vlm.updates.runner import UpdateRunner
    return UpdateRunner(journal_path(state).parent / 'status.json',
                        lambda product:update_product(state, product, learn=False),
                        learn=lambda product:_learn(state, product), timeout_s=state.settings.auto_update_timeout_s)


async def periodic_updates(state) -> None:  # type: ignore[no-untyped-def]
    replay_results(state)
    state.updater = make_runner(state)
    try:
        await state.updater.run_forever()
    finally:
        await finish_learning(state)


def update_status(state) -> dict:  # type: ignore[no-untyped-def]
    status = state.updater.status() if state.updater is not None else {'products':{}}
    return {'enabled':state.settings.auto_update_enabled, 'running':state.updater is not None, 'timezone':'Asia/Ho_Chi_Minh',
            'observed_at':datetime.now(VN).isoformat(), 'replay_conflicts':state.replay_conflicts, **status}
