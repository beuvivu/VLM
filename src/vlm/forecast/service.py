"""Shared synchronization and update integration for the legacy and ML forecasters."""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import threading
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from vietlott_engine.core.products import get_product
from vietlott_engine.forecast.data import Series, load_series
from vietlott_engine.forecast.engine import refresh as legacy_refresh, state_path
from vietlott_engine.forecast.schedule import FAST, now_vn, record_window
from vlm.forecast.pipeline import MLConfig, MLForecaster
from vlm.updates.lease import WriterLease

_LOCK = threading.RLock()
_LOCAL = threading.local()


@contextmanager
def forecast_guard(directory: Path):
    """Thread serialization plus an OS-held lease; nested same-thread calls are safe."""
    directory = Path(directory).resolve()
    with _LOCK:
        held = getattr(_LOCAL, 'held', set())
        if directory in held:
            yield
            return
        with WriterLease(directory / 'forecast.lock'):
            _LOCAL.held = held | {directory}
            try:
                yield
            finally:
                _LOCAL.held = held


def model_config(settings) -> MLConfig:  # type: ignore[no-untyped-def]
    return MLConfig(bootstrap=settings.ml_bootstrap, tree_every=settings.ml_tree_every,
                    backends=tuple(settings.ml_backends), search_nodes=settings.ml_search_nodes)


def directory_for(state) -> Path:  # type: ignore[no-untyped-def]
    return Path(state.settings.forecast_dir or state.settings.data_dir / 'forecast')


def _read_events(path: Path) -> list[dict]:
    """Only an unterminated, incomplete final JSON object is safely recoverable."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines, events, prefix = raw.splitlines(keepends=True), [], bytearray()
    for index, line in enumerate(lines):
        if not line.strip():
            prefix.extend(line)
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b'\n'):
                raise
            suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            quarantine = path.with_name(path.name + '.' + suffix + '.corrupt')
            with quarantine.open('wb') as out:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
            name = None
            try:
                with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as out:
                    name = out.name
                    out.write(prefix)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, path)
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
            return events
        if not isinstance(event, dict) or not all(k in event for k in ('event', 'product', 'target_id')):
            raise ValueError('Invalid ML ledger event')
        events.append(event)
        prefix.extend(line)
    if raw and not raw.endswith(b'\n'):
        with path.open('ab') as out:
            out.write(b'\n')
            out.flush()
            os.fsync(out.fileno())
    return events


def _event(directory: Path, entry: dict) -> dict:
    """Immutable audit events. Replayed scores/issues use the original event once."""
    path = directory / 'ml-ledger.jsonl'
    identity = (entry['event'], entry['product'], entry['target_id'], entry.get('history_sha256'))
    for old in _read_events(path):
        if (old['event'], old['product'], old['target_id'], old.get('history_sha256')) == identity:
            return old
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        file.flush()
        os.fsync(file.fileno())
    return entry


def refresh_snapshot(product: str, series: Series, directory: Path, *, config: MLConfig | None = None,
                     refit: bool = False, issue: bool = True, include_legacy: bool = False,
                     require_config_match: bool = False):
    """Update a frozen history snapshot; callers capture it while holding the sync lock."""
    code, directory = get_product(product), Path(directory)
    with forecast_guard(directory):
        path = directory / 'ml' / f'{code.value}.json.gz'
        recovery = False
        if path.exists() and not refit:
            try:
                model = MLForecaster.load(path)
            except (ValueError, KeyError, TypeError, OSError, EOFError, gzip.BadGzipFile):
                # Keep failed bytes for diagnosis; a failed restore never keeps its scores.
                suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                path.replace(path.with_name(path.name + '.' + suffix + '.invalid'))
                model, recovery = MLForecaster(code, config), True
        else:
            model = MLForecaster(code, config)
        if model.product != code:
            raise ValueError('Checkpoint belongs to another product')
        if require_config_match and config is not None and model.config != config:
            raise ValueError('config_change_requires_fit')
        if recovery:
            model.warnings.append('checkpoint_recovered')
        # A delayed thread must not overwrite a checkpoint that already learned a
        # newer snapshot. Deliberate rollback is available only through explicit fit.
        if not refit and model.last_id is not None and len(series) and int(series.draw_ids[-1]) < model.last_id:
            return model, 0, {'enabled':True, 'learned_draws':0, 'last_id':model.last_id,
                              'issued':False, 'note':'stale_snapshot_ignored', 'error':None}
        previous_components = model.components
        learned = model.update(series, now=now_vn())
        for score in model.last_live_scores:
            _event(directory, {'event':'score', 'product':code.value, **score})
        verified, note, target = record_window(code, series.dates)
        if code in FAST:
            verified, note = False, 'date_only: chưa xác minh thời điểm từng kỳ để đăng ký bằng chứng live'
        issued = False
        if os.environ.get('VQE_FORECAST_NOW'):
            model.warnings = list(dict.fromkeys([*model.warnings, 'simulation_clock']))
            verified, note = False, 'simulation_clock: không đăng ký bằng chứng live'
        if issue and verified and target is not None:
            issued = model.issue(target.isoformat(), now_vn().isoformat())
            if issued:
                event = _event(directory, {'event':'issue', 'product':code.value,
                    'target_id':model.pending['target_id'], 'history_sha256':model.prefix_hash,
                    'pending':model.pending, 'config':model.config.model_dump(mode='json')})
                model.pending = event['pending']  # preserve the first issued distribution
                if not model.pending_valid():
                    raise ValueError('Unverified issued ledger event')
        model.save(path)
        legacy = None
        if include_legacy and state_path(directory, code).exists():
            try:
                _, count, scored = legacy_refresh(code, series, directory,
                    refit=previous_components is not model.components)
                legacy = {'learned_draws':count, 'scored_forecasts':len(scored)}
            except Exception as exc:
                legacy = {'error':type(exc).__name__}
        summary = {'enabled':True, 'learned_draws':learned, 'draws_learned':model.learned,
                   'last_id':model.last_id, 'issued':issued, 'note':note,
                   'validated':model.confidence()['validated'], 'warnings':model.warnings,
                   'legacy':legacy, 'error':None}
        return model, learned, summary


async def refresh_models(state, product: str) -> dict:  # type: ignore[no-untyped-def]
    """Keep CPU fitting off the event loop. The caller owns the product sync lock."""
    code = get_product(product)
    directory = directory_for(state)
    def learn_latest():
        with forecast_guard(directory):
            series = load_series(code, repository=state.repository,
                store=state.product_store(), seed_dir=state.settings.product_seed_dir)
            _, _, summary = refresh_snapshot(code.value, series, directory,
                config=model_config(state.settings), include_legacy=True)
            return summary
    return await asyncio.to_thread(learn_latest)
