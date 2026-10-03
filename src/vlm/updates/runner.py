"""Bounded, independent updates with an atomic restart checkpoint."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path

from vietlott_engine.forecast.schedule import VN
from vlm.updates.schedule import LIVE_PRODUCTS, freshness, poll_interval, slot_key, vietnam_time


class UpdateRunner:
    def __init__(self, path: Path, update: Callable[[str], Awaitable[dict]], *,
                 products: tuple[str, ...] = LIVE_PRODUCTS, clock: Callable[[], datetime] | None = None,
                 timeout_s: float = 180, concurrency: int = 7,
                 learn: Callable[[str], Awaitable[dict]] | None = None) -> None:
        if timeout_s <= 0 or concurrency < 1 or any(p not in LIVE_PRODUCTS for p in products):
            raise ValueError('Invalid updater timeout, concurrency or live product')
        self.path, self.update, self.products = Path(path), update, tuple(dict.fromkeys(products))
        self.clock = clock or (lambda: datetime.now(VN))
        self.timeout_s = timeout_s
        self.learn = learn
        self._slots = asyncio.Semaphore(concurrency)
        self._inflight: dict[str, asyncio.Task] = {}
        self._state = {'version': 1, 'timezone': 'Asia/Ho_Chi_Minh', 'products': {}}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version') != 1 or not isinstance(data.get('products'), dict):
                raise ValueError('Invalid updater checkpoint')
            for product, entry in data['products'].items():
                if product not in LIVE_PRODUCTS or not isinstance(entry, dict):
                    continue
                if entry.get('next_attempt'):
                    vietnam_time(datetime.fromisoformat(entry['next_attempt']))
                self._state['products'][product] = entry

    def status(self) -> dict:
        result = copy.deepcopy(self._state)
        now = vietnam_time(self.clock())
        for product, entry in result['products'].items():
            entry['freshness'] = self._freshness(product, entry, now)
        return result

    @staticmethod
    def _freshness(product: str, entry: dict, now: datetime) -> dict:
        result = freshness(product, now, entry.get('last_draw_date'), entry.get('draws_on_last_date') or 0)
        # Date-only fast-game histories cannot prove the latest intraday slot.
        # Monitor actual ID progress instead, allowing a startup/publication grace.
        progress = entry.get('last_draw_advance') or entry.get('first_polled_at')
        minute = now.hour * 60 + now.minute
        if product in ('keno', 'bingo18') and result['status'] == 'date_only' and progress and 360 <= minute < 1314:
            baseline = max(vietnam_time(datetime.fromisoformat(progress)), now.replace(hour=6, minute=0, second=0, microsecond=0))
            if now - baseline >= timedelta(minutes=20):
                result.update(status='not_advancing', verified=False,
                              note='No observed draw-ID advance for 20 minutes during selling hours; check source delay.')
        return result

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=self.path.parent, suffix='.tmp', delete=False) as f:
                name = f.name
                json.dump(self._state, f, ensure_ascii=False, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    async def tick(self, *, force: bool = False, wait: bool = True) -> dict:
        now = vietnam_time(self.clock())
        due = []
        for product in self.products:
            if product in self._inflight:
                continue
            entry = self._state['products'].get(product, {})
            next_at = datetime.fromisoformat(entry['next_attempt']) if entry.get('next_attempt') else None
            if force or next_at is None or now >= vietnam_time(next_at) or entry.get('slot') != slot_key(product, now):
                task = asyncio.create_task(self._tracked_attempt(product), name=f'vlm-update-{product}')
                self._inflight[product] = task
                due.append(task)
        if wait and due:
            await asyncio.gather(*due)
        return self.status()

    async def _tracked_attempt(self, product: str) -> None:
        try:
            await self._attempt(product)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._state['worker_error'] = type(exc).__name__
        finally:
            self._inflight.pop(product, None)

    async def stop(self) -> None:
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _attempt(self, product: str) -> None:
        async with self._slots:
            now = vietnam_time(self.clock())
            entry = self._state['products'].setdefault(product, {})
            entry.setdefault('first_polled_at', now.isoformat())
            entry.update(last_attempt=now.isoformat(), slot=slot_key(product, now))
            try:
                result = await asyncio.wait_for(self.update(product), self.timeout_s)
                if result.get('error'):
                    raise RuntimeError('Source did not return a usable result')
                if result.get('last_draw_date') and result['last_draw_date'][:10] > now.date().isoformat():
                    raise ValueError('Future result date')
                previous_id = entry.get('last_draw_id')
                if result.get('last_draw_id') is not None and (previous_id is None or result['last_draw_id'] > previous_id):
                    entry['last_draw_advance'] = vietnam_time(self.clock()).isoformat()
                entry.update({key: result.get(key) for key in
                              ('last_draw_id', 'last_draw_date', 'draws_on_last_date', 'inserted', 'source', 'prize_error', 'learning')})
                entry.update(last_success=vietnam_time(self.clock()).isoformat(), failures=0, error=None)
                entry['freshness'] = self._freshness(product, entry, vietnam_time(self.clock()))
                if result.get('inserted', 0) > 0:
                    entry['last_new_result'] = entry['last_success']
                delay = poll_interval(product, now)
                if entry['freshness']['status'] in ('behind_schedule', 'not_advancing', 'invalid_schedule'):
                    delay = min(delay, 300)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures = int(entry.get('failures', 0)) + 1
                entry.update(failures=failures, error=type(exc).__name__)
                delay = min(300, 30 * 2 ** min(failures - 1, 4))
            entry['next_attempt'] = (vietnam_time(self.clock()) + timedelta(seconds=delay)).isoformat()
            self._state['updated_at'] = vietnam_time(self.clock()).isoformat()
            self._save()
            # Persist source success before independent model work. Its deadline
            # and errors must never turn a saved draw into a source failure.
            if self.learn is not None and entry.get('error') is None:
                entry['learning'] = {'enabled':True, 'status':'running', 'error':None}
                self._save()
                try:
                    entry['learning'] = await self.learn(product)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    entry['learning'] = {'enabled':True, 'error':type(exc).__name__}
                self._state['updated_at'] = vietnam_time(self.clock()).isoformat()
                self._save()

    async def run_forever(self) -> None:
        try:
            while True:
                await self.tick(wait=False)
                await asyncio.sleep(15)
        finally:
            await self.stop()
