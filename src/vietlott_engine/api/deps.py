"""Application state and FastAPI dependencies."""

from __future__ import annotations

import threading
import asyncio
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from vietlott_engine.core.config import Settings
from vietlott_engine.core.exceptions import InsufficientDataError
from vietlott_engine.core.games import GameSpec, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.prizes import PrizeHistory
from vietlott_engine.crawler.storage import DrawRepository


@dataclass
class AppState:
    settings: Settings
    repository: DrawRepository
    _cache: dict[str, DrawHistory] = field(default_factory=dict)
    _models: dict[str, object] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _sync_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    updater: object | None = None
    _journal_seen: dict[tuple[str, int], tuple] | None = None
    replay_conflicts: list[dict] = field(default_factory=list)
    _ml_jobs: dict[str, asyncio.Task] = field(default_factory=dict)

    def sync_lock(self, product: str) -> asyncio.Lock:
        """One in-flight sync per product, shared by HTTP and periodic updates."""
        return self._sync_locks.setdefault(product, asyncio.Lock())

    def history(self, spec: GameSpec) -> DrawHistory:
        key = spec.code.value
        with self._lock:
            if key not in self._cache:
                draws = self.repository.load(spec.code)
                if not draws:
                    raise InsufficientDataError(f"no draws stored for {key}; call POST /games/{key}/sync first")
                self._cache[key] = DrawHistory.from_draws(draws, spec)
            return self._cache[key]

    def prize_history(self, spec: GameSpec) -> tuple[DrawHistory, PrizeHistory]:
        h, ph = self.repository.load_prize_history(spec.code)
        if not ph.has_winners.any():
            raise InsufficientDataError(f"no prize data for {spec.code.value}; call POST /games/{spec.code.value}/prizes/sync first")
        return h, ph

    def _cached(self, key: str, loader):  # type: ignore[no-untyped-def]
        with self._lock:
            if key not in self._models:
                self._models[key] = loader()
            return self._models[key]

    def calibration(self, spec: GameSpec):  # type: ignore[no-untyped-def]
        """Fitted crowd-behaviour model (``vietlott market``) or None."""
        from vietlott_engine.game_theory.calibration import load_calibration

        return self._cached(f"cal:{spec.code.value}", lambda: load_calibration(spec.code, self.settings.calibration_dir))

    def market(self, spec: GameSpec):  # type: ignore[no-untyped-def]
        """Market report (payout share, sales, calibration, rolldowns) or None."""
        import json

        from vietlott_engine.game_theory.market import MarketReport

        def load():  # type: ignore[no-untyped-def]
            path = self.settings.calibration_dir / f"market_{spec.code.value}.json"
            return MarketReport.model_validate(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None

        return self._cached(f"market:{spec.code.value}", load)

    def sales_model(self, spec: GameSpec):  # type: ignore[no-untyped-def]
        rep = self.market(spec)
        return rep.sales_model if rep is not None else None

    def typical_tickets_sold(self, spec: GameSpec) -> int:
        """Median estimated sales of the game (fallback when no sales model)."""
        rep = self.market(spec)
        if rep is not None and rep.tickets_sold is not None:
            return int(rep.tickets_sold.median)
        return int(self.settings.default_tickets_sold)

    def product_store(self):  # type: ignore[no-untyped-def]
        from vietlott_engine.crawler.product_store import ProductStore

        return ProductStore(self.settings.data_dir, self.settings.product_seed_dir)

    def product_history(self, code):  # type: ignore[no-untyped-def]
        """Keno / Bingo18 / Max 3D / Max 3D Pro history (bundled seed + local store)."""
        h = self._cached(f"prod:{code.value}", lambda: self.product_store().load(code))
        if not len(h):
            raise InsufficientDataError(f"no draws stored for {code.value}; call POST /products/{code.value}/sync first")
        return h

    def invalidate_product(self, code) -> None:  # type: ignore[no-untyped-def]
        with self._lock:
            for key in [k for k in self._models if k.startswith(("prod:", "rand:"))]:
                if key.endswith(code.value) or key.endswith("digit"):
                    self._models.pop(key, None)

    def invalidate(self, spec: GameSpec) -> None:
        with self._lock:
            self._cache.pop(spec.code.value, None)

    def reload_models(self) -> None:
        with self._lock:
            self._models.clear()


def get_state(request: Request) -> AppState:
    return request.app.state.vqe  # type: ignore[no-any-return]


def resolve_game(game: str) -> GameSpec:
    try:
        return get_game(game)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
