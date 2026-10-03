"""Fallback when vietlott.vn is unreachable (or a source has gaps).

``FallbackDrawSource`` tries its sources in order and keeps the first that answers; every
attempt (source, error, rows) is recorded so a sync report shows *where* the data came from.
Default order for ``--source auto``:

    vietlott (direct, Vietnamese IP) → nhanaz (community archive, ~10-minute updates)
    → github_mirror (vietvudanh) → v130 (local Vietlott Quant Engine 1.3.0 snapshot, if set)

``ArchiveDrawSource`` adapts the community archive (remote or local folder) and the 1.3.0
snapshot to the ``DrawSource`` interface used by ``SyncPipeline`` for Mega / Power / Lotto.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import GameSpec
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.products import ProductCode
from vietlott_engine.crawler.official_data import import_canonical
from vietlott_engine.crawler.sources.base import DrawSource, FetchResult

log = get_logger(__name__)


@dataclass
class Attempt:
    source: str
    ok: bool
    rows: int = 0
    error: str | None = None


@dataclass
class FallbackLog:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def used(self) -> str | None:
        return next((a.source for a in self.attempts if a.ok), None)

    def to_list(self) -> list[dict]:
        return [a.__dict__ for a in self.attempts]


class ArchiveDrawSource(DrawSource):
    """Mega / Power / Lotto draws from the community archive or a 1.3.0 snapshot."""

    def __init__(self, archive=None, v130_dir=None) -> None:  # type: ignore[no-untyped-def]
        if (archive is None) == (v130_dir is None):
            raise ValueError("give exactly one of archive / v130_dir")
        self.archive = archive
        self.v130_dir = v130_dir
        self.name = "nhanaz" if archive is not None else "v130"
        self.last_records: list[dict] = []  # canonical records (with prize tables) of the last fetch

    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        product = ProductCode(spec.code.value)
        if self.archive is not None:
            got = await self.archive.draws(product)
        else:
            from vietlott_engine.crawler.sources.vqe130 import import_v130

            got = import_v130(self.v130_dir, product)
        records = [r for r in got.rows if since_id is None or int(r["draw_id"]) > since_id]
        self.last_records = records
        imp = import_canonical(records, spec.code)
        origin = {int(r["draw_id"]): r.get("data_source", "unknown") for r in records}
        draws = [d.model_copy(update={"source": f"{self.name}:{origin.get(d.draw_id, 'unknown')}"}) for d in imp.draws]
        res = FetchResult(draws=sorted(draws, key=lambda d: d.draw_id))
        res.rejected = [(str(r.get("draw_id")), r.get("error", "")) for r in imp.rejected + got.rejected]
        return res


class FallbackDrawSource(DrawSource):
    name = "auto"

    def __init__(self, sources: list[DrawSource], source_timeout_s: float = 20) -> None:
        if not sources:
            raise ValueError("no sources")
        self.sources = sources
        self.source_timeout_s = source_timeout_s
        self.log = FallbackLog()

    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        self.log = FallbackLog()
        empty = None
        for src in self.sources:
            try:
                res = await asyncio.wait_for(src.fetch(spec, since_id), self.source_timeout_s)
            except (SourceError, OSError, KeyError, ValueError, TimeoutError) as exc:
                msg = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
                log.warning("%s: source %s failed (%s); trying the next one", spec.code.value, src.name, msg)
                self.log.attempts.append(Attempt(src.name, False, error=msg))
                continue
            if not res.draws:
                self.log.attempts.append(Attempt(src.name, False, error='no newer draws'))
                empty = empty or res
                continue
            self.log.attempts.append(Attempt(src.name, True, rows=len(res.draws)))
            self.name = f"auto:{src.name}"
            return res
        if empty is not None:
            return empty  # empty is legitimate, but must not hide a newer fallback
        raise SourceError("all sources failed: " + "; ".join(f"{a.source}: {a.error}" for a in self.log.attempts))
