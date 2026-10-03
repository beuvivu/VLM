"""Incremental sync pipeline: source → validation → integrity checks → repository → Parquet."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vietlott_engine.core.config import Settings
from vietlott_engine.core.games import GameSpec
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.models import Draw
from vietlott_engine.crawler.http import AsyncHttpClient, RetryPolicy
from vietlott_engine.crawler.sources.base import DrawSource
from vietlott_engine.crawler.sources.mirror import GithubMirrorSource, JsonlFileSource
from vietlott_engine.crawler.sources.vietlott_official import VietlottOfficialSource
from vietlott_engine.crawler.storage import DrawRepository, DuckDBRepository

log = get_logger(__name__)


@dataclass
class IntegrityReport:
    duplicate_ids: list[int] = field(default_factory=list)
    id_gaps: list[tuple[int, int]] = field(default_factory=list)
    date_order_violations: list[int] = field(default_factory=list)
    off_schedule_draws: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.duplicate_ids or self.date_order_violations)


@dataclass
class SyncReport:
    game: str
    source: str
    fetched: int
    inserted: int
    rejected: int
    total_after: int
    max_draw_id: int | None
    integrity: IntegrityReport
    parquet_path: str | None
    elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def check_integrity(draws: list[Draw], spec: GameSpec) -> IntegrityReport:
    """Data-quality checks on the full (sorted) history. Gaps are reported, not fatal."""
    rep = IntegrityReport()
    ids = [d.draw_id for d in draws]
    if ids and min(ids) > 1:
        rep.id_gaps.append((1, min(ids) - 1))  # leading draws missing from the source
    seen: set[int] = set()
    for i in ids:
        if i in seen:
            rep.duplicate_ids.append(i)
        seen.add(i)
    by_id = sorted(draws, key=lambda d: d.draw_id)
    for a, b in zip(by_id[:-1], by_id[1:]):
        if b.draw_id - a.draw_id > 1:
            rep.id_gaps.append((a.draw_id + 1, b.draw_id - 1))
        if b.draw_date < a.draw_date:
            rep.date_order_violations.append(b.draw_id)
    if spec.draw_weekdays:
        rep.off_schedule_draws = [d.draw_id for d in draws if d.draw_date.weekday() not in spec.draw_weekdays]
    return rep


class SyncPipeline:
    def __init__(self, source: DrawSource, repository: DrawRepository) -> None:
        self.source = source
        self.repository = repository

    async def run(self, spec: GameSpec, full_refresh: bool = False) -> SyncReport:
        t0 = time.perf_counter()
        since = None if full_refresh else self.repository.max_draw_id(spec.code)
        log.info("sync %s from %s (since draw_id=%s)", spec.code.value, self.source.name, since)
        fetched = await self.source.fetch(spec, since_id=since)
        inserted = self.repository.upsert(fetched.draws)
        self.repository.log_sync(spec.code, self.source.name, len(fetched.draws), inserted, len(fetched.rejected))

        all_draws = self.repository.load(spec.code)
        integrity = check_integrity(all_draws, spec)
        if integrity.id_gaps:
            log.warning("%s: %d missing draw-id ranges, e.g. %s", spec.code.value, len(integrity.id_gaps), integrity.id_gaps[:3])
        if not integrity.ok:
            log.error("%s integrity problems: %s", spec.code.value, integrity)

        parquet_path = None
        if isinstance(self.repository, DuckDBRepository) and self.repository.parquet_dir is not None:
            parquet_path = str(self.repository.export_parquet(spec.code))

        return SyncReport(
            game=spec.code.value,
            source=self.source.name,
            fetched=len(fetched.draws),
            inserted=inserted,
            rejected=len(fetched.rejected),
            total_after=len(all_draws),
            max_draw_id=self.repository.max_draw_id(spec.code),
            integrity=integrity,
            parquet_path=parquet_path,
            elapsed_s=round(time.perf_counter() - t0, 3),
        )


def build_http_client(settings: Settings) -> AsyncHttpClient:
    if settings.http_backend == "curl_cffi":
        import os
        from dataclasses import replace
        from vlm.crawler.async_scraper import AsyncScraper, ScraperConfig, FlareSolverrSolver

        config = replace(ScraperConfig.from_env(), timeout_s=settings.http_timeout_s,
                         rate_per_s=settings.rate_limit_per_s, max_concurrency=settings.max_concurrency,
                         max_retries=settings.max_retries)
        endpoint = os.getenv("VLM_FLARESOLVERR_URL")
        solver = FlareSolverrSolver(endpoint) if endpoint else None
        return AsyncScraper(config, solver=solver)  # type: ignore[return-value]
    return AsyncHttpClient(
        timeout=settings.http_timeout_s,
        rate_limit_per_s=settings.rate_limit_per_s,
        burst=settings.rate_limit_burst,
        max_concurrency=settings.max_concurrency,
        retry=RetryPolicy(
            max_retries=settings.max_retries,
            base_delay=settings.backoff_base_s,
            max_delay=settings.backoff_max_s,
        ),
        headers={"User-Agent": settings.user_agent, "Accept": "*/*"},
    )


def build_source(settings: Settings, client: AsyncHttpClient | None, kind: str | None = None, path: Path | None = None) -> DrawSource:
    kind = kind or settings.source
    if kind == "file":
        if path is None:
            raise ValueError("file source needs a path")
        return JsonlFileSource(path)
    if kind == "v130":
        from vietlott_engine.crawler.sources.fallback import ArchiveDrawSource

        root = path or settings.v130_dir
        if root is None:
            raise ValueError("v130 source needs VQE_V130_DIR (Vietlott Quant Engine 1.3.0 folder) or --path")
        return ArchiveDrawSource(v130_dir=root)
    if kind == "nhanaz" and (path or settings.nhanaz_dir):
        from vietlott_engine.crawler.sources.fallback import ArchiveDrawSource
        from vietlott_engine.crawler.sources.nhanaz import NhanAZArchive

        return ArchiveDrawSource(archive=NhanAZArchive(local_dir=path or settings.nhanaz_dir))
    if client is None:
        raise ValueError(f"source {kind!r} needs an HTTP client")
    if kind == "nhanaz":
        from vietlott_engine.crawler.sources.fallback import ArchiveDrawSource
        from vietlott_engine.crawler.sources.nhanaz import NhanAZArchive

        return ArchiveDrawSource(archive=NhanAZArchive(client, settings.nhanaz_base_url))
    if kind == "auto":
        from vietlott_engine.crawler.sources.fallback import FallbackDrawSource

        chain = []
        for k in settings.fallback_order:
            if k == "v130" and settings.v130_dir is None:
                continue
            chain.append(build_source(settings, client, k))
        return FallbackDrawSource(chain, settings.source_timeout_s)
    if kind == "github_mirror":
        return GithubMirrorSource(client, settings.github_mirror_base_url)
    if kind == "vietlott":
        return VietlottOfficialSource(client, settings.vietlott_base_url, settings.official_max_pages, settings.max_concurrency, bootstrap_cookie=settings.vietlott_cookie_bootstrap)
    raise ValueError(f"unknown source {kind!r}")


# ----------------------------------------------------------------- prize data
@dataclass
class PrizeSyncReport:
    game: str
    records: int
    inserted: int
    with_pots: int
    reconcile: dict
    elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class PrizeSyncPipeline:
    """Download public prize data, reconcile it against the stored draws, upsert."""

    def __init__(self, client: AsyncHttpClient, repository: DrawRepository, winners_base_url: str, power_history_url: str) -> None:
        self.client = client
        self.repository = repository
        self.winners_base_url = winners_base_url.rstrip("/")
        self.power_history_url = power_history_url

    async def run(self, spec: GameSpec) -> PrizeSyncReport:
        from vietlott_engine.crawler.prize_sources import COMPAL_FILES, parse_compal_winners, parse_leoodz_power, reconcile

        t0 = time.perf_counter()
        compal = parse_compal_winners((await self.client.get(f"{self.winners_base_url}/{COMPAL_FILES[spec.code]}")).text, spec.code)
        leo = []
        if spec.code.value == "power655":
            leo = parse_leoodz_power((await self.client.get(self.power_history_url)).text)
        draws = self.repository.load(spec.code)
        records, rep = reconcile(spec, draws, compal, leo)
        inserted = self.repository.upsert_prizes(records)
        return PrizeSyncReport(
            game=spec.code.value,
            records=len(records),
            inserted=inserted,
            with_pots=rep.with_pots,
            reconcile=asdict(rep),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
