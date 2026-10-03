"""Store and sync for Keno, Bingo18, Max 3D, Max 3D Pro and Max 4D histories, and official
prize tables of Mega / Power / Lotto — with a fallback chain for when vietlott.vn is out of reach.

Layout: ``<data_dir>/products/<file>`` (same file names as ``data/seed``) plus
``exclusions.json``. Reading merges the bundled seed with the local store (store rows win on
equal draw ids) and leaves out draws Vietlott announced as *not confirmed*.

Sources (``--source``):

* ``vietlott``  — vietlott.vn results lists (AjaxPro). Vietnamese IPs only.
* ``nhanaz``    — community archive ``NhanAZ-Data/vietlott-research`` (GitHub raw, or a local
  folder via ``VQE_NHANAZ_DIR``); Keno/Bingo18 updated about every 10 minutes in the day;
  all 8 products + exclusions.
* ``mirror``    — ``vietvudanh/vietlott-data`` JSONL (Keno, Bingo18, Max 3D, Max 3D Pro).
* ``canonical`` — ``pqminh-4/vietlott-data`` canonical records (Max 3D, Max 3D Pro).
* ``v130``      — the snapshot of Vietlott Quant Engine 1.3.0 (``VQE_V130_DIR``).
* ``auto``      — the first of ``VQE_FALLBACK_ORDER`` that answers (default vietlott → nhanaz →
  mirror/canonical → v130); every attempt is in the report.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.products import (
    EXCLUSIONS_FILE,
    SEED_FILES,
    ProductCode,
    ProductHistory,
    drop_ids,
    get_product,
    history_to_rows,
    load_exclusions,
    parse_product_rows,
    read_jsonl,
    write_jsonl,
)
from vietlott_engine.crawler.http import AsyncHttpClient

log = get_logger(__name__)

PRODUCTS = tuple(SEED_FILES)
MIRROR_FILES = {ProductCode.KENO: "keno.jsonl", ProductCode.BINGO18: "bingo18.jsonl", ProductCode.MAX3D: "3d.jsonl", ProductCode.MAX3D_PRO: "3d_pro.jsonl"}
CANONICAL_FILES = {ProductCode.MAX3D: "max3d.jsonl", ProductCode.MAX3D_PRO: "max3d_pro.jsonl"}
SOURCE_ALIASES = {"github_mirror": "mirror"}


class ProductStore:
    def __init__(self, data_dir: Path, seed_dir: Path | None = None) -> None:
        self.dir = Path(data_dir) / "products"
        self.seed_dir = Path(seed_dir) if seed_dir else None

    def _rows(self, path: Path) -> list[dict]:
        return read_jsonl(path) if path.exists() else []

    def exclusions(self) -> dict[str, set[int]]:
        return load_exclusions(self.seed_dir, self.dir)

    def load(self, product: str | ProductCode, include_unconfirmed: bool = False) -> ProductHistory:
        code = get_product(product)
        if code not in SEED_FILES:
            raise KeyError(f"{code.value} is a matrix game; use the draw repository")
        rows = (self._rows(self.seed_dir / SEED_FILES[code]) if self.seed_dir else []) + self._rows(self.dir / SEED_FILES[code])
        sources = [s for s, p in (("seed", self.seed_dir / SEED_FILES[code] if self.seed_dir else None), ("store", self.dir / SEED_FILES[code])) if p is not None and p.exists()]
        h, _ = parse_product_rows(code, rows, source="+".join(sources) or "empty")
        return h if include_unconfirmed else drop_ids(h, self.exclusions().get(code.value, set()))

    def max_draw_id(self, product: ProductCode) -> int | None:
        h = self.load(product, include_unconfirmed=True)
        return int(h.draw_ids.max()) if len(h) else None

    def last_date(self, product: ProductCode) -> date | None:
        h = self.load(product, include_unconfirmed=True)
        return h.dates.max().astype("datetime64[D]").astype(date) if len(h) else None

    def upsert(self, product: ProductCode, rows: list[dict], source: str) -> tuple[int, int]:
        """Validate and merge ``rows`` into the store file → (new draw ids, rejected)."""
        new, rejected = parse_product_rows(product, rows, source)
        if not len(new):
            return 0, len(rejected)
        known = set(self.load(product, include_unconfirmed=True).draw_ids.tolist())
        path = self.dir / SEED_FILES[product]
        old, _ = parse_product_rows(product, self._rows(path), "store")
        merged = {int(i): r for i, r in zip(old.draw_ids, history_to_rows(old))}
        merged.update({int(i): r for i, r in zip(new.draw_ids, history_to_rows(new))})
        write_jsonl(path, [merged[i] for i in sorted(merged)])
        return len(set(new.draw_ids.tolist()) - known), len(rejected)

    def add_exclusions(self, items: list[dict]) -> int:
        """Record not-confirmed draws (issuer notices) in the store → number added."""
        if not items:
            return 0
        path = self.dir / EXCLUSIONS_FILE
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        known = {(e["product"], int(e["draw_id"])) for e in current} | {(p, i) for p, ids in load_exclusions(self.seed_dir).items() for i in ids}
        added = [e for e in items if (e["product"], int(e["draw_id"])) not in known]
        if added:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current + added, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(added)

    def coverage(self) -> list[dict]:
        out = []
        excl = self.exclusions()
        for code in PRODUCTS:
            try:
                out.append(self.load(code).coverage() | {"excluded_not_confirmed": len(excl.get(code.value, set()))})
            except (FileNotFoundError, ValueError) as exc:
                out.append({"product": code.value, "draws": 0, "error": str(exc)})
        return out


@dataclass
class ProductSyncReport:
    product: str
    source: str
    fetched: int
    inserted: int
    rejected: int
    total_after: int
    last_id: int | None
    last_date: str | None
    missing_ids_inside_range: int
    elapsed_s: float
    error: str | None = None
    source_used: str | None = None
    exclusions_added: int = 0
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ProductSyncPipeline:
    def __init__(
        self,
        client: AsyncHttpClient,
        store: ProductStore,
        *,
        vietlott_base_url: str,
        mirror_base_url: str,
        canonical_base_url: str,
        bootstrap_cookie: bool = True,
        pages_per_batch: int = 4,
        nhanaz_base_url: str | None = None,
        nhanaz_dir: Path | None = None,
        v130_dir: Path | None = None,
        fallback_order: list[str] | None = None,
        source_timeout_s: float = 20,
    ) -> None:
        from vietlott_engine.crawler.sources.nhanaz import NHANAZ_RAW_BASE

        self.client = client
        self.store = store
        self.vietlott_base_url = vietlott_base_url
        self.mirror_base_url = mirror_base_url.rstrip("/")
        self.canonical_base_url = canonical_base_url.rstrip("/")
        self.bootstrap_cookie = bootstrap_cookie
        self.pages_per_batch = pages_per_batch
        self.nhanaz_base_url = nhanaz_base_url or NHANAZ_RAW_BASE
        self.nhanaz_dir = nhanaz_dir
        self.v130_dir = v130_dir
        self.fallback_order = [SOURCE_ALIASES.get(s, s) for s in (fallback_order or ["vietlott", "nhanaz", "mirror", "canonical", "v130"])]
        self._pending_exclusions: list[dict] = []
        self.source_timeout_s = source_timeout_s

    async def _jsonl(self, url: str) -> list[dict]:
        text = (await self.client.get(url)).text
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _archive(self):  # type: ignore[no-untyped-def]
        from vietlott_engine.crawler.sources.nhanaz import NhanAZArchive

        return NhanAZArchive(local_dir=self.nhanaz_dir) if self.nhanaz_dir else NhanAZArchive(self.client, self.nhanaz_base_url)

    def chain(self, product: ProductCode) -> list[str]:
        """Sources of ``auto`` that can serve this product."""
        out = []
        for s in self.fallback_order:
            if (s == "mirror" and product not in MIRROR_FILES) or (s == "canonical" and product not in CANONICAL_FILES) or (s == "v130" and self.v130_dir is None):
                continue
            if s == "vietlott" and product == ProductCode.MAX4D:
                continue  # discontinued: no live list
            out.append(s)
        return out

    async def fetch(self, product: ProductCode, source: str, max_pages: int, full: bool) -> list[dict]:
        source = SOURCE_ALIASES.get(source, source)
        if source == "vietlott":
            from vietlott_engine.crawler.sources.vietlott_official import VietlottProductSource

            src = VietlottProductSource(self.client, self.vietlott_base_url, self.pages_per_batch, self.bootstrap_cookie)
            since = None if full else self.store.max_draw_id(product)
            return await src.fetch_pages(product, max_pages=max_pages, since_id=since)
        if source == "nhanaz":
            archive = self._archive()
            got = await archive.draws(product, since=None if full else self.store.last_date(product))
            self._pending_exclusions += [e for e in await archive.exclusions() if e["product"] == product.value]
            return got.rows
        if source == "v130":
            if self.v130_dir is None:
                raise SourceError("v130 source needs VQE_V130_DIR (Vietlott Quant Engine 1.3.0 folder)")
            from vietlott_engine.crawler.sources.vqe130 import import_v130, v130_exclusions

            got = import_v130(self.v130_dir, product)
            self._pending_exclusions += [e for e in v130_exclusions(self.v130_dir) if e["product"] == product.value]
            return got.rows
        if source == "mirror":
            if product not in MIRROR_FILES:
                raise SourceError(f"no mirror file for {product.value}")
            return await self._jsonl(f"{self.mirror_base_url}/{MIRROR_FILES[product]}")
        if source == "canonical":
            if product not in CANONICAL_FILES:
                raise SourceError(f"no canonical records for {product.value}; use --source nhanaz, mirror or vietlott")
            return await self._jsonl(f"{self.canonical_base_url}/{CANONICAL_FILES[product]}")
        raise SourceError(f"unknown product source {source!r}")

    async def run(self, product: ProductCode, source: str = "auto", max_pages: int = 50, full: bool = False) -> ProductSyncReport:
        t0 = time.perf_counter()
        self._pending_exclusions = []
        attempts: list[dict] = []
        rows: list[dict] = []
        used = None
        best_rows, best_source, best_id = [], None, -1
        known_max = self.store.max_draw_id(product) or 0
        for s in self.chain(product) if source == "auto" else [source]:
            try:
                rows = await asyncio.wait_for(self.fetch(product, s, max_pages, full), self.source_timeout_s)
            except (SourceError, OSError, ValueError, TimeoutError) as exc:
                msg = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
                attempts.append({"source": s, "ok": False, "rows": 0, "error": msg})
                log.warning("%s: source %s failed (%s)", product.value, s, msg)
                continue
            attempts.append({"source": s, "ok": True, "rows": len(rows), "error": None})
            parsed, _ = parse_product_rows(product, rows, s)
            source_max = int(parsed.draw_ids.max()) if len(parsed) else -1
            if source_max > best_id:
                best_rows, best_source, best_id = rows, s, source_max
            if source == 'auto' and source_max <= known_max:
                attempts[-1]['stale'] = True
                continue
            used = s
            break
        if used is None and best_source is not None:
            rows, used = best_rows, best_source
        error = None if used else "; ".join(f"{a['source']}: {a['error']}" for a in attempts)
        inserted, rejected = self.store.upsert(product, rows, used or source) if rows else (0, 0)
        excl = self.store.add_exclusions(self._pending_exclusions)
        h = self.store.load(product)
        cov = h.coverage()
        return ProductSyncReport(
            product=product.value,
            source=source,
            fetched=len(rows),
            inserted=inserted,
            rejected=rejected,
            total_after=len(h),
            last_id=cov["last_id"],
            last_date=cov["last_date"],
            missing_ids_inside_range=cov["missing_ids_inside_range"],
            elapsed_s=round(time.perf_counter() - t0, 3),
            error=error,
            source_used=used,
            exclusions_added=excl,
            attempts=attempts,
        )


# ----------------------------------------------------------------- official prize data
@dataclass
class CanonicalPrizeReport:
    game: str
    source: str
    records: int
    draws_inserted: int
    prize_records: int
    prize_records_inserted: int
    with_jackpot_pots: int
    rejected: int
    elapsed_s: float
    error: str | None = None
    source_used: str | None = None
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


async def _prize_records(client, repository, spec, source: str, *, canonical_base_url: str, vietlott_base_url: str, last: int, bootstrap_cookie: bool, nhanaz_base_url: str | None, nhanaz_dir: Path | None, v130_dir: Path | None) -> list[dict]:  # type: ignore[no-untyped-def]
    from vietlott_engine.crawler.sources.vietlott_official import VietlottOfficialSource

    if source == "canonical":
        text = (await client.get(f"{canonical_base_url.rstrip('/')}/{spec.code.value}.jsonl")).text
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if source == "vietlott":
        ids = sorted(d.draw_id for d in repository.load(spec.code))[-last:]
        return await VietlottOfficialSource(client, vietlott_base_url, bootstrap_cookie=bootstrap_cookie).fetch_canonical(spec, ids)
    if source == "nhanaz":
        from vietlott_engine.crawler.sources.nhanaz import NHANAZ_RAW_BASE, NhanAZArchive

        archive = NhanAZArchive(local_dir=nhanaz_dir) if nhanaz_dir else NhanAZArchive(client, nhanaz_base_url or NHANAZ_RAW_BASE)
        return (await archive.draws(ProductCode(spec.code.value))).rows
    if source == "v130":
        if v130_dir is None:
            raise SourceError("v130 source needs VQE_V130_DIR (Vietlott Quant Engine 1.3.0 folder)")
        from vietlott_engine.crawler.sources.vqe130 import import_v130

        return import_v130(v130_dir, ProductCode(spec.code.value)).rows
    raise SourceError(f"unknown prize source {source!r}")


async def sync_canonical_prizes(
    client: AsyncHttpClient,
    repository,  # type: ignore[no-untyped-def]
    spec,  # type: ignore[no-untyped-def]
    *,
    source: str,
    canonical_base_url: str,
    vietlott_base_url: str,
    last: int = 20,
    bootstrap_cookie: bool = True,
    nhanaz_base_url: str | None = None,
    nhanaz_dir: Path | None = None,
    v130_dir: Path | None = None,
    fallback_order: list[str] | None = None,
) -> CanonicalPrizeReport:
    """Official prize tables → repository.

    ``canonical``: the ``pqminh-4`` record file. ``vietlott``: detail pages of the ``last`` most
    recent stored draws (from Vietnam). ``nhanaz`` / ``v130``: community archive / 1.3.0
    snapshot. ``auto``: the first of vietlott → canonical → nhanaz → v130 that answers."""
    from vietlott_engine.crawler.official_data import import_canonical

    t0 = time.perf_counter()
    order = [s for s in (fallback_order or ["vietlott", "canonical", "nhanaz", "v130"]) if not (s == "v130" and v130_dir is None)]
    attempts: list[dict] = []
    rows: list[dict] = []
    used = None
    for s in order if source == "auto" else [source]:
        try:
            rows = await asyncio.wait_for(_prize_records(
                client, repository, spec, s, canonical_base_url=canonical_base_url, vietlott_base_url=vietlott_base_url, last=last,
                bootstrap_cookie=bootstrap_cookie, nhanaz_base_url=nhanaz_base_url, nhanaz_dir=nhanaz_dir, v130_dir=v130_dir,
            ), 20)
        except (SourceError, OSError, ValueError, TimeoutError) as exc:
            attempts.append({"source": s, "ok": False, "rows": 0, "error": (str(exc).splitlines() or [type(exc).__name__])[0][:300]})
            continue
        attempts.append({"source": s, "ok": True, "rows": len(rows), "error": None})
        if source == 'auto' and not rows:
            continue
        used = s
        break
    imp = import_canonical(rows, spec.code)
    draws_inserted = repository.upsert(imp.draws) if imp.draws else 0
    prizes_inserted = repository.upsert_prizes(imp.prizes) if imp.prizes else 0
    return CanonicalPrizeReport(
        game=spec.code.value,
        source=source,
        records=len(rows),
        draws_inserted=draws_inserted,
        prize_records=len(imp.prizes),
        prize_records_inserted=prizes_inserted,
        with_jackpot_pots=sum(p.jackpot_pots is not None for p in imp.prizes),
        rejected=len(imp.rejected),
        elapsed_s=round(time.perf_counter() - t0, 3),
        error=None if used else "; ".join(f"{a['source']}: {a['error']}" for a in attempts),
        source_used=used,
        attempts=attempts,
    )
