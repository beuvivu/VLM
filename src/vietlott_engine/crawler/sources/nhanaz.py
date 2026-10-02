"""Community archive ``NhanAZ-Data/vietlott-research`` — fallback when vietlott.vn is unreachable.

The archive is updated by GitHub Actions — Keno/Bingo18 about every 10 minutes during the day,
the other products several times a day — from vietlott.vn pages when they are reachable and
from declared secondary result sites (xosominhngoc.net.vn, xoso.com.vn archive, onbit)
otherwise; how each official row was obtained is the archive's own business. Every row keeps its
``data_source`` and ``draw_status``; draws that Vietlott announced as *not confirmed* are listed
in ``datasets/exclusions.csv``. MIT licence. Vietlott Quant Engine 1.3.0 imported a pinned
revision of the same archive.

Layout (repository root or the ``datasets`` folder; a ``git clone``, a downloaded ZIP or the
product cache written by Vietlott Quant Engine 1.3.0 ``collect_product_history.py`` all work):

    datasets/draws/<product>/all.csv            Mega, Power, Lotto, Max 3D, Max 3D Pro, Max 4D
    datasets/draws/<product>/YYYY-MM.csv        Keno, Bingo18 (monthly partitions)
    datasets/prizes/<product>/all.csv           prize tables
    datasets/exclusions.csv                     not-confirmed draws
    datasets/prize_rules.csv                    prize rules read from detail pages

This module converts rows to the engine's formats: canonical records (schema 1.0, as in
``crawler.official_data``) for the matrix games and product rows for the others.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.products import FOUR_DIGIT_LAYOUT, THREE_DIGIT_LAYOUT, ProductCode
from vietlott_engine.crawler.official_data import fold

NHANAZ_REPO = "NhanAZ-Data/vietlott-research"
NHANAZ_RAW_BASE = f"https://raw.githubusercontent.com/{NHANAZ_REPO}/main/datasets"
MONTHLY = {ProductCode.KENO, ProductCode.BINGO18}
FIRST_MONTH = {ProductCode.KENO: (2019, 8), ProductCode.BINGO18: (2024, 12)}
MATRIX = {ProductCode.MEGA_645: "mega645", ProductCode.POWER_655: "power655", ProductCode.LOTTO_535: "lotto535"}
OFFICIAL_SOURCE = "official_vietlott"

_NUM = re.compile(r"\d{1,3}(?:\.\d{3})+|\d+")


def _num(text: str | None) -> int | None:
    """'1.234.567' → 1234567; None unless the whole cell is a number."""
    t = (text or "").strip()
    return int(t.replace(".", "")) if t and _NUM.fullmatch(t) else None


def read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _json(cell: str | None) -> dict | list:
    try:
        return json.loads(cell) if cell else {}
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------------ draws
@dataclass
class ArchiveDraws:
    product: ProductCode
    rows: list[dict] = field(default_factory=list)  # engine rows (product rows or canonical records)
    not_confirmed: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    sources: Counter = field(default_factory=Counter)


def _digits_tiers(result: dict, layout: tuple[tuple[str, int], ...]) -> list[dict]:
    tiers = result.get("tiers") or {}
    out = []
    for code, n in layout:
        nums = [str(x) for x in tiers.get(code, [])]
        if len(nums) != n:
            raise ValueError(f"tier {code}: {len(nums)} numbers, expected {n}")
        out.append({"code": code, "numbers": nums})
    return out


def convert_draws(product: ProductCode, csv_rows: list[dict]) -> ArchiveDraws:
    """Archive draw rows → engine rows. Matrix games become canonical records (prizes are
    attached by :func:`attach_prizes`); the other products become ``{"id","date","result"}``."""
    out = ArchiveDraws(product)
    for r in csv_rows:
        try:
            did = int(str(r["draw_id"]).lstrip("#"))
            d = date.fromisoformat(r["draw_date"][:10]).isoformat()
            result = _json(r.get("result_json"))
            attrs = _json(r.get("attributes_json"))
            src = attrs.get("data_source", "unknown") if isinstance(attrs, dict) else "unknown"
            status = (r.get("draw_status") or "confirmed").strip()
            if product in MATRIX:
                row = {
                    "draw_id": f"{did:05d}",
                    "draw_date": d,
                    "game": MATRIX[product],
                    "result": {"kind": "number_set", "main_numbers": [int(x) for x in result["numbers"]], "bonus_numbers": [int(x) for x in result.get("special_numbers") or []]},
                    "prizes": [],
                    "jackpots_vnd": attrs.get("jackpots_vnd") if isinstance(attrs, dict) else None,
                    "source_url": r.get("source_url"),
                    "source_pdf_url": (_json(r.get("official_pdf_urls_json")) or [None])[0],
                    "data_source": src,
                    "schema_version": "1.0",
                }
            elif product == ProductCode.KENO:
                row = {"id": did, "date": d, "result": [int(x) for x in result["numbers"]], "data_source": src}
            elif product == ProductCode.BINGO18:
                row = {"id": did, "date": d, "result": [int(x) for x in result["digits"]], "data_source": src}
            elif product in (ProductCode.MAX3D, ProductCode.MAX3D_PRO):
                row = {"draw_id": f"{did:05d}", "draw_date": d, "result": {"kind": "three_digit_tiers", "tiers": _digits_tiers(result, THREE_DIGIT_LAYOUT)}, "data_source": src}
            elif product == ProductCode.MAX4D:
                row = {"draw_id": f"{did:05d}", "draw_date": d, "result": {"kind": "four_digit_tiers", "tiers": _digits_tiers(result, FOUR_DIGIT_LAYOUT)}, "data_source": src}
            else:
                raise ValueError(f"unsupported product {product.value}")
        except (KeyError, ValueError, TypeError) as exc:
            out.rejected.append({"draw_id": r.get("draw_id"), "error": str(exc).splitlines()[0]})
            continue
        if status != "confirmed":
            out.not_confirmed.append({"product": product.value, "draw_id": did, "draw_status": status, "draw_date": d})
            continue
        out.sources[src] += 1
        out.rows.append(row)
    return out


# ------------------------------------------------------------------ prizes
def prize_cells(r: dict) -> tuple[str, int, int] | None:
    """(tier name, winners, value) of one archive prize row, or None for header/odd rows.

    Two layouts exist: explicit ``winner_count``/``prize_value_vnd`` columns, and older Lotto
    rows where the page header shifted the cells (``column_3`` = winners, ``column_4`` = value)."""
    name = (r.get("prize_tier") or "").strip()
    if not name or fold(name) == "giai thuong":
        return None
    w, v = _num(r.get("winner_count")), _num(r.get("prize_value_vnd"))
    if w is None or v is None:
        cols = _json(r.get("details_json")).get("columns") if isinstance(_json(r.get("details_json")), dict) else None
        if isinstance(cols, dict) and "column_3" in cols and "column_4" in cols:
            w, v = _num(cols["column_3"]), _num(cols["column_4"])
        elif isinstance(cols, list) and len(cols) >= 4:
            w, v = _num(cols[-2]), _num(cols[-1])
    if w is None or v is None:
        return None
    return name, w, v


def attach_prizes(records: list[dict], prize_rows: list[dict]) -> int:
    """Add the prize table of each draw (canonical ``prizes`` list) in place → draws with prizes."""
    from vietlott_engine.crawler.official_data import slug

    by_draw: dict[int, list[dict]] = defaultdict(list)
    for r in prize_rows:
        cells = prize_cells(r)
        if cells is None:
            continue
        name, w, v = cells
        is_jp = "jackpot" in fold(name) or "doc dac" in fold(name)
        by_draw[int(str(r["draw_id"]).lstrip("#"))].append(
            {"code": slug(name), "name": name, "winner_count": w, "amount_vnd": None if is_jp else v, "jackpot_vnd": v if is_jp else None}
        )
    n = 0
    for rec in records:
        p = by_draw.get(int(rec["draw_id"]))
        if p:
            seen = {}
            for x in p:  # one row per tier (archive may repeat a tier across page tables)
                seen.setdefault(x["code"], x)
            rec["prizes"] = list(seen.values())
            n += 1
    return n


def parse_exclusions(csv_rows: list[dict]) -> list[dict]:
    out = []
    for r in csv_rows:
        try:
            out.append(
                {
                    "product": r["product"].strip(),
                    "draw_id": int(str(r["draw_id"]).lstrip("#")),
                    "draw_status": (r.get("draw_status") or "not_confirmed").strip(),
                    "effective_date": (r.get("effective_date") or "")[:10],
                    "reason": r.get("reason", ""),
                    "source_url": r.get("source_url", ""),
                }
            )
        except (KeyError, ValueError):
            continue
    return out


def keno_rules(csv_rows: list[dict]) -> dict:
    """Side-bet and per-bậc prize cells of a Keno detail page (``prize_rules.csv``)."""
    side, bac = {}, defaultdict(dict)
    for r in csv_rows:
        if r.get("product") != "keno":
            continue
        name, v = r.get("prize_tier", ""), _num(r.get("prize_value_vnd"))
        d = _json(r.get("details_json"))
        ti = d.get("table_index") if isinstance(d, dict) else None
        m = re.match(r"Trùng (\d+) trong 20 số", name)
        if m and isinstance(ti, int) and 5 <= ti <= 14:
            bac[15 - ti][int(m.group(1))] = v  # table 5 = bậc 10 … table 14 = bậc 1
        elif v is not None:
            side[name] = v
    return {"side_bets": side, "bac": {k: dict(sorted(v.items())) for k, v in sorted(bac.items())}, "source_url": next((r.get("source_url") for r in csv_rows if r.get("product") == "keno"), None)}


# ------------------------------------------------------------------ access
class NhanAZArchive:
    """Read the archive from a local folder or from GitHub (raw files)."""

    name = "nhanaz"

    def __init__(self, client=None, base_url: str = NHANAZ_RAW_BASE, local_dir: Path | str | None = None) -> None:  # type: ignore[no-untyped-def]
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.root: Path | None = None
        if local_dir is not None:
            p = Path(local_dir)
            self.root = p / "datasets" if (p / "datasets").is_dir() else p
            if not (self.root / "draws").is_dir():
                raise SourceError(f"{local_dir}: not a vietlott-research archive (no datasets/draws folder)")

    async def _text(self, rel: str, missing_ok: bool = False) -> str | None:
        if self.root is not None:
            path = self.root / rel
            if not path.exists():
                if missing_ok:
                    return None
                raise SourceError(f"archive file missing: {path}")
            return path.read_text(encoding="utf-8")
        if self.client is None:
            raise SourceError("NhanAZArchive needs an HTTP client or a local folder")
        try:
            return (await self.client.get(f"{self.base_url}/{rel}")).text
        except SourceError as exc:
            if missing_ok and "404" in str(exc):
                return None
            raise

    def _months(self, product: ProductCode, since: date | None) -> list[str]:
        if self.root is not None:
            files = sorted(p.stem for p in (self.root / "draws" / product.value).glob("*.csv"))
            return [m for m in files if since is None or m >= f"{since.year:04d}-{since.month:02d}"]
        y, m = (since.year, since.month) if since else FIRST_MONTH[product]
        today = date.today()
        out = []
        while (y, m) <= (today.year, today.month):
            out.append(f"{y:04d}-{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    async def draws(self, product: ProductCode, since: date | None = None) -> ArchiveDraws:
        if product in MONTHLY:
            rows: list[dict] = []
            for month in self._months(product, since):
                text = await self._text(f"draws/{product.value}/{month}.csv", missing_ok=True)
                if text:
                    rows += read_csv(text)
        else:
            rows = read_csv(await self._text(f"draws/{product.value}/all.csv") or "")
        out = convert_draws(product, rows)
        if product in MATRIX:
            prizes = await self._text(f"prizes/{product.value}/all.csv", missing_ok=True)
            if prizes:
                attach_prizes(out.rows, read_csv(prizes))
        return out

    async def exclusions(self) -> list[dict]:
        text = await self._text("exclusions.csv", missing_ok=True)
        return parse_exclusions(read_csv(text)) if text else []

    async def prize_rules(self) -> list[dict]:
        text = await self._text("prize_rules.csv", missing_ok=True)
        return read_csv(text) if text else []
