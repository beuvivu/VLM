"""Official vietlott.vn source for every product.

Access
------
vietlott.vn serves **Vietnamese IP addresses only**: requests from data centres abroad
(cloud machines, GitHub-hosted runners) get HTTP 403. Run this crawler from a machine in
Vietnam (a home/office PC, a VPS in Vietnam, a self-hosted runner) — or set
``HTTPS_PROXY`` to a proxy you operate in Vietnam. Everything else (the GitHub mirrors
and the official-record dataset) works from anywhere.

Protocol (as used by the maintained open-source crawlers vietvudanh/vietlott-data and
pqminh-4/vietlott-data, MIT)
---------------------------------------------------------------------------
* History lists — POST ``/ajaxpro/Vietlott.PlugIn.WebParts.<Part>,Vietlott.PlugIn.WebParts.ashx``
  with header ``X-AjaxPro-Method: ServerSideDrawResult`` and a JSON body carrying
  ``PageIndex``; the answer is ``{"value": {"HtmlContent": "<table>…"}}``.
  Mega/Power/Lotto bodies carry a per-game ``Key``; Max 3D / Max 3D Pro / Keno / Bingo18
  carry a ``GameId`` (5 / 7 / 6 / 8).
* Some edges require a first-party cookie that ``GET /ajaxpro/`` sets through
  ``document.cookie="…"``; it is bootstrapped once per client.
* Draw details — GET ``<results page>?id=<draw id>&nocatche=1``: result, prize table
  (tier, matches, winners, value), official PDF link on media.vietlott.vn.

Markup and keys belong to the website and can change; parsers are defensive and every
row that fails validation is reported, never silently kept.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import GameCode, GameSpec
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.products import THREE_DIGIT_LAYOUT, ProductCode
from vietlott_engine.crawler.http import AsyncHttpClient
from vietlott_engine.crawler.official_data import fold, slug
from vietlott_engine.crawler.schemas import OfficialRow
from vietlott_engine.crawler.sources.base import DrawSource, FetchResult

log = get_logger(__name__)

WEB_BASE = "https://www.vietlott.vn"
GEO_BLOCK_HINT = (
    "vietlott.vn rejected the request (HTTP 403). The site only serves Vietnamese IP addresses: run the crawler "
    "from a machine in Vietnam (or through a proxy you operate there), or use the github_mirror / canonical sources."
)
CLOUDFLARE_HINT = (
    "vietlott.vn answered with a Cloudflare 'verify you are human' challenge. The engine does not try to solve or "
    "bypass it. Pass the check yourself in a browser and save the pages or the network log (HAR), then run "
    "`vietlott products import-pages <folder>`; or use `--source auto` (community archive and mirrors)."
)
PROXY_BLOCK_HINT = (
    "this network's HTTPS proxy does not allow www.vietlott.vn (egress policy). Ask the network administrator to "
    "allow the host, or run the crawler on another network in Vietnam; the github_mirror / canonical sources need "
    "raw.githubusercontent.com instead."
)


def _access_error(exc: SourceError) -> SourceError:
    """Translate a 403 into an actionable message: proxy policy vs. vietlott.vn's geo-block."""
    from vietlott_engine.crawler.http import CF_CHALLENGE_PREFIX, PROXY_BLOCK_PREFIX

    msg = str(exc)
    if CF_CHALLENGE_PREFIX in msg:
        return SourceError(CLOUDFLARE_HINT)
    if PROXY_BLOCK_PREFIX in msg:
        return SourceError(PROXY_BLOCK_HINT)
    if "403" in msg:
        return SourceError(GEO_BLOCK_HINT)
    return exc

_RENDER_INFO = {
    "SiteId": "main.frontend.vi",
    "SiteAlias": "main.vi",
    "UserSessionId": "",
    "SiteLang": "vi",
    "IsPageDesign": False,
    "ExtraParam1": "",
    "ExtraParam2": "",
    "ExtraParam3": "",
    "SiteURL": "",
    "WebPage": None,
    "SiteName": "Vietlott",
    "OrgPageAlias": None,
    "PageAlias": None,
    "RefKey": None,
    "FullPageAlias": None,
    "System": 1,
}


@dataclass(frozen=True)
class OfficialEndpoint:
    part: str  # AjaxPro web part
    detail_path: str  # results / detail page
    body: Callable[[int], dict[str, Any]]

    @property
    def path(self) -> str:
        return f"/ajaxpro/Vietlott.PlugIn.WebParts.{self.part},Vietlott.PlugIn.WebParts.ashx"


def _matrix_body(key: str, rows: int, cols: int) -> Callable[[int], dict[str, Any]]:
    def body(page: int) -> dict[str, Any]:
        return {
            "ORenderInfo": _RENDER_INFO,
            "Key": key,
            "GameDrawId": "",
            "ArrayNumbers": [["" for _ in range(cols)] for _ in range(rows)],
            "CheckMulti": False,
            "PageIndex": page,
        }

    return body


def _three_digit_body(game_id: str) -> Callable[[int], dict[str, Any]]:
    def body(page: int) -> dict[str, Any]:
        return {"ORenderInfo": _RENDER_INFO, "GameId": game_id, "GameDrawId": "", "PageIndex": page, "CheckMulti": 0, "number01": "123", "number02": "321"}

    return body


def _keno_body(page: int) -> dict[str, Any]:
    return {"DrawDate": "", "GameDrawNo": "", "GameId": "6", "ORenderInfo": _RENDER_INFO, "OddEven": 2, "PageIndex": page, "ProcessType": 0, "TotalRow": 112453, "UpperLower": 2, "number": ""}


def _bingo_body(page: int) -> dict[str, Any]:
    return {"ORenderInfo": _RENDER_INFO, "GameId": "8", "GameDrawNo": "", "number": "", "DrawDate": "", "PageIndex": page, "TotalRow": 43569}


ENDPOINTS: dict[ProductCode, OfficialEndpoint] = {
    ProductCode.MEGA_645: OfficialEndpoint("Game645CompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/645", _matrix_body("8290fce2", 6, 18)),
    ProductCode.POWER_655: OfficialEndpoint("Game655CompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/655", _matrix_body("23bbd667", 5, 18)),
    ProductCode.LOTTO_535: OfficialEndpoint("Game535CompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/535", _matrix_body("d0ea794f", 5, 35)),
    ProductCode.MAX3D: OfficialEndpoint("GameMax3DCompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/max-3D", _three_digit_body("5")),
    ProductCode.MAX3D_PRO: OfficialEndpoint("GameMax3DProCompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/max-3DPro", _three_digit_body("7")),
    ProductCode.KENO: OfficialEndpoint("GameKenoCompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/winning-number-keno", _keno_body),
    ProductCode.BINGO18: OfficialEndpoint("GameBingoCompareWebPart", "/vi/trung-thuong/ket-qua-trung-thuong/winning-number-bingo18", _bingo_body),
}
_GAME_TO_PRODUCT = {GameCode.MEGA_645: ProductCode.MEGA_645, GameCode.POWER_655: ProductCode.POWER_655, GameCode.LOTTO_535: ProductCode.LOTTO_535}

_INT = re.compile(r"^\d{1,2}$")
_DMY = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_DRAW = re.compile(r"(?:kỳ\s*(?:quay\s*)?(?:thưởng)?|#)\s*#?\s*(\d{1,10})", re.I)


def _date(text: str) -> str | None:
    m = _DMY.search(text)
    return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date().isoformat() if m else None


def _draw_id(text: str) -> int | None:
    m = _DRAW.search(text)
    if m:
        return int(m.group(1))
    t = text.strip().lstrip("#")
    return int(t) if t.isdigit() else None


# ------------------------------------------------------------------ list parsers
def parse_results_html(html: str) -> list[OfficialRow]:
    """Mega / Power / Lotto history rows: ``date | id | numbers`` (header row skipped)."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[OfficialRow] = []
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue  # header (th) or malformed row
        values = [int(s.get_text(strip=True)) for s in tds[2].find_all("span") if _INT.match(s.get_text(strip=True))]
        rows.append(OfficialRow(date_text=tds[0].get_text(strip=True), id_text=tds[1].get_text(strip=True), values=values))
    return rows


def parse_keno_html(html: str) -> list[dict]:
    """Keno rows: ``[date, #id] | 20 numbers | lớn/nhỏ | chẵn/lẻ``."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        links = tds[0].find_all("a")
        text0 = " ".join(a.get_text(" ", strip=True) for a in links) or tds[0].get_text(" ", strip=True)
        d = _date(text0)
        did = _draw_id(links[1].get_text(strip=True) if len(links) > 1 else text0.split()[-1])
        nums = [int(s.get_text(strip=True)) for s in tds[1].find_all("span") if s.get_text(strip=True).isdigit()]
        if d and did and nums:
            row = {"id": did, "date": d, "result": nums}
            if len(tds) > 3:
                row["side_1"], row["side_2"] = tds[2].get_text(" ", strip=True), tds[3].get_text(" ", strip=True)
            out.append(row)
    return out


def parse_bingo18_html(html: str) -> list[dict]:
    """Bingo18 rows: ``[date, #id] | 3 numbers | total | lớn/nhỏ/hòa``."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        links = tds[0].find_all("a")
        text0 = " ".join(a.get_text(" ", strip=True) for a in links) or tds[0].get_text(" ", strip=True)
        d = _date(text0)
        did = _draw_id(links[1].get_text(strip=True) if len(links) > 1 else text0.split()[-1])
        nums = [int(s.get_text(strip=True)) for s in tds[1].find_all("span") if s.get_text(strip=True).isdigit()]
        if d and did and len(nums) == 3:
            out.append({"id": did, "date": d, "result": nums})
    return out


def parse_three_digit_html(html: str) -> list[dict]:
    """Max 3D / Max 3D Pro rows: 20 three-digit numbers in ``.tong_day_so_ket_qua``
    (one digit per ``span.bong_tron`` or one number per span), ordered 2 / 4 / 6 / 8."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for box in soup.select(".tong_day_so_ket_qua"):
        row = box.find_parent("tr") or box
        text = row.get_text(" ", strip=True)
        d = _date(text)
        did = _draw_id(text)
        if did is None and isinstance(row, Tag) and row.find("a"):
            did = _draw_id(row.find("a").get_text(strip=True))
        spans = [s.get_text(strip=True) for s in box.find_all("span", class_="bong_tron")]
        if spans and all(len(t) == 1 and t.isdigit() for t in spans):
            values = ["".join(spans[i : i + 3]) for i in range(0, len(spans) - 2, 3)]
        else:
            values = [t for t in spans if re.fullmatch(r"\d{3}", t)] or re.findall(r"(?<!\d)\d{3}(?!\d)", box.get_text(" "))
        if d and did and len(values) >= 20:
            values = values[:20]
            tiers, cur = [], 0
            for code, n in THREE_DIGIT_LAYOUT:
                tiers.append({"code": code, "numbers": values[cur : cur + n]})
                cur += n
            out.append({"draw_id": f"{did:05d}", "draw_date": d, "result": {"kind": "three_digit_tiers", "tiers": tiers}})
    return out


# ------------------------------------------------------------------ detail page
def _cell_int(cells: list, i: int | None) -> int | None:  # type: ignore[type-arg]
    """Digits of cell ``i`` as an int ("1.234.567 đ" → 1234567); None if absent/empty."""
    if i is None or i >= len(cells):
        return None
    digits = re.sub(r"\D", "", cells[i].get_text(" ", strip=True))
    return int(digits) if digits else None


def parse_prize_table(html: str) -> list[dict]:
    """Prize table of a detail page → ``[{code, name, winner_count, amount_vnd, jackpot_vnd}]``.

    The header row is located by its labels (``Giải thưởng`` / ``Số lượng giải`` /
    ``Giá trị giải``), so column order and extra columns do not matter."""
    soup = BeautifulSoup(html, "html.parser")
    prizes: dict[str, dict] = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        header = None
        for pos, tr in enumerate(rows):
            heads = [fold(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            pi = next((i for i, h in enumerate(heads) if "giai" in h and "so luong" not in h and "gia tri" not in h), None)
            ai = next((i for i, h in enumerate(heads) if "gia tri" in h or "muc thuong" in h), None)
            wi = next((i for i, h in enumerate(heads) if "so luong" in h or "so ve" in h), None)
            if pi is not None and (ai is not None or wi is not None):
                header = (pos, pi, ai, wi)
                break
        if header is None:
            continue
        pos, pi, ai, wi = header
        for tr in rows[pos + 1 :]:
            cells = tr.find_all(["th", "td"])
            if pi >= len(cells):
                continue
            name = cells[pi].get_text(" ", strip=True)
            f = fold(name)
            if not name or ("giai" not in f and "jackpot" not in f and "doc dac" not in f):
                continue
            value, winners = _cell_int(cells, ai), _cell_int(cells, wi)
            is_jp = "jackpot" in f or "doc dac" in f
            prizes[slug(name)] = {"code": slug(name), "name": name, "winner_count": winners, "amount_vnd": None if is_jp else value, "jackpot_vnd": value if is_jp else None}
    return list(prizes.values())


def parse_detail_page(html: str, url: str) -> dict:
    """Draw id, date, prize table and official PDF link of a draw-detail page."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    pdf = None
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if ".pdf" in href.lower():
            pdf = href if href.startswith("http") else "https://media.vietlott.vn" + href if href.startswith("/") else href
            break
    numbers: list[int] = []
    box = soup.select_one(".day_so_ket_qua_v2") or soup.select_one(".day_so_ket_qua")
    if box is not None:
        numbers = [int(s.get_text(strip=True)) for s in box.find_all("span") if s.get_text(strip=True).isdigit()]
    return {
        "draw_id": _draw_id(text),
        "draw_date": _date(text),
        "numbers": numbers,
        "prizes": parse_prize_table(html),
        "source_url": url,
        "source_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "source_pdf_url": pdf,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ------------------------------------------------------------------ client
class VietlottClientMixin:
    client: AsyncHttpClient
    base_url: str
    bootstrap_cookie: bool
    _cookie_ready: bool

    async def _ensure_cookie(self) -> None:
        if not self.bootstrap_cookie or self._cookie_ready:
            return
        self._cookie_ready = True
        try:
            resp = await self.client.get(self.base_url + "/ajaxpro/")
        except SourceError as exc:
            err = _access_error(exc)
            if err is not exc:
                raise err from exc
            log.warning("cookie bootstrap failed (continuing without it): %s", exc)
            return
        m = re.search(r'document\.cookie\s*=\s*["\']([^"\']+)', resp.text)
        if m and "=" in m.group(1).split(";", 1)[0]:
            name, value = m.group(1).split(";", 1)[0].split("=", 1)
            self.client.set_cookie(name, value)

    async def _ajax(self, ep: OfficialEndpoint, page: int) -> str:
        await self._ensure_cookie()
        try:
            response = await self.client.post(
                self.base_url + ep.path,
                json=ep.body(page),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "X-AjaxPro-Method": "ServerSideDrawResult",
                    "X-Requested-With": "XMLHttpRequest",
                    "Origin": self.base_url,
                    "Referer": self.base_url + ep.detail_path,
                },
            )
        except SourceError as exc:
            raise _access_error(exc) from exc
        try:
            return response.json()["value"]["HtmlContent"] or ""
        except (ValueError, KeyError, TypeError) as exc:
            raise SourceError(f"unexpected AjaxPro payload on page {page}: {response.text[:200]!r}") from exc

    async def fetch_detail(self, product: ProductCode, draw_id: int) -> dict:
        ep = ENDPOINTS[product]
        url = f"{self.base_url}{ep.detail_path}?id={draw_id:05d}&nocatche=1"
        try:
            resp = await self.client.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
        except SourceError as exc:
            raise _access_error(exc) from exc
        detail = parse_detail_page(resp.text, url)
        if detail["draw_id"] not in (None, draw_id):
            raise SourceError(f"detail page {url} shows draw {detail['draw_id']}, expected {draw_id}")
        detail["draw_id"] = draw_id
        return detail


class VietlottOfficialSource(VietlottClientMixin, DrawSource):
    """Mega / Power / Lotto: pages backwards through the results list until ``since_id``."""

    name = "vietlott"

    def __init__(self, client: AsyncHttpClient, base_url: str = WEB_BASE, max_pages: int = 400, pages_per_batch: int = 4, bootstrap_cookie: bool = False) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self.pages_per_batch = pages_per_batch
        self.bootstrap_cookie = bootstrap_cookie
        self._cookie_ready = False

    async def fetch_page(self, spec: GameSpec, page: int) -> list[OfficialRow]:
        return parse_results_html(await self._ajax(ENDPOINTS[_GAME_TO_PRODUCT[spec.code]], page))

    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        if spec.code not in _GAME_TO_PRODUCT:
            raise SourceError(f"no official endpoint configured for {spec.code.value}")
        result = FetchResult()
        seen: set[int] = set()
        page = 0
        while page < self.max_pages:
            batch = list(range(page, min(page + self.pages_per_batch, self.max_pages)))
            pages = await asyncio.gather(*(self.fetch_page(spec, p) for p in batch))
            reached_end = False
            for rows in pages:
                if not rows:
                    reached_end = True
                for row in rows:
                    try:
                        draw = row.to_draw(spec, source=self.name)
                    except (ValidationError, ValueError) as exc:
                        result.rejected.append((row.model_dump_json(), str(exc).splitlines()[0]))
                        continue
                    if since_id is not None and draw.draw_id <= since_id:
                        reached_end = True
                        continue
                    if draw.draw_id not in seen:
                        seen.add(draw.draw_id)
                        result.draws.append(draw)
            log.info("pages %s done, %d new draws so far", batch, len(result.draws))
            if reached_end:
                break
            page += self.pages_per_batch
        result.draws.sort(key=lambda d: d.draw_id)
        return result

    async def fetch_canonical(self, spec: GameSpec, draw_ids: list[int]) -> list[dict]:
        """Detail pages of the given draws → canonical records (result + prize table + provenance)."""
        product = _GAME_TO_PRODUCT[spec.code]
        details = await asyncio.gather(*(self.fetch_detail(product, i) for i in draw_ids))
        out = []
        for d in details:
            nums = d["numbers"]
            main, bonus = nums[: spec.pick], nums[spec.pick : spec.pick + 1] if spec.has_bonus else []
            out.append(
                {
                    "draw_id": f"{d['draw_id']:05d}",
                    "draw_date": d["draw_date"],
                    "game": spec.code.value,
                    "result": {"kind": "number_set", "main_numbers": main, "bonus_numbers": bonus},
                    "prizes": d["prizes"],
                    "source_url": d["source_url"],
                    "source_sha256": d["source_sha256"],
                    "source_pdf_url": d["source_pdf_url"],
                    "retrieved_at": d["retrieved_at"],
                    "schema_version": "1.0",
                }
            )
        return out


class VietlottProductSource(VietlottClientMixin):
    """Keno, Bingo18, Max 3D and Max 3D Pro history lists (newest page first)."""

    PARSERS: dict[ProductCode, Callable[[str], list[dict]]] = {
        ProductCode.KENO: parse_keno_html,
        ProductCode.BINGO18: parse_bingo18_html,
        ProductCode.MAX3D: parse_three_digit_html,
        ProductCode.MAX3D_PRO: parse_three_digit_html,
    }

    def __init__(self, client: AsyncHttpClient, base_url: str = WEB_BASE, pages_per_batch: int = 4, bootstrap_cookie: bool = True) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.pages_per_batch = pages_per_batch
        self.bootstrap_cookie = bootstrap_cookie
        self._cookie_ready = False

    async def fetch_pages(self, product: ProductCode, max_pages: int, since_id: int | None = None, first_page: int = 1) -> list[dict]:
        """Rows from the newest page backwards; stops at ``since_id`` or an empty page."""
        if product not in self.PARSERS:
            raise SourceError(f"{product.value} is a matrix game: use VietlottOfficialSource")
        parser = self.PARSERS[product]
        ep = ENDPOINTS[product]
        rows: dict[int, dict] = {}
        page = first_page
        while page < first_page + max_pages:
            batch = list(range(page, min(page + self.pages_per_batch, first_page + max_pages)))
            htmls = await asyncio.gather(*(self._ajax(ep, p) for p in batch))
            stop = False
            for html in htmls:
                parsed = parser(html)
                if not parsed:
                    stop = True
                for r in parsed:
                    did = int(str(r.get("id", r.get("draw_id"))).lstrip("#"))
                    if since_id is not None and did <= since_id:
                        stop = True
                        continue
                    rows[did] = r
            log.info("%s pages %s: %d rows so far", product.value, batch, len(rows))
            if stop:
                break
            page += self.pages_per_batch
        return [rows[i] for i in sorted(rows)]
