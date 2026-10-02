"""Import vietlott.vn pages that a person saved after passing the site's human check.

vietlott.vn is protected by Cloudflare's "verify you are human" check. The engine does not
try to solve or evade it. Instead a person browses the site normally, passes the check, and
hands the engine what the browser already received:

* **Saved pages** — Ctrl+S in the browser: ``.html`` ("Webpage, HTML only" / "complete") or
  ``.mhtml`` ("Webpage, single file", keeps what the page rendered with JavaScript).
* **A network log** (``.har``) — DevTools ▸ Network, opened before loading the pages, then the
  toolbar's download-arrow button "Export HAR (sanitized)…" (Chrome/Edge 130+; older versions:
  right-click ▸ "Save all as HAR with content"; Firefox: right-click ▸ "Save All As HAR"), while
  paging through a results list: every AjaxPro answer (``{"value": {"HtmlContent": …}}``)
  and every detail page visited is inside.

Pages are classified by their URL (Chrome writes ``<!-- saved from url=… -->`` into saved
HTML; MHTML and HAR keep the URL), then parsed with the same parsers as the live crawler:
results lists and AjaxPro answers → draws; draw-detail pages → draws with prize tables
(Mega / Power / Lotto) and, for Keno, the prize cells of every bậc and side bet.

The parsers were written against the published structure of the site; a page they cannot
read is reported with the reason instead of being skipped silently.
"""

from __future__ import annotations

import base64
import email
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email import policy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from vietlott_engine.core.products import ProductCode

DETAIL_PATHS: list[tuple[str, ProductCode]] = [  # order matters: max-3dpro before max-3d
    ("/ket-qua-trung-thuong/645", ProductCode.MEGA_645),
    ("/ket-qua-trung-thuong/655", ProductCode.POWER_655),
    ("/ket-qua-trung-thuong/535", ProductCode.LOTTO_535),
    ("/ket-qua-trung-thuong/max-3dpro", ProductCode.MAX3D_PRO),
    ("/ket-qua-trung-thuong/max-3d", ProductCode.MAX3D),
    ("/ket-qua-trung-thuong/max-4d", ProductCode.MAX4D),
    ("view-detail-keno-result", ProductCode.KENO),
    ("view-detail-bingo18-result", ProductCode.BINGO18),
    ("winning-number-keno", ProductCode.KENO),
    ("winning-number-bingo18", ProductCode.BINGO18),
]
AJAX_PARTS = {
    "game645comparewebpart": ProductCode.MEGA_645,
    "game655comparewebpart": ProductCode.POWER_655,
    "game535comparewebpart": ProductCode.LOTTO_535,
    "gamemax3dprocomparewebpart": ProductCode.MAX3D_PRO,
    "gamemax3dcomparewebpart": ProductCode.MAX3D,
    "gamekenocomparewebpart": ProductCode.KENO,
    "gamebingocomparewebpart": ProductCode.BINGO18,
}
GAME_IDS = {"5": ProductCode.MAX3D, "7": ProductCode.MAX3D_PRO, "6": ProductCode.KENO, "8": ProductCode.BINGO18}
MATRIX = {ProductCode.MEGA_645: "mega645", ProductCode.POWER_655: "power655", ProductCode.LOTTO_535: "lotto535"}


@dataclass
class SavedDoc:
    origin: str  # file (and HAR entry)
    url: str | None
    body: str
    ajax: bool = False
    post: str | None = None


@dataclass
class PageImport:
    """What the saved pages contain, per product."""

    product_rows: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))  # Keno, Bingo18, Max 3D/Pro/4D
    canonical: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))  # Mega/Power/Lotto (detail pages: + prizes)
    keno_cells: list[dict] = field(default_factory=list)  # prize cells of Keno detail pages
    pages: list[dict] = field(default_factory=list)  # one line per document: product, kind, rows, problem
    cloudflare_pages: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        kinds = Counter((p["product"] or "?", p["kind"]) for p in self.pages)
        return {
            "documents": len(self.pages),
            "parsed": sum(1 for p in self.pages if p["rows"]),
            "unparsed": [p for p in self.pages if not p["rows"]][:20],
            "by_product_kind": {f"{k[0]}:{k[1]}": v for k, v in sorted(kinds.items())},
            "rows": {k: len(v) for k, v in self.product_rows.items()} | {k: len(v) for k, v in self.canonical.items()},
            "cloudflare_challenge_pages": self.cloudflare_pages[:10],
        }


# ------------------------------------------------------------------ reading files
def _saved_from(html: str) -> str | None:
    m = re.search(r"<!--\s*saved from url=\(\d+\)(\S+?)\s*-->", html[:2000], re.I)
    if m:
        return m.group(1)
    soup = BeautifulSoup(html[:200_000], "html.parser")
    for sel, attr in (("link[rel=canonical]", "href"), ("meta[property='og:url']", "content")):
        tag = soup.select_one(sel)
        if tag is not None and tag.get(attr):
            return str(tag.get(attr))
    return None


def read_mhtml(raw: bytes, origin: str) -> list[SavedDoc]:
    msg = email.message_from_bytes(raw, policy=policy.default)
    url = msg.get("Snapshot-Content-Location") or msg.get("Content-Location")
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            raw_body = part.get_payload(decode=True) or b""
            charset = part.get_content_charset()
            if not charset:  # Chrome may omit it: use the page's own <meta charset>, else UTF-8
                m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", raw_body[:4000], re.I)
                charset = m.group(1).decode("ascii") if m else "utf-8"
            try:
                body = raw_body.decode(charset, errors="replace")
            except LookupError:
                body = raw_body.decode("utf-8", errors="replace")
            return [SavedDoc(origin, part.get("Content-Location") or url or _saved_from(body), body)]
    return []


def read_har(text: str, origin: str) -> list[SavedDoc]:
    docs = []
    for i, e in enumerate(json.loads(text).get("log", {}).get("entries", [])):
        req, resp = e.get("request", {}), e.get("response", {})
        content = resp.get("content", {}) or {}
        body = content.get("text") or ""
        if content.get("encoding") == "base64" and body:
            try:
                body = base64.b64decode(body).decode("utf-8", "replace")
            except ValueError:
                continue
        url = req.get("url", "")
        if not body or "vietlott" not in url:
            continue
        mime = (content.get("mimeType") or "").lower()
        ajax = "/ajaxpro/" in url.lower()
        if ajax or "html" in mime or body.lstrip().startswith("<"):
            docs.append(SavedDoc(f"{origin}#{i}", url, body, ajax=ajax, post=(req.get("postData") or {}).get("text")))
    return docs


def har_diagnosis(text: str) -> str:
    """Why a HAR held no vietlott.vn page: say what it does hold and how to record one that works."""
    entries = json.loads(text).get("log", {}).get("entries", [])
    on_site = [e for e in entries if "vietlott.vn" in e.get("request", {}).get("url", "")]
    with_body = [e for e in on_site if (e.get("response", {}).get("content", {}) or {}).get("text")]
    pages = sorted({e["request"]["url"].split("?")[0] for e in on_site if "/trung-thuong/" in e["request"]["url"] or "/ajaxpro/" in e["request"]["url"].lower()})
    return (
        f"HAR has {len(entries)} requests, {len(on_site)} to vietlott.vn, {len(with_body)} with a body, "
        f"none of them a result page or AjaxPro answer{(' (result URLs without content: ' + ', '.join(pages[:3]) + ')') if pages else ''}. "
        "Set the Network filter to 'All' (the export keeps only the listed requests), open DevTools before loading the pages, "
        "open the result pages themselves (…/trung-thuong/ket-qua-trung-thuong/…), then export again — or save each page with Ctrl+S as .mhtml"
    )


def load_documents(path: Path | str) -> tuple[list[SavedDoc], list[dict]]:
    """Every saved page under ``path`` (a file or a folder, recursively)."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    docs, unreadable = [], []
    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix in (".mhtml", ".mht"):
                docs += read_mhtml(f.read_bytes(), f.name)
            elif suffix == ".har":
                text = f.read_text(encoding="utf-8", errors="replace")
                found = read_har(text, f.name)
                docs += found
                if not found:
                    unreadable.append({"file": f.name, "error": har_diagnosis(text)})
            elif suffix in (".html", ".htm"):
                body = f.read_text(encoding="utf-8", errors="replace")
                docs.append(SavedDoc(f.name, _saved_from(body), body))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            unreadable.append({"file": f.name, "error": str(exc).splitlines()[0][:200]})
    return docs, unreadable


# ------------------------------------------------------------------ classification
def classify(doc: SavedDoc) -> tuple[ProductCode | None, str]:
    """(product, kind) with kind ∈ {"ajax", "detail", "list", "unknown"}."""
    url = (doc.url or "").lower()
    if doc.ajax or "/ajaxpro/" in url:
        m = re.search(r"webparts\.([a-z0-9]+),", url)
        product = AJAX_PARTS.get(m.group(1)) if m else None
        if product is None and doc.post:
            g = re.search(r'"GameId"\s*:\s*"?(\d+)', doc.post)
            product = GAME_IDS.get(g.group(1)) if g else None
        return product, "ajax"
    for frag, product in DETAIL_PATHS:
        if frag in url:
            has_id = "id" in parse_qs(urlparse(doc.url or "").query)
            return product, "detail" if has_id or "view-detail" in url else "list"
    return None, "unknown"


def _draw_id_from_url(url: str | None) -> int | None:
    q = parse_qs(urlparse(url or "").query)
    v = (q.get("id") or [None])[0]
    return int(v) if v and v.isdigit() else None


# ------------------------------------------------------------------ detail-page readers
def _number_tokens(soup: BeautifulSoup, width: int | None) -> list[str]:
    """Number tokens inside the result boxes (``bong_tron`` / ``day_so`` / ``ket_qua`` elements)."""
    boxes = soup.select("[class*=day_so], [class*=ket_qua], [class*=bong_tron]")
    out: list[str] = []
    for el in boxes:
        if el.find(class_=re.compile("bong_tron")) is not None:
            continue  # take the innermost elements only
        t = el.get_text("", strip=True)
        if re.fullmatch(r"\d+", t) and (width is None or len(t) == width):
            out.append(t)
    return out


def keno_cells(html: str) -> list[dict]:
    """Prize cells of a Keno detail page: bậc (from the largest 'Trùng NN' of its table),
    hits and value; side bets by label."""
    soup = BeautifulSoup(html, "html.parser")
    cells = []
    for ti, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(tds) < 2:
                continue
            label = next((t for t in tds if re.search(r"Trùng \d+ trong 20 số|số chẵn|số lẻ|Lớn|Nhỏ|Hòa", t)), None)
            money = next((t for t in tds if re.search(r"\d[\d.]*\s*(đ|đồng)", t)), None)
            if label and money:
                value = int(re.sub(r"\D", "", re.match(r"\s*([\d.]+)", money).group(1)) or 0)
                rows.append((label, value))
        hits = [int(m.group(1)) for lab, _ in rows if (m := re.match(r"Trùng (\d+) trong 20 số", lab))]
        bac = max(hits) if hits else None
        for lab, value in rows:
            m = re.match(r"Trùng (\d+) trong 20 số", lab)
            cells.append({"table": ti, "bac": bac if m else None, "hits": int(m.group(1)) if m else None, "label": lab, "value": value})
    return cells


def parse_detail(doc: SavedDoc, product: ProductCode) -> tuple[list[dict], str | None]:
    from vietlott_engine.crawler.sources.vietlott_official import _date, parse_detail_page

    soup = BeautifulSoup(doc.body, "html.parser")
    text = soup.get_text(" ", strip=True)
    did = _draw_id_from_url(doc.url)
    d = _date(text)
    if product in MATRIX:
        det = parse_detail_page(doc.body, doc.url or doc.origin)
        did = did or det["draw_id"]
        from vietlott_engine.core.games import get_game

        spec = get_game(MATRIX[product])
        nums = det["numbers"] or [int(t) for t in _number_tokens(soup, None) if 1 <= int(t) <= spec.pool_size]
        need = spec.pick + (1 if spec.has_bonus else 0)
        if not did or not det["draw_date"] or len(nums) < need:
            return [], f"detail page: draw id {did}, date {det['draw_date']}, {len(nums)} numbers (need {need})"
        rec = {
            "draw_id": f"{did:05d}",
            "draw_date": det["draw_date"],
            "game": MATRIX[product],
            "result": {"kind": "number_set", "main_numbers": nums[: spec.pick], "bonus_numbers": nums[spec.pick : need] if spec.has_bonus else []},
            "prizes": det["prizes"],
            "source_url": doc.url,
            "source_sha256": det["source_sha256"],
            "source_pdf_url": det["source_pdf_url"],
            "data_source": None,  # official page
        }
        return [rec], None if det["prizes"] else "no prize table found (draw kept)"
    if product == ProductCode.KENO:
        vals = sorted({int(t) for t in _number_tokens(soup, None) if 1 <= int(t) <= 80})
        if not did or not d or len(vals) != 20:
            return [], f"Keno detail: draw id {did}, date {d}, {len(vals)} distinct numbers 1–80 (need 20)"
        return [{"id": did, "date": d, "result": vals}], None
    if product == ProductCode.BINGO18:
        vals = [int(t) for t in _number_tokens(soup, 1) if 1 <= int(t) <= 6][:3]
        if not did or not d or len(vals) != 3:
            return [], f"Bingo18 detail: draw id {did}, date {d}, {len(vals)} numbers 1–6 (need 3)"
        return [{"id": did, "date": d, "result": vals}], None
    width, n = (4, 6) if product == ProductCode.MAX4D else (3, 20)
    toks = _number_tokens(soup, width)
    if len(toks) < n:  # digits rendered one per ball
        digits = _number_tokens(soup, 1)
        toks = ["".join(digits[i : i + width]) for i in range(0, len(digits) - width + 1, width)]
    if not did or not d or len(toks) < n:
        return [], f"{product.value} detail: draw id {did}, date {d}, {len(toks)} numbers of {width} digits (need {n})"
    return [{"id": did, "date": d, "result": toks[:n]}], None


def parse_list(html: str, product: ProductCode) -> list[dict]:
    from vietlott_engine.crawler.sources.vietlott_official import parse_bingo18_html, parse_keno_html, parse_results_html, parse_three_digit_html

    if product == ProductCode.KENO:
        return parse_keno_html(html)
    if product == ProductCode.BINGO18:
        return parse_bingo18_html(html)
    if product in (ProductCode.MAX3D, ProductCode.MAX3D_PRO):
        return parse_three_digit_html(html)
    if product in MATRIX:
        from vietlott_engine.core.games import get_game

        spec = get_game(MATRIX[product])
        out = []
        for row in parse_results_html(html):
            try:
                dr = row.to_draw(spec, source="vietlott.vn")
            except ValueError:
                continue
            out.append({"draw_id": f"{dr.draw_id:05d}", "draw_date": dr.draw_date.isoformat(), "game": MATRIX[product], "result": {"kind": "number_set", "main_numbers": list(dr.numbers), "bonus_numbers": [dr.bonus] if dr.bonus else []}, "prizes": [], "data_source": None})
        return out
    return []


def import_pages(path: Path | str, product: ProductCode | None = None) -> PageImport:
    docs, unreadable = load_documents(path)
    out = PageImport()
    for u in unreadable:
        out.pages.append({"document": u["file"], "url": None, "product": None, "kind": "unreadable", "rows": 0, "problem": u["error"]})
    for doc in docs:
        low = doc.body[:4000].lower()
        if "just a moment" in low or "cf-chl" in low or "challenge-platform" in low:
            out.cloudflare_pages.append(doc.origin)
            out.pages.append({"document": doc.origin, "url": doc.url, "product": None, "kind": "cloudflare", "rows": 0, "problem": "the saved page is the Cloudflare check itself — save it again after the result page has loaded"})
            continue
        prod, kind = classify(doc)
        prod = product or prod
        rows: list[dict] = []
        problem = None
        if prod is None:
            problem = "could not tell which product this page belongs to (no vietlott.vn URL in the file)"
        elif kind == "ajax":
            try:
                html = json.loads(doc.body)["value"]["HtmlContent"] or ""
            except (ValueError, KeyError, TypeError):
                html, problem = "", "AjaxPro answer without value.HtmlContent"
            rows = parse_list(html, prod) if html else []
        elif kind == "detail":
            rows, problem = parse_detail(doc, prod)
            if prod == ProductCode.KENO:
                cells = keno_cells(doc.body)
                did = _draw_id_from_url(doc.url)
                out.keno_cells += [c | {"draw_id": did} for c in cells]
        else:
            rows = parse_list(doc.body, prod)
        if not rows and problem is None:
            problem = "no result rows recognised"
        if prod in MATRIX:
            out.canonical[prod.value] += rows
        elif prod is not None:
            out.product_rows[prod.value] += rows
        out.pages.append({"document": doc.origin, "url": doc.url, "product": prod.value if prod else None, "kind": kind, "rows": len(rows), "problem": problem})
    return out


# ------------------------------------------------------------------ verification
def _upper_bound(k: int, n: int, conf: float = 0.95) -> float | None:
    """One-sided Clopper–Pearson upper bound of a mismatch rate (k of n)."""
    if n == 0:
        return None
    from scipy import stats

    return 1.0 if k >= n else float(stats.beta.ppf(conf, k + 1, n - k))


def compare_products(imp: PageImport, store) -> dict:  # type: ignore[no-untyped-def]
    """Saved official pages vs the engine's data (seed + store, unconfirmed included)."""
    from vietlott_engine.core.products import parse_product_rows

    out = {}
    for name, rows in imp.product_rows.items():
        code = ProductCode(name)
        page, _ = parse_product_rows(code, rows, "pages")
        have = store.load(code, include_unconfirmed=True)
        known = {int(i): (str(d), v.tolist()) for i, d, v in zip(have.draw_ids, have.dates, have.values)}
        cmp_ids = [int(i) for i in page.draw_ids if int(i) in known]
        pv = {int(i): (str(d), v.tolist()) for i, d, v in zip(page.draw_ids, page.dates, page.values)}
        bad = [i for i in cmp_ids if pv[i] != known[i]]
        out[name] = {
            "pages_rows": len(page),
            "compared": len(cmp_ids),
            "identical": len(cmp_ids) - len(bad),
            "mismatches": [{"draw_id": i, "page": pv[i], "data": known[i]} for i in bad[:10]],
            "mismatch_rate_upper95": _upper_bound(len(bad), len(cmp_ids)),
            "mismatch_ids": bad,
            "new_draws": sorted(set(pv) - set(known))[:50],
        }
    return out


def compare_matrix(imp: PageImport, repository) -> dict:  # type: ignore[no-untyped-def]
    from vietlott_engine.core.games import GameCode
    from vietlott_engine.crawler.official_data import import_canonical

    out = {}
    for name, recs in imp.canonical.items():
        game = GameCode(name)
        got = import_canonical(recs, game)
        have = {d.draw_id: d for d in repository.load(game)}
        prizes = {p.draw_id: p for p in repository.load_prizes(game)}
        draw_bad = [d.draw_id for d in got.draws if d.draw_id in have and (d.numbers, d.bonus) != (have[d.draw_id].numbers, have[d.draw_id].bonus)]
        cmp_p = [p for p in got.prizes if p.draw_id in prizes]
        prize_bad = [p.draw_id for p in cmp_p if (p.winners, p.jackpot_pots) != (prizes[p.draw_id].winners, prizes[p.draw_id].jackpot_pots)]
        out[name] = {
            "draws": len(got.draws),
            "draws_compared": sum(d.draw_id in have for d in got.draws),
            "draw_mismatches": draw_bad[:10],
            "prize_tables": len(got.prizes),
            "prize_tables_compared": len(cmp_p),
            "prize_mismatches": [
                {"draw_id": i, "page": {"winners": next(p for p in cmp_p if p.draw_id == i).winners, "pots": next(p for p in cmp_p if p.draw_id == i).jackpot_pots}, "data": {"winners": prizes[i].winners, "pots": prizes[i].jackpot_pots, "source": prizes[i].source}}
                for i in prize_bad[:10]
            ],
            "mismatch_ids": sorted(set(draw_bad) | set(prize_bad)),
            "new_draws": sorted(d.draw_id for d in got.draws if d.draw_id not in have),
        }
    return out


def compare_keno_cells(imp: PageImport) -> dict:
    """Keno prize cells read from the pages vs the engine's table (shows the bậc 5 / 4-hit cell)."""
    from vietlott_engine.game_theory.fastgames import KENO_TABLE

    diffs, seen = [], 0
    for c in imp.keno_cells:
        if c["bac"] and c["hits"] is not None and c["bac"] in KENO_TABLE:
            seen += 1
            ours = KENO_TABLE[c["bac"]].get(c["hits"], 0)
            if ours != c["value"]:
                diffs.append({"draw_id": c["draw_id"], "bac": c["bac"], "hits": c["hits"], "page": c["value"], "engine": ours})
    labels = sorted({(c["label"], c["value"]) for c in imp.keno_cells if c["hits"] is None})
    return {"cells_compared": seen, "differences": diffs[:30], "side_bet_cells": [{"label": a, "value": b} for a, b in labels]}
