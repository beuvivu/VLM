"""v3.4: Cloudflare challenge handling (detect, never bypass) and import of pages a person
saved from vietlott.vn (HTML, MHTML, HAR) with comparison against the engine's data."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import quopri
from pathlib import Path

import httpx
import pytest

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.products import ProductCode
from vietlott_engine.crawler.http import CF_CHALLENGE_PREFIX, is_cloudflare_challenge
from vietlott_engine.crawler.product_store import ProductStore
from vietlott_engine.crawler.sources.saved_pages import classify, compare_keno_cells, compare_matrix, compare_products, import_pages, keno_cells, SavedDoc
from vietlott_engine.crawler.sources.vietlott_official import CLOUDFLARE_HINT, VietlottProductSource
from vietlott_engine.crawler.storage import InMemoryRepository
from tests.test_all_products import DETAIL_645, _client, _keno_html, _three_digit_html

ROOT = Path(__file__).resolve().parents[1]
CF_PAGE = "<!DOCTYPE html><html><head><title>Just a moment...</title></head><body><div id='challenge-platform'></div><script>window._cf_chl_opt={}</script></body></html>"
MEGA_URL = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/645?id=01569&nocatche=1"
KENO_AJAX = "https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.GameKenoCompareWebPart,Vietlott.PlugIn.WebParts.ashx"


def _mhtml(url: str, html: str) -> str:
    qp = quopri.encodestring(html.encode("utf-8")).decode("ascii")
    return (
        f"From: <Saved by Blink>\r\nSnapshot-Content-Location: {url}\r\nMIME-Version: 1.0\r\n"
        'Content-Type: multipart/related; type="text/html"; boundary="----B"\r\n\r\n'
        f"------B\r\nContent-Type: text/html\r\nContent-Transfer-Encoding: quoted-printable\r\nContent-Location: {url}\r\n\r\n{qp}\r\n------B--\r\n"
    )


def _har(entries: list[tuple[str, str, dict | None]]) -> str:
    return json.dumps({"log": {"entries": [{"request": {"url": u, "postData": {"text": json.dumps(p)} if p else {}}, "response": {"content": {"mimeType": "application/json" if "ajaxpro" in u else "text/html", "text": t}}} for u, t, p in entries]}})


def keno_detail_html(did: int, nums: list[int]) -> str:
    balls = "".join(f'<span class="bong_tron">{n:02d}</span>' for n in nums)
    bac5 = "".join(f"<tr><td>Trùng {h:02d} trong 20 số</td><td>{v} đ ( Số lượng: 3 )</td></tr>" for h, v in ((5, "4.400.000"), (4, "0"), (3, "10.000")))
    bac1 = "<tr><td>Trùng 01 trong 20 số</td><td>20.000 đ</td></tr>"
    side = "<tr><td>Lớn từ 13 số trở lên từ 41-80</td><td>26.000 đồng</td></tr>"
    return f"<html><body><h5>Kỳ quay thưởng #{did:07d} ngày 01/10/2026</h5><div class='day_so_ket_qua'>{balls}</div><table>{bac5}</table><table>{bac1}</table><table>{side}</table></body></html>"


# ------------------------------------------------------------------ Cloudflare
def test_cloudflare_challenge_is_detected_and_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403, headers={"server": "cloudflare", "cf-mitigated": "challenge"}, text=CF_PAGE)

    async def run() -> tuple[str, str]:
        async with _client(handler) as c:
            with pytest.raises(SourceError) as e1:
                await c.get("https://www.vietlott.vn/vi/")
            with pytest.raises(SourceError) as e2:
                await VietlottProductSource(c, bootstrap_cookie=False).fetch_pages(ProductCode.KENO, 1)
            return str(e1.value), str(e2.value)

    raw, mapped = asyncio.run(run())
    assert CF_CHALLENGE_PREFIX in raw and mapped == CLOUDFLARE_HINT
    assert len(calls) == 2  # one request each: never retried
    assert is_cloudflare_challenge(httpx.Response(503, headers={"server": "cloudflare"}, text=CF_PAGE))
    assert not is_cloudflare_challenge(httpx.Response(503, headers={"server": "nginx"}, text="busy"))  # ordinary 503s are still retried


# ------------------------------------------------------------------ saved pages
def test_import_pages_all_formats(tmp_path: Path) -> None:
    # MHTML already has CRLF; text-mode Windows output would turn it into CRCRLF.
    (tmp_path / "mega.mhtml").write_bytes(_mhtml(MEGA_URL, DETAIL_645).encode("ascii"))
    (tmp_path / "keno_list.har").write_text(_har([(KENO_AJAX, json.dumps({"value": {"HtmlContent": _keno_html([("01/10/2026", 297700, list(range(1, 21)))])}}), {"GameId": "6", "PageIndex": 1})]), encoding="utf-8")
    nums = [3, 7, 11, 15, 19, 22, 26, 30, 33, 38, 41, 44, 47, 52, 56, 60, 63, 69, 72, 80]
    (tmp_path / "keno_detail.html").write_text("<!-- saved from url=(0090)https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297701 -->\n" + keno_detail_html(297701, nums), encoding="utf-8")
    m3 = [f"{i * 37 % 1000:03d}" for i in range(20)]
    (tmp_path / "max3d.html").write_text("<!-- saved from url=(0070)https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/max-3D -->\n" + _three_digit_html(1139, "30/09/2026", m3), encoding="utf-8")
    (tmp_path / "check.html").write_text("<!-- saved from url=(0030)https://vietlott.vn/vi/ -->\n" + CF_PAGE, encoding="utf-8")
    (tmp_path / "other.html").write_text("<html><body>hello</body></html>", encoding="utf-8")
    imp = import_pages(tmp_path)
    s = imp.summary()
    assert s["by_product_kind"] == {"?:cloudflare": 1, "?:unknown": 1, "keno:ajax": 1, "keno:detail": 1, "max3d:list": 1, "mega645:detail": 1}
    assert s["rows"] == {"keno": 2, "max3d": 1, "mega645": 1} and s["cloudflare_challenge_pages"] == ["check.html"]
    mega = imp.canonical["mega645"][0]
    assert mega["result"]["main_numbers"] == [3, 9, 14, 22, 31, 44] and {p["code"]: p["winner_count"] for p in mega["prizes"]}["giai-ba"] == 35877
    assert {r["id"]: r["result"] for r in imp.product_rows["keno"]}[297701] == nums
    cells = {(c["bac"], c["hits"]): c["value"] for c in imp.keno_cells if c["hits"] is not None}
    assert cells == {(5, 5): 4_400_000, (5, 4): 0, (5, 3): 10_000, (1, 1): 20_000}
    k = compare_keno_cells(imp)
    assert k["differences"] == [{"draw_id": 297701, "bac": 5, "hits": 4, "page": 0, "engine": 150_000}] and k["side_bet_cells"] == [{"label": "Lớn từ 13 số trở lên từ 41-80", "value": 26_000}]


def test_har_without_result_pages_is_explained(tmp_path: Path) -> None:
    # what an export with the Network filter on "Other" looks like: favicon, Cloudflare beacon, widget preflights
    (tmp_path / "x.har").write_text(_har([("https://vietlott.vn/favicon.ico", "", None), ("https://widget.chat.zalo.me/login-tab", "", None)]), encoding="utf-8")
    imp = import_pages(tmp_path)
    (problem,) = [p["problem"] for p in imp.pages if p["kind"] == "unreadable"]
    assert "2 requests, 1 to vietlott.vn, 0 with a body" in problem and "'All'" in problem


def test_classify_urls() -> None:
    def c(url: str, ajax: bool = False, post: str | None = None):  # type: ignore[no-untyped-def]
        return classify(SavedDoc("x", url, "", ajax=ajax, post=post))

    assert c("https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/max-3DPro?id=00786&nocatche=1") == (ProductCode.MAX3D_PRO, "detail")
    assert c("https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/max-3D?id=00786") == (ProductCode.MAX3D, "detail")
    assert c("https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-bingo18") == (ProductCode.BINGO18, "list")
    assert c("https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.Game535CompareWebPart,Vietlott.PlugIn.WebParts.ashx", True) == (ProductCode.LOTTO_535, "ajax")
    assert c("https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.OtherPart,Vietlott.PlugIn.WebParts.ashx", True, '{"GameId":"7"}') == (ProductCode.MAX3D_PRO, "ajax")
    assert keno_cells("<table><tr><td>Trùng 10 trong 20 số</td><td>2.000.000.000 đ</td></tr></table>")[0]["bac"] == 10


def test_compare_with_data_and_cli_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    store = ProductStore(tmp_path / "data", None)
    store.upsert(ProductCode.BINGO18, [{"id": 10, "date": "2026-10-01", "result": [1, 2, 3]}, {"id": 11, "date": "2026-10-01", "result": [4, 5, 6]}], "seed")
    pages = tmp_path / "pages"
    pages.mkdir()
    html = "<table>" + "".join(f"<tr><td><a>01/10/2026</a><a>#{i:07d}</a></td><td>{''.join(f'<span>{n}</span>' for n in r)}</td></tr>" for i, r in ((10, [1, 2, 3]), (11, [6, 6, 6]), (12, [2, 2, 5]))) + "</table>"
    (pages / "b.html").write_text("<!-- saved from url=(0080)https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-bingo18 -->\n" + html, encoding="utf-8")
    imp = import_pages(pages)
    cmp = compare_products(imp, store)["bingo18"]
    assert (cmp["compared"], cmp["identical"], cmp["new_draws"]) == (2, 1, [12]) and cmp["mismatches"][0]["draw_id"] == 11
    assert 0.9 < cmp["mismatch_rate_upper95"] <= 1.0

    repo = InMemoryRepository()
    (pages / "mega.mhtml").write_bytes(_mhtml(MEGA_URL, DETAIL_645).encode("ascii"))
    m = compare_matrix(import_pages(pages), repo)["mega645"]
    assert m["new_draws"] == [1569] and m["prize_tables"] == 1

    from vietlott_engine.cli import main
    from vietlott_engine.core.config import get_settings

    monkeypatch.setenv("VQE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VQE_STORAGE_BACKEND", "memory")
    get_settings.cache_clear()
    try:
        assert main(["products", "import-pages", "--path", str(pages), "--dry-run", "--json"]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["imported"] == {} and rep["products"]["bingo18"]["identical"] == 1
        assert main(["products", "import-pages", "--path", str(pages), "--json"]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["imported"]["bingo18"]["new_draws"] == 1 and rep["imported"]["mega645"]["prize_tables"] == 1
        assert rep["imported"]["bingo18"]["held_back_mismatches"] == 1

        def draw11() -> list[int]:
            h = store.load(ProductCode.BINGO18)
            return h.values[list(h.draw_ids).index(11)].tolist()

        assert draw11() == [4, 5, 6]  # a disagreeing page is shown, not written, by default
        assert main(["products", "import-pages", "--path", str(pages), "--replace-mismatches", "--json"]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["imported"]["bingo18"]["held_back_mismatches"] == 0 and draw11() == [6, 6, 6]  # on request, the official page wins
    finally:
        get_settings.cache_clear()


def test_verification_checklist_is_reproducible() -> None:
    spec = importlib.util.spec_from_file_location("vc", ROOT / "scripts" / "verification_checklist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    a, b = mod.build(seed=7), mod.build(seed=7)
    assert a == b and a[0]["product"] == "lotto535" and a[0]["draw_id"] == 920
    assert sum(r["group"].startswith("2") for r in a) == 30 and all(r["url"].startswith("https://vietlott.vn/") for r in a)
    assert any("bậc 5" in r["why"] for r in a)
