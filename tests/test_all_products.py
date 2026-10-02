"""v3.2: all seven products — parsers, official records, store/sync, Keno/Bingo18 odds,
randomness tests (size and power), Max 3D digit cross-check, Lotto two-phase accrual and
rolldown forecast, API and CLI."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest

from vietlott_engine.analytics.products import digit_replication, product_randomness, split_history
from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import LOTTO_535, MEGA_645
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.prizes import PrizeHistory
from vietlott_engine.core.products import SEED_FILES, ProductCode, ProductHistory, get_product, load_product_history, parse_product_rows
from vietlott_engine.crawler.http import PROXY_BLOCK_PREFIX, AsyncHttpClient, RetryPolicy
from vietlott_engine.crawler.official_data import import_canonical
from vietlott_engine.crawler.product_store import ProductStore, ProductSyncPipeline, sync_canonical_prizes
from vietlott_engine.crawler.sources.vietlott_official import (
    GEO_BLOCK_HINT,
    PROXY_BLOCK_HINT,
    VietlottOfficialSource,
    VietlottProductSource,
    parse_bingo18_html,
    parse_detail_page,
    parse_keno_html,
    parse_prize_table,
    parse_three_digit_html,
)
from vietlott_engine.crawler.storage import InMemoryRepository
from vietlott_engine.game_theory.fastgames import KENO_TABLE, bingo18_odds, bingo18_sum_distribution, keno_hits_distribution, keno_odds, keno_split_distribution
from vietlott_engine.game_theory.rolldown import backtest_rolldown_forecast, forecast_next_rolldown, rolldown_history_official
from tests.test_vietlott_products import _records, _simulate_winners

SEED = Path(__file__).resolve().parents[1] / "data" / "seed"


# ------------------------------------------------------------------ fixtures (HTML)
def _keno_html(rows: list[tuple[str, int, list[int]]]) -> str:
    trs = "".join(
        f"<tr><td><a href='#'>{d}</a> <a href='#'>#{i:07d}</a></td><td>{''.join(f'<span>{n:02d}</span>' for n in nums)}</td><td>Lớn</td><td>Chẵn</td></tr>"
        for d, i, nums in rows
    )
    return f"<table><tr><th>Kỳ</th><th>Kết quả</th></tr>{trs}</table>"


def _bingo_html(rows: list[tuple[str, int, list[int]]]) -> str:
    trs = "".join(
        f"<tr><td><a>{d}</a><a>#{i:07d}</a></td><td>{''.join(f'<span>{n}</span>' for n in nums)}</td><td>{sum(nums)}</td><td>Hòa</td></tr>"
        for d, i, nums in rows
    )
    return f"<table>{trs}</table>"


def _three_digit_html(did: int, d: str, numbers: list[str]) -> str:
    digits = "".join(f'<span class="bong_tron">{c}</span>' for n in numbers for c in n)
    return f'<table><tr><td>Kỳ #{did:05d} ngày {d}</td><td><div class="tong_day_so_ket_qua">{digits}</div></td></tr></table>'


DETAIL_645 = """
<div class="chitietketqua_title"><h5>Kỳ quay thưởng <b>#01569</b> | Ngày quay thưởng <b>30/09/2026</b></h5></div>
<div class="day_so_ket_qua_v2"><span class="bong_tron">03</span><span class="bong_tron">09</span><span class="bong_tron">14</span>
<span class="bong_tron">22</span><span class="bong_tron">31</span><span class="bong_tron">44</span></div>
<table class="table"><thead><tr><th>Giải thưởng</th><th>Trùng khớp</th><th>Số lượng giải</th><th>Giá trị giải (đồng)</th></tr></thead>
<tbody>
<tr><td>Jackpot</td><td>6 số</td><td>2</td><td>79.409.173.250</td></tr>
<tr><td>Giải nhất</td><td>5 số</td><td>45</td><td>10.000.000</td></tr>
<tr><td>Giải nhì</td><td>4 số</td><td>2.105</td><td>300.000</td></tr>
<tr><td>Giải ba</td><td>3 số</td><td>35.877</td><td>30.000</td></tr>
</tbody></table>
<a href="/media/ketqua/645/01569.pdf">Tải biên bản</a>
"""


# ------------------------------------------------------------------ parsers
def test_parse_keno_bingo_three_digit_lists() -> None:
    keno = parse_keno_html(_keno_html([("28/09/2026", 297327, list(range(1, 21))), ("28/09/2026", 297326, list(range(41, 61)))]))
    assert [r["id"] for r in keno] == [297327, 297326]
    assert keno[0]["date"] == "2026-09-28" and keno[0]["result"] == list(range(1, 21)) and keno[0]["side_1"] == "Lớn"
    bingo = parse_bingo18_html(_bingo_html([("28/09/2026", 150001, [1, 4, 6])]))
    assert bingo == [{"id": 150001, "date": "2026-09-28", "result": [1, 4, 6]}]
    nums = [f"{i * 37 % 1000:03d}" for i in range(20)]
    m3 = parse_three_digit_html(_three_digit_html(1139, "30/09/2026", nums))
    assert len(m3) == 1 and m3[0]["draw_id"] == "01139" and m3[0]["draw_date"] == "2026-09-30"
    tiers = m3[0]["result"]["tiers"]
    assert [len(t["numbers"]) for t in tiers] == [2, 4, 6, 8] and [x for t in tiers for x in t["numbers"]] == nums
    h, rejected = parse_product_rows(ProductCode.MAX3D, m3, "test")
    assert not rejected and h.values[0].tolist() == [int(x) for x in nums]


def test_parse_prize_table_and_detail_page() -> None:
    prizes = {p["code"]: p for p in parse_prize_table(DETAIL_645)}
    assert prizes["jackpot"]["jackpot_vnd"] == 79_409_173_250 and prizes["jackpot"]["winner_count"] == 2
    assert prizes["giai-nhi"] == {"code": "giai-nhi", "name": "Giải nhì", "winner_count": 2105, "amount_vnd": 300_000, "jackpot_vnd": None}
    d = parse_detail_page(DETAIL_645, "https://www.vietlott.vn/x?id=01569")
    assert d["draw_id"] == 1569 and d["draw_date"] == "2026-09-30" and d["numbers"] == [3, 9, 14, 22, 31, 44]
    assert d["source_pdf_url"] == "https://media.vietlott.vn/media/ketqua/645/01569.pdf" and len(d["source_sha256"]) == 64


def test_product_rows_formats_and_validation() -> None:
    canon = {"draw_id": "00007", "draw_date": "2019-05-06", "result": {"kind": "three_digit_tiers", "tiers": [{"code": "special", "numbers": ["001", "002"]}, {"code": "first", "numbers": ["003"] * 4}, {"code": "second", "numbers": ["004"] * 6}, {"code": "third", "numbers": ["005"] * 8}]}}
    vv = {"id": "00008", "date": "2019-05-08", "result": {"Giải Đặc biệt": ["010", "011"], "Giải Nhất": ["012"] * 4, "Giải Nhì": ["013"] * 6, "Giải ba": ["014"] * 8}}
    h, rej = parse_product_rows(ProductCode.MAX3D, [canon, vv], "t")
    assert h.draw_ids.tolist() == [7, 8] and not rej and h.values[1, :2].tolist() == [10, 11]
    bad_keno = {"id": "1", "date": "2024-01-01", "result": [1] * 20}
    bad_bingo = {"id": "1", "date": "2024-01-01", "result": [1, 2, 7]}
    assert len(parse_product_rows(ProductCode.KENO, [bad_keno], "t")[1]) == 1
    assert len(parse_product_rows(ProductCode.BINGO18, [bad_bingo], "t")[1]) == 1
    assert get_product("Max 3D+") == ProductCode.MAX3D and get_product("3dpro") == ProductCode.MAX3D_PRO


def test_canonical_import_pot_is_value_times_winners() -> None:
    d = parse_detail_page(DETAIL_645, "u")
    row = {"draw_id": "01569", "draw_date": "2026-09-30", "result": {"kind": "number_set", "main_numbers": d["numbers"], "bonus_numbers": []}, "prizes": d["prizes"], "source_pdf_url": d["source_pdf_url"]}
    imp = import_canonical([row, {"draw_id": "x"}], MEGA_645.code)
    assert len(imp.draws) == 1 and len(imp.rejected) == 1 and imp.with_pdf == 1
    rec = imp.prizes[0]
    assert rec.winners == {"jackpot1": 2, "first": 45, "second": 2105, "third": 35877}
    assert rec.jackpot_pots == {"jackpot1": 2 * 79_409_173_250}


def test_bundled_official_seed_is_consistent() -> None:
    rep = json.loads((SEED / "OFFICIAL_DATA_REPORT.json").read_text(encoding="utf-8"))
    for game in ("mega645", "power655", "lotto535"):
        assert rep[game]["prize_records_with_jackpot_pots"] >= 0.95 * rep[game]["draws"]
    for code in SEED_FILES:
        h = load_product_history(code, SEED)
        assert len(h) > 500 and len(np.unique(h.draw_ids)) == len(h)


# ------------------------------------------------------------------ rules / odds
def test_keno_distributions_and_rtp() -> None:
    for b in KENO_TABLE:
        assert sum(keno_hits_distribution(b).values()) == pytest.approx(1.0)
    split = keno_split_distribution()
    assert sum(split.values()) == pytest.approx(1.0) and split[10] == pytest.approx(0.2032, abs=1e-4)
    assert sum(v for x, v in split.items() if x >= 13) == pytest.approx(0.0980, abs=1e-4)
    bacs, sides = keno_odds()
    for b in bacs:  # every bậc of the official table pays back ≈ 50–58 % (bậc 10: 57,7 %)
        assert 0.49 < b.return_to_player < 0.58, b.bac
    assert any("20.000" in k and v > 0.6 for b in bacs for k, v in b.variants.items())  # the outlier variant is flagged, not used
    assert {s.name for s in sides} >= {"Lớn (hoặc Nhỏ)", "Hòa Lớn–Nhỏ", "Chẵn (hoặc Lẻ)"}


def test_keno_rtp_matches_simulation() -> None:
    rng = np.random.default_rng(3)
    draws = np.argsort(rng.random((60_000, 80)), axis=1)[:, :20] + 1
    pick = np.arange(1, 6)  # bậc 5
    hits = np.isin(draws, pick).sum(axis=1)
    exact = keno_hits_distribution(5)
    freq = np.bincount(hits, minlength=6) / len(hits)
    for m in range(6):  # hit-count law (the RTP itself is too noisy: 1-in-1,551 top prize)
        assert freq[m] == pytest.approx(exact[m], abs=4 * np.sqrt(exact[m] * (1 - exact[m]) / len(hits)) + 1e-4)


def test_bingo18_exact_odds() -> None:
    dist = bingo18_sum_distribution()
    assert sum(dist.values()) == pytest.approx(1.0)
    assert sum(dist[s] for s in (10, 11)) == pytest.approx(54 / 216) and sum(dist[s] for s in range(3, 10)) == pytest.approx(81 / 216)
    for r in bingo18_odds():
        assert 0.45 < r.return_to_player < 0.62, r.bet


# ------------------------------------------------------------------ randomness
def _fair(product: ProductCode, n: int, rng: np.random.Generator, units_weights: np.ndarray | None = None) -> ProductHistory:
    if product == ProductCode.KENO:
        vals = np.sort(np.argsort(rng.random((n, 80)), axis=1)[:, :20] + 1, axis=1)
    elif product == ProductCode.BINGO18:
        vals = rng.integers(1, 7, (n, 3))
    else:
        p = np.full(10, 0.1) if units_weights is None else units_weights / units_weights.sum()
        vals = rng.integers(0, 100, (n, 20)) * 10 + rng.choice(10, size=(n, 20), p=p)
    dates = np.datetime64("2024-01-01") + np.arange(n)
    return ProductHistory(product, np.arange(1, n + 1), dates, vals.astype(np.int16), "sim")


def test_fair_histories_pass_and_size_is_controlled() -> None:
    rng = np.random.default_rng(11)
    for product in (ProductCode.KENO, ProductCode.BINGO18, ProductCode.MAX3D):
        rep = product_randomness(_fair(product, 1500, rng))
        assert rep.min_q_value > 0.05, (product, [(t.name, t.p_value) for t in rep.tests])
    rejections = sum(product_randomness(_fair(ProductCode.BINGO18, 1500, rng)).min_q_value < 0.05 for _ in range(60))
    assert rejections <= 8  # family-wise level ≈ 5 % (BH under the global null)


def test_digit_bias_is_detected_and_replicated() -> None:
    rng = np.random.default_rng(5)
    w = np.ones(10)
    w[6] = 1.25  # units digit 6 drawn 25 % too often
    a, b = _fair(ProductCode.MAX3D, 1200, rng, w), _fair(ProductCode.MAX3D_PRO, 800, rng, w)
    units = next(t for t in product_randomness(a).tests if t.name == "Chữ số hàng đơn vị")
    assert units.q_value < 1e-3
    rep = digit_replication(a, b)
    assert (rep.cell.position, rep.cell.digit) == ("đơn vị", 6) and rep.cell.confirm_p_value < 1e-4
    assert rep.per_number_multiplier_max > 1.05
    null = digit_replication(_fair(ProductCode.MAX3D, 1200, rng), _fair(ProductCode.MAX3D_PRO, 800, rng))
    assert null.cell.confirm_p_value > 1e-3  # a selected-but-spurious cell does not replicate
    first, second = split_history(a)
    assert len(first) + len(second) == len(a)


# ------------------------------------------------------------------ Lotto: two-phase accrual
def _simulate_two_phase(days: int = 420, cs: float = 0.078, cf: float = 0.375, refund: float = 5.64e9, p_win: float = 0.012, seed: int = 7):  # type: ignore[no-untyped-def]
    spec = LOTTO_535
    rng = np.random.default_rng(seed)
    d0 = date(2024, 1, 1)
    dates = [d0 + timedelta(days=i // 2) for i in range(2 * days)]
    sold = np.exp(rng.normal(np.log(180_000), 0.1, len(dates)))
    jmin, thr = 6e9, float(spec.rolldown.threshold)
    j, owed, pending = jmin, refund, None
    pots = np.zeros(len(dates))
    won = np.zeros(len(dates), bool)
    rolls: list[int] = []
    for t, d in enumerate(dates):
        rev = sold[t] * spec.ticket_price
        last_of_day = t + 1 == len(dates) or dates[t + 1] != d
        if owed > 0:
            j += cs * rev
            owed -= (cf - cs) * rev
        else:
            j += cf * rev
        pots[t] = j
        if rng.random() < p_win:
            won[t], j, pending = True, jmin, None
            owed += refund
            continue
        if pending == d and last_of_day:
            rolls.append(t)
            j, pending = jmin, None
            owed += refund
            continue
        if pending is None and j > thr:
            pending = d + timedelta(days=1)
    nums = np.argpartition(rng.random((len(dates), 35)), 5, axis=1)[:, :5] + 1
    h = DrawHistory.from_arrays(spec, nums, dates=np.array(dates, dtype="datetime64[D]"), bonus=rng.integers(1, 13, len(dates)))
    w = _simulate_winners(spec, h, sold, rng)
    w[:, 0] = won.astype(int)
    ph = PrizeHistory.align(h, _records(spec, h, w, pots=pots[:, None]))
    return h, ph, sold, rolls, owed


def test_official_rolldowns_and_two_phase_accrual_recovered() -> None:
    h, ph, sold, rolls, _ = _simulate_two_phase()
    off = rolldown_history_official(LOTTO_535, h, ph, sold)
    executed = [e.draw_id for e in off.events if e.executed]
    expected = [int(h.draw_ids[r]) for r in rolls if r + 1 < len(h)]
    assert executed == expected and len(expected) >= 8
    acc = off.accrual
    assert acc.slow_rate == pytest.approx(0.078, rel=0.02) and acc.fast_rate == pytest.approx(0.375, rel=0.02)
    assert acc.diverted_per_restart_median == pytest.approx(5.64e9, rel=0.2)
    assert 6e9 < acc.switch_level_median < 9e9
    for e in off.events:
        if e.executed:
            assert e.realised_rtp_pre_tax > 0 and sum(e.tier_prizes.values()) > 0


def test_rolldown_forecast_phase_and_backtest() -> None:
    h, ph, sold, _, owed = _simulate_two_phase(seed=9)
    off = rolldown_history_official(LOTTO_535, h, ph, sold)
    f = forecast_next_rolldown(LOTTO_535, h, ph, sold, off)
    assert f.phase == ("slow" if owed > 0 else "fast") or abs(owed) < 0.6e9  # switch draw itself is ambiguous
    assert f.draws_to_threshold >= 0 and f.draws_to_fast_phase >= 0
    bt = backtest_rolldown_forecast(LOTTO_535, h, ph, sold, off, start=60, step=5)
    assert bt["cases"] >= 20 and bt["median_abs_error_draws"] <= 3 and bt["spearman_rho"] > 0.8


# ------------------------------------------------------------------ HTTP: proxy vs geo-block
def _client(handler) -> AsyncHttpClient:  # type: ignore[no-untyped-def]
    async def no_sleep(_: float) -> None:
        return None

    return AsyncHttpClient(rate_limit_per_s=1000, burst=100, retry=RetryPolicy(max_retries=3), transport=httpx.MockTransport(handler), sleep=no_sleep)


def test_proxy_refusal_is_not_retried_and_is_reported_as_policy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("403 Forbidden")

    async def run() -> tuple[str, int, str]:
        async with _client(handler) as c:
            with pytest.raises(SourceError) as e1:
                await c.get("https://raw.githubusercontent.com/x/y")
            n = c.stats["requests"]
            with pytest.raises(SourceError) as e2:
                await VietlottProductSource(c, bootstrap_cookie=False).fetch_pages(ProductCode.KENO, 1)
            return str(e1.value), n, str(e2.value)

    msg, n, vl = asyncio.run(run())
    assert PROXY_BLOCK_PREFIX in msg and n == 1 and vl == PROXY_BLOCK_HINT


def test_vietlott_403_is_reported_as_geo_block() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Access denied")

    async def run() -> str:
        async with _client(handler) as c:
            with pytest.raises(SourceError) as e:
                await VietlottProductSource(c).fetch_pages(ProductCode.BINGO18, 1)
            return str(e.value)

    assert asyncio.run(run()) == GEO_BLOCK_HINT


def test_vietlott_product_source_cookie_paging_and_since() -> None:
    rows = [("28/09/2026", 297327 - i, sorted(np.random.default_rng(i).choice(80, 20, replace=False) + 1)) for i in range(6)]
    pages = {1: _keno_html(rows[:3]), 2: _keno_html(rows[3:]), 3: ""}
    seen_cookies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/ajaxpro/":
            return httpx.Response(200, text='<script>document.cookie="RNKEY=abc123;path=/";location.reload();</script>')
        seen_cookies.append(request.headers.get("cookie", ""))
        assert request.headers["X-AjaxPro-Method"] == "ServerSideDrawResult"
        page = json.loads(request.content)["PageIndex"]
        return httpx.Response(200, json={"value": {"HtmlContent": pages.get(page, "")}})

    async def run() -> tuple[list[dict], list[dict]]:
        async with _client(handler) as c:
            src = VietlottProductSource(c, pages_per_batch=2)
            return await src.fetch_pages(ProductCode.KENO, max_pages=10), await src.fetch_pages(ProductCode.KENO, max_pages=10, since_id=297325)

    full, newer = asyncio.run(run())
    assert [r["id"] for r in full] == sorted(297327 - i for i in range(6))
    assert [r["id"] for r in newer] == [297326, 297327]
    assert all("RNKEY=abc123" in ck for ck in seen_cookies)


def test_fetch_canonical_from_detail_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["id"] == "01569"
        return httpx.Response(200, text=DETAIL_645)

    async def run() -> list[dict]:
        async with _client(handler) as c:
            return await VietlottOfficialSource(c).fetch_canonical(MEGA_645, [1569])

    rows = asyncio.run(run())
    imp = import_canonical(rows, MEGA_645.code)
    assert rows[0]["schema_version"] == "1.0" and imp.prizes[0].jackpot_pots == {"jackpot1": 2 * 79_409_173_250}


# ------------------------------------------------------------------ store + sync
def test_product_store_merges_seed_and_sync(tmp_path: Path) -> None:
    def row(i: int) -> dict:
        return {"id": str(i), "date": "2026-09-28", "result": [1 + (i + k) % 6 for k in range(3)]}

    mirror = [row(i) for i in range(1, 4)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/bingo18.jsonl")
        return httpx.Response(200, text="\n".join(json.dumps(r) for r in mirror))

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / SEED_FILES[ProductCode.BINGO18]).write_bytes(b"")  # empty seed file (gz handled by name)
    store = ProductStore(tmp_path / "data", None)

    async def run() -> list[dict]:
        async with _client(handler) as c:
            pipe = ProductSyncPipeline(c, store, vietlott_base_url="https://www.vietlott.vn", mirror_base_url="https://m", canonical_base_url="https://c")
            first = (await pipe.run(ProductCode.BINGO18, "mirror")).to_dict()
            mirror.append(row(4))
            second = (await pipe.run(ProductCode.BINGO18, "mirror")).to_dict()
            bad = (await pipe.run(ProductCode.BINGO18, "canonical")).to_dict()
            return [first, second, bad]

    first, second, bad = asyncio.run(run())
    assert (first["inserted"], first["total_after"]) == (3, 3)
    assert (second["inserted"], second["total_after"], second["last_id"]) == (1, 4, 4)
    assert bad["error"] and "no canonical records" in bad["error"]
    with_seed = ProductStore(tmp_path / "data", SEED)
    assert len(with_seed.load(ProductCode.BINGO18)) >= len(load_product_history(ProductCode.BINGO18, SEED))


def test_sync_canonical_prizes_into_repository() -> None:
    d = parse_detail_page(DETAIL_645, "u")
    recs = [
        {"draw_id": f"{i:05d}", "draw_date": f"2026-09-{i - 1540:02d}", "result": {"kind": "number_set", "main_numbers": d["numbers"], "bonus_numbers": []}, "prizes": d["prizes"]}
        for i in (1568, 1569)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/mega645.jsonl")
        return httpx.Response(200, text="\n".join(json.dumps(r) for r in recs))

    repo = InMemoryRepository()

    async def run():  # type: ignore[no-untyped-def]
        async with _client(handler) as c:
            return await sync_canonical_prizes(c, repo, MEGA_645, source="canonical", canonical_base_url="https://c", vietlott_base_url="https://v")

    rep = asyncio.run(run())
    assert (rep.records, rep.draws_inserted, rep.prize_records, rep.with_jackpot_pots, rep.error) == (2, 2, 2, 2, None)
    assert len(repo.load_prizes(MEGA_645.code)) == 2


# ------------------------------------------------------------------ API + CLI
@pytest.fixture(scope="module")
def api():  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from vietlott_engine.api.main import create_app
    from vietlott_engine.core.config import Settings

    s = Settings(storage_backend="memory", seed_file_dir=SEED, product_seed_dir=SEED, data_dir=Path("/nonexistent-vqe"))
    with TestClient(create_app(s)) as client:
        yield client


def test_api_catalogue_covers_all_products(api) -> None:  # type: ignore[no-untyped-def]
    rows = {r["code"]: r for r in api.get("/products").json()}
    assert set(rows) == {c.value for c in ProductCode}
    assert all(r["stored_draws"] > 500 for r in rows.values())
    assert rows["mega645"]["endpoints"] == "/games/mega645" and rows["keno"]["endpoints"] == "/products/keno"


def test_api_product_endpoints(api) -> None:  # type: ignore[no-untyped-def]
    k = api.get("/products/keno/odds").json()
    assert len(k["bac"]) == 10 and k["side_bets"]
    assert len(api.get("/products/bingo18/odds").json()) == 23
    d = api.get("/products/max3dpro/draws", params={"limit": 3}).json()
    assert len(d["draws"]) == 3 and all(len(x) == 3 for x in d["draws"][0]["result"])
    r = api.get("/products/bingo18/randomness").json()
    assert r["draws"] > 50_000 and r["tests"]
    c = api.get("/products/max3d/digit-check").json()
    assert c["cell"]["position"] in ("trăm", "chục", "đơn vị")
    assert api.get("/products/mega645/draws").status_code == 404
    ro = api.get("/games/lotto535/rolldown/official")
    if ro.status_code == 200:  # needs data/calibration (bundled)
        body = ro.json()
        assert body["official"]["executed"] >= 10 and body["forecast"]["draws_to_threshold"] >= 0
    assert api.get("/products/nope/draws").status_code == 404


def test_cli_products(capsys: pytest.CaptureFixture[str]) -> None:
    from vietlott_engine.cli import main

    assert main(["products", "list"]) == 0
    assert "Max 3D Pro" in capsys.readouterr().out
    assert main(["products", "odds", "--product", "bingo18"]) == 0
    assert "Bộ ba bất kỳ" in capsys.readouterr().out
    assert main(["products", "analyze", "--product", "max3dpro"]) == 0
    assert "Chữ số hàng đơn vị" in capsys.readouterr().out


def test_break_even_finds_first_crossing_with_convex_sales() -> None:
    """Convex sales extrapolated far out bring EV back below the price; the first crossing
    must still be found (v3.2 fix: Mega showed RTP > 1 at 100 tỷ but 'no break-even')."""
    from vietlott_engine.game_theory.ev import EVCalculator

    class ConvexSales:
        def predict(self, j: float) -> float:
            x = np.log(j / 30e9)
            return float(1e6 * np.exp(0.6 * x + 0.25 * x**2))

    calc = EVCalculator(MEGA_645, sales=ConvexSales())
    t = [3, 9, 14, 22, 31, 38]
    be = calc.break_even_jackpot(t, tickets_sold=None)
    assert be is not None
    sold = lambda j: int(ConvexSales().predict(j))  # noqa: E731
    below = calc.evaluate_ev_only(t, be * 0.95, None, sold(be * 0.95))
    above = calc.evaluate_ev_only(t, be * 1.05, None, sold(be * 1.05))
    assert below < MEGA_645.ticket_price < above
    assert calc.evaluate_ev_only(t, 1e14, None, sold(1e14)) < MEGA_645.ticket_price  # the far end is below again


def test_no_announcement_the_day_after_an_executed_rolldown() -> None:
    """Day d−1 crossed 12 tỷ at 13:00 and was itself a rolldown day: the 21:00 draw
    distributed the pot, so a jackpot win on day d is not a pre-empted rolldown."""
    spec = LOTTO_535
    d0 = date(2026, 9, 7)
    dates = [d0, d0, d0 + timedelta(1), d0 + timedelta(1), d0 + timedelta(2), d0 + timedelta(2)]
    pots = np.array([11.5e9, 12.3e9, 12.5e9, 12.8e9, 6.1e9, 6.2e9])
    winners = np.zeros((6, len(spec.tiers)), dtype=np.int64)
    winners[:, 1:] = 50
    winners[4, 0] = 1  # jackpot won at 13:00 the day after the rolldown
    nums = np.tile(np.arange(1, 6), (6, 1))
    h = DrawHistory.from_arrays(spec, nums, dates=np.array(dates, dtype="datetime64[D]"), bonus=np.ones(6, dtype=int))
    ph = PrizeHistory.align(h, _records(spec, h, winners, pots=pots[:, None]))
    off = rolldown_history_official(spec, h, ph, np.full(6, 2e5))
    assert [(e.draw_id, e.executed) for e in off.events] == [(int(h.draw_ids[3]), True)]
    assert off.pre_empted == 0
