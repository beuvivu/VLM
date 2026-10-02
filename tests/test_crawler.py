from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import MEGA_645, POWER_655
from vietlott_engine.crawler.http import AsyncHttpClient, AsyncRateLimiter, RetryPolicy
from vietlott_engine.crawler.pipeline import SyncPipeline, check_integrity
from vietlott_engine.crawler.sources.mirror import GithubMirrorSource, JsonlFileSource, parse_jsonl
from vietlott_engine.crawler.sources.vietlott_official import VietlottOfficialSource, parse_results_html
from vietlott_engine.crawler.storage import InMemoryRepository

FIXTURES = Path(__file__).parent / "fixtures"


def _client(handler, sleeps: list[float] | None = None, retries: int = 3) -> AsyncHttpClient:
    async def fake_sleep(s: float) -> None:
        if sleeps is not None:
            sleeps.append(s)

    return AsyncHttpClient(
        rate_limit_per_s=1000,
        burst=1000,
        retry=RetryPolicy(max_retries=retries, base_delay=0.01, max_delay=0.05),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )


def test_retry_then_success_and_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.03"})
        if calls["n"] == 2:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, text="ok")

    sleeps: list[float] = []

    async def run() -> str:
        async with _client(handler, sleeps) as c:
            return (await c.get("https://example.test/x")).text

    assert asyncio.run(run()) == "ok"
    assert calls["n"] == 3
    assert sleeps[0] == pytest.approx(0.03)  # Retry-After honoured
    assert 0 <= sleeps[1] <= 0.02  # full-jitter backoff for attempt 1


def test_non_retryable_and_exhausted() -> None:
    async def run(status: int) -> None:
        async with _client(lambda r: httpx.Response(status), retries=2) as c:
            await c.get("https://example.test/x")

    with pytest.raises(SourceError, match="non-retryable"):
        asyncio.run(run(404))
    with pytest.raises(SourceError, match="after 2 retries"):
        asyncio.run(run(503))


def test_rate_limiter_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}
    real_sleep = asyncio.sleep

    async def advance(dt: float) -> None:  # virtual time: sleeping advances the clock
        clock["t"] += dt
        await real_sleep(0)

    monkeypatch.setattr("vietlott_engine.crawler.http.asyncio.sleep", advance)

    async def run() -> list[float]:
        limiter = AsyncRateLimiter(rate=10, burst=1, clock=lambda: clock["t"])
        stamps = []
        for _ in range(5):
            await limiter.acquire()
            stamps.append(clock["t"])
        return stamps

    stamps = asyncio.run(run())
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(g == pytest.approx(0.1, abs=1e-9) for g in gaps)


def test_parse_jsonl_mirror_and_reject() -> None:
    text = "\n".join(
        [
            json.dumps({"date": "2017-08-01", "id": "00001", "result": [5, 10, 14, 23, 24, 38, 35]}),
            json.dumps({"date": "2017-08-03", "id": "00002", "result": [1, 2, 3, 4, 5, 6]}),  # missing bonus
            "not json",
        ]
    )
    res = parse_jsonl(text, POWER_655, None, "test")
    assert [d.draw_id for d in res.draws] == [1]
    assert res.draws[0].bonus == 35
    assert len(res.rejected) == 2


def test_official_html_parser() -> None:
    html = (FIXTURES / "official_655.html").read_text(encoding="utf-8")
    rows = parse_results_html(html)
    assert len(rows) == 2
    draw = rows[0].to_draw(POWER_655)
    assert draw.draw_id == 1403 and draw.numbers == (14, 18, 21, 38, 48, 52) and draw.bonus == 49
    assert str(draw.draw_date) == "2026-09-26"


def test_official_source_pagination_stops_at_since_id() -> None:
    html = (FIXTURES / "official_655.html").read_text(encoding="utf-8")
    pages = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        pages["n"] += 1
        body = json.loads(request.content)
        content = html if body["PageIndex"] == 0 else ""
        assert request.headers["X-AjaxPro-Method"] == "ServerSideDrawResult"
        return httpx.Response(200, json={"value": {"HtmlContent": content}})

    async def run():
        async with _client(handler) as c:
            return await VietlottOfficialSource(c, "https://vietlott.test", pages_per_batch=2).fetch(POWER_655, since_id=1401)

    res = asyncio.run(run())
    assert [d.draw_id for d in res.draws] == [1402, 1403]


def test_mirror_source_and_incremental_pipeline() -> None:
    lines = [json.dumps({"date": f"2020-01-{i:02d}", "id": i, "result": [i, i + 1, i + 2, i + 3, i + 4, i + 5]}) for i in range(1, 11)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/power645.jsonl")
        return httpx.Response(200, text="\n".join(lines))

    repo = InMemoryRepository()

    async def run():
        async with _client(handler) as c:
            src = GithubMirrorSource(c, "https://mirror.test/data")
            first = await SyncPipeline(src, repo).run(MEGA_645)
            second = await SyncPipeline(src, repo).run(MEGA_645)
            return first, second

    first, second = asyncio.run(run())
    assert first.inserted == 10 and second.fetched == 0 and repo.count("mega645") == 10
    assert first.integrity.ok


def test_integrity_detects_gaps(tmp_path: Path) -> None:
    lines = [json.dumps({"date": f"2020-01-{i:02d}", "id": i, "result": [1, 2, 3, 4, 5, 6]}) for i in (3, 4, 7)]
    (tmp_path / "power645.jsonl").write_text("\n".join(lines))
    res = asyncio.run(JsonlFileSource(tmp_path).fetch(MEGA_645))
    rep = check_integrity(res.draws, MEGA_645)
    assert rep.id_gaps == [(1, 2), (5, 6)]


def test_duckdb_repository_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    from vietlott_engine.crawler.storage import DuckDBRepository

    lines = [json.dumps({"date": f"2020-01-{i:02d}", "id": i, "result": [i, i + 1, i + 2, i + 3, i + 4, i + 5, 50]}) for i in range(1, 6)]
    (tmp_path / "power655.jsonl").write_text("\n".join(lines))
    repo = DuckDBRepository(tmp_path / "db.duckdb", tmp_path / "parquet")
    report = asyncio.run(SyncPipeline(JsonlFileSource(tmp_path), repo).run(POWER_655))
    assert report.inserted == 5 and repo.count("power655") == 5 and repo.max_draw_id("power655") == 5
    assert repo.upsert(asyncio.run(JsonlFileSource(tmp_path).fetch(POWER_655)).draws) == 5  # idempotent upsert
    assert repo.count("power655") == 5
    h = repo.load_history("power655")
    assert h.bonus.tolist() == [50] * 5 and h.numbers[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert (tmp_path / "parquet" / "power655.parquet").exists()
