"""v3.3: fallback when vietlott.vn is unreachable — community archive (NhanAZ-Data/vietlott-research),
Vietlott Quant Engine 1.3.0 snapshot import, ordered fallback, exclusions, merge layers,
Max 4D, official Keno table."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest

from vietlott_engine.analytics.products import digit_cell_test, digit_tests, product_randomness
from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import LOTTO_535, MEGA_645
from vietlott_engine.core.products import ProductCode, ProductHistory, load_product_history, parse_product_rows
from vietlott_engine.crawler.official_data import import_canonical
from vietlott_engine.crawler.product_store import ProductStore, ProductSyncPipeline
from vietlott_engine.crawler.sources.base import DrawSource
from vietlott_engine.crawler.sources.fallback import ArchiveDrawSource, FallbackDrawSource
from vietlott_engine.crawler.sources.nhanaz import NhanAZArchive, keno_rules, prize_cells
from vietlott_engine.game_theory.fastgames import KENO_SIDE_BETS, KENO_TABLE
from tests.test_all_products import _client

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"
DRAW_COLS = ["product", "draw_id", "draw_date", "draw_status", "result_json", "attributes_json", "official_pdf_urls_json", "source_url", "prize_status", "validation_status", "validation_warnings_json", "fetched_at"]
PRIZE_COLS = ["product", "draw_id", "game_variant", "prize_tier", "winning_rule", "winner_count", "prize_value_vnd", "details_json", "source_url", "fetched_at"]


def _csv(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue()


def _draw(product: str, did: str, d: str, result: dict, status: str = "confirmed", source: str = "official_vietlott", **attrs) -> dict:
    return {"product": product, "draw_id": did, "draw_date": d, "draw_status": status, "result_json": json.dumps(result), "attributes_json": json.dumps({"data_source": source} | attrs)}


KENO_A = list(range(1, 21))
KENO_B = list(range(21, 41))


def make_archive(root: Path) -> Path:
    ds = root / "datasets"
    (ds / "draws" / "keno").mkdir(parents=True)
    (ds / "draws" / "keno" / "2026-03.csv").write_text(_csv([_draw("keno", "0275900", "2026-03-31", {"numbers": KENO_A}, source="xosominhngoc_net_vn")], DRAW_COLS), encoding="utf-8")
    (ds / "draws" / "keno" / "2026-04.csv").write_text(
        _csv([_draw("keno", "0275985", "2026-04-02", {"numbers": KENO_B}), _draw("keno", "0275986", "2026-04-02", {"numbers": KENO_A}, status="not_confirmed")], DRAW_COLS), encoding="utf-8"
    )
    (ds / "draws" / "mega645").mkdir(parents=True)
    (ds / "draws" / "mega645" / "all.csv").write_text(
        _csv([_draw("mega645", "01569", "2026-09-30", {"numbers": [3, 9, 14, 22, 31, 44], "special_numbers": []}, jackpots_vnd={"Jackpot": 79_409_173_250})], DRAW_COLS), encoding="utf-8"
    )
    (ds / "prizes" / "mega645").mkdir(parents=True)
    tiers = [("Jackpot", 2, 79_409_173_250), ("Giải Nhất", 45, 10_000_000), ("Giải Nhì", 2105, 300_000), ("Giải Ba", 35877, 30_000)]
    (ds / "prizes" / "mega645" / "all.csv").write_text(
        _csv([{"product": "mega645", "draw_id": "01569", "prize_tier": t, "winner_count": str(w), "prize_value_vnd": str(v)} for t, w, v in tiers], PRIZE_COLS), encoding="utf-8"
    )
    (ds / "draws" / "lotto535").mkdir(parents=True)
    (ds / "draws" / "lotto535" / "all.csv").write_text(_csv([_draw("lotto535", "00001", "2025-06-29", {"numbers": [7, 9, 10, 16, 19], "special_numbers": [9]})], DRAW_COLS), encoding="utf-8")
    (ds / "prizes" / "lotto535").mkdir(parents=True)
    shifted = [("Giải thưởng", "Số lượng giải", "Giá trị giải (đồng)"), ("Giải Độc Đắc", "0", "6.112.312.500"), ("Giải Nhất", "0", "10.000.000"), ("Giải Nhì", "13", "5.000.000"), ("Giải Ba", "118", "500.000"), ("Giải Tư", "285", "100.000"), ("Giải Năm", "2.921", "30.000"), ("Giải Khuyến Khích", "19.563", "10.000")]
    (ds / "prizes" / "lotto535" / "all.csv").write_text(
        _csv([{"product": "lotto535", "draw_id": "00001", "prize_tier": t, "details_json": json.dumps({"columns": {"6.112.312.500 VND": "O", "Giải Độc Đắc": t, "column_3": w, "column_4": v}})} for t, w, v in shifted], PRIZE_COLS), encoding="utf-8"
    )
    (ds / "draws" / "max4d").mkdir(parents=True)
    (ds / "draws" / "max4d" / "all.csv").write_text(
        _csv([_draw("max4d", "00001", "2016-11-19", {"tiers": {"first": ["7589"], "second": ["3448", "0273"], "third": ["0210", "3184", "6691"], "consolation_1": ["X589"], "consolation_2": ["XX89"]}})], DRAW_COLS), encoding="utf-8"
    )
    (ds / "exclusions.csv").write_text("product,draw_id,draw_status,effective_date,reason,source_url\nkeno,0275986,not_confirmed,2026-04-02,Không được xác nhận,https://vietlott.vn/x\n", encoding="utf-8")
    rules = [("Trùng 10 trong 20 số", 2_000_000_000, 5), ("Trùng 07 trong 20 số", 710_000, 5), ("Trùng 04 trong 20 số", 150_000, 10), ("Lớn từ 13 số trở lên từ 41-80", 26_000, 16)]
    (ds / "prize_rules.csv").write_text(
        _csv([{"product": "keno", "prize_tier": t, "prize_value_vnd": str(v), "details_json": json.dumps({"table_index": ti}), "source_url": "https://vietlott.vn/k"} for t, v, ti in rules], ["product", "game_variant", "prize_tier", "winning_rule", "prize_value_vnd", "details_json", "source_url"]),
        encoding="utf-8",
    )
    return root


# ------------------------------------------------------------------ archive parsing
def test_archive_local_folder_all_formats(tmp_path: Path) -> None:
    a = NhanAZArchive(local_dir=make_archive(tmp_path))

    async def run():  # type: ignore[no-untyped-def]
        return await a.draws(ProductCode.KENO), await a.draws(ProductCode.MEGA_645), await a.draws(ProductCode.LOTTO_535), await a.draws(ProductCode.MAX4D), await a.exclusions(), await a.prize_rules()

    keno, mega, lotto, max4d, excl, rules = asyncio.run(run())
    assert [r["id"] for r in keno.rows] == [275900, 275985] and keno.not_confirmed[0]["draw_id"] == 275986
    assert keno.sources == {"xosominhngoc_net_vn": 1, "official_vietlott": 1}
    imp = import_canonical(mega.rows, MEGA_645.code)
    assert imp.prizes[0].jackpot_pots == {"jackpot1": 2 * 79_409_173_250} and imp.prizes[0].winners["third"] == 35877
    lp = import_canonical(lotto.rows, LOTTO_535.code).prizes[0]  # shifted header columns
    assert lp.winners == {"jackpot1": 0, "first": 0, "second": 13, "third": 118, "fourth": 285, "fifth": 2921, "consolation": 19563}
    assert lp.jackpot_pots == {"jackpot1": 6_112_312_500}
    h, rej = parse_product_rows(ProductCode.MAX4D, max4d.rows, "t")
    assert not rej and h.values[0].tolist() == [7589, 3448, 273, 210, 3184, 6691]
    assert excl == [{"product": "keno", "draw_id": 275986, "draw_status": "not_confirmed", "effective_date": "2026-04-02", "reason": "Không được xác nhận", "source_url": "https://vietlott.vn/x"}]
    kr = keno_rules(rules)
    assert kr["bac"][10] == {7: 710_000, 10: 2_000_000_000} and kr["bac"][5] == {4: 150_000} and kr["side_bets"]["Lớn từ 13 số trở lên từ 41-80"] == 26_000


def test_prize_cells_layouts() -> None:
    assert prize_cells({"prize_tier": "Giải Ba", "winner_count": "1387", "prize_value_vnd": "4130000"}) == ("Giải Ba", 1387, 4_130_000)
    assert prize_cells({"prize_tier": "Giải Ba", "details_json": json.dumps({"columns": ["Giải Ba", "4 số", "1.387", "4.130.000"]})}) == ("Giải Ba", 1387, 4_130_000)
    assert prize_cells({"prize_tier": "Giải thưởng", "details_json": "{}"}) is None
    assert prize_cells({"prize_tier": "Giải Nhì", "details_json": json.dumps({"columns": {"column_3": "Số lượng giải", "column_4": "x"}})}) is None


def test_archive_remote_months_and_missing_month(tmp_path: Path) -> None:
    base = make_archive(tmp_path) / "datasets"
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        rel = request.url.path.split("/datasets/")[1]
        asked.append(rel)
        f = base / rel
        return httpx.Response(200, text=f.read_text(encoding="utf-8")) if f.exists() else httpx.Response(404)

    async def run():  # type: ignore[no-untyped-def]
        async with _client(handler) as c:
            return await NhanAZArchive(c, "https://raw.example/datasets").draws(ProductCode.KENO, since=date(2026, 3, 15))

    got = asyncio.run(run())
    months = [a for a in asked if a.startswith("draws/keno/")]
    assert months[0] == "draws/keno/2026-03.csv" and "draws/keno/2026-04.csv" in months
    assert [r["id"] for r in got.rows] == [275900, 275985]  # later months 404 → skipped


# ------------------------------------------------------------------ fallback
class _Failing(DrawSource):
    name = "vietlott"

    async def fetch(self, spec, since_id=None):  # type: ignore[no-untyped-def]
        raise SourceError("this network's HTTPS proxy does not allow www.vietlott.vn")


def test_fallback_uses_next_source_and_records_attempts(tmp_path: Path) -> None:
    archive = ArchiveDrawSource(archive=NhanAZArchive(local_dir=make_archive(tmp_path)))
    fb = FallbackDrawSource([_Failing(), archive])
    res = asyncio.run(fb.fetch(MEGA_645))
    assert [d.draw_id for d in res.draws] == [1569] and res.draws[0].source == "nhanaz:official_vietlott"
    assert [(a["source"], a["ok"]) for a in fb.log.to_list()] == [("vietlott", False), ("nhanaz", True)] and fb.log.used == "nhanaz"
    assert archive.last_records and archive.last_records[0]["prizes"]
    with pytest.raises(SourceError, match="all sources failed"):
        asyncio.run(FallbackDrawSource([_Failing()]).fetch(MEGA_645))


def test_product_sync_auto_falls_back_to_archive_and_records_exclusions(tmp_path: Path) -> None:
    base = make_archive(tmp_path / "arch") / "datasets"

    def handler(request: httpx.Request) -> httpx.Response:
        if "vietlott.vn" in request.url.host:
            raise httpx.ProxyError("403 Forbidden")
        rel = request.url.path.split("/datasets/")[1]
        f = base / rel
        return httpx.Response(200, text=f.read_text(encoding="utf-8")) if f.exists() else httpx.Response(404)

    store = ProductStore(tmp_path / "data", None)

    async def run():  # type: ignore[no-untyped-def]
        async with _client(handler) as c:
            pipe = ProductSyncPipeline(c, store, vietlott_base_url="https://www.vietlott.vn", mirror_base_url="https://m", canonical_base_url="https://c", nhanaz_base_url="https://raw.example/datasets", fallback_order=["vietlott", "nhanaz"])
            return await pipe.run(ProductCode.KENO, "auto")

    rep = asyncio.run(run())
    assert rep.source_used == "nhanaz" and [a["source"] for a in rep.attempts] == ["vietlott", "nhanaz"] and not rep.attempts[0]["ok"]
    assert (rep.inserted, rep.total_after, rep.exclusions_added) == (2, 2, 1)
    store.upsert(ProductCode.KENO, [{"id": 275986, "date": "2026-04-02", "result": KENO_A}], "test")  # a not-confirmed draw reaches the store
    assert 275986 not in store.load(ProductCode.KENO).draw_ids and 275986 in store.load(ProductCode.KENO, include_unconfirmed=True).draw_ids


def test_bundled_exclusions_are_applied() -> None:
    h = load_product_history(ProductCode.KENO, SEED)
    full = load_product_history(ProductCode.KENO, SEED, include_unconfirmed=True)
    excluded = {e["draw_id"] for e in json.loads((SEED / "exclusions.json").read_text(encoding="utf-8")) if e["product"] == "keno"}
    assert len(excluded) == 31 and not set(h.draw_ids.tolist()) & excluded and set(full.draw_ids.tolist()) & excluded


# ------------------------------------------------------------------ 1.3.0 snapshot
def _duckdb_cli() -> str | None:
    return os.environ.get("VQE_DUCKDB_CLI") or shutil.which("duckdb")


@pytest.mark.skipif(_duckdb_cli() is None, reason="needs the DuckDB CLI to write a test Parquet file")
def test_v130_snapshot_import(tmp_path: Path) -> None:
    from vietlott_engine.crawler.sources.vqe130 import import_v130, v130_exclusions

    d = tmp_path / "data" / "products"
    d.mkdir(parents=True)
    rows = [
        {"draw_number": 297759, "draw_date": "2026-10-01", "status": "reported", "payload": json.dumps({"draw_id": "297759", "result": {"kind": "keno", "numbers": KENO_A}, "attributes": {"data_source": "xosominhngoc_net_vn"}})},
        {"draw_number": 275986, "draw_date": "2026-04-02", "status": "not_confirmed", "payload": json.dumps({"draw_id": "275986", "result": {"kind": "keno", "numbers": KENO_B}, "attributes": {}})},
    ]
    src = tmp_path / "keno.json"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    subprocess.run([_duckdb_cli(), "-c", f"COPY (SELECT * FROM read_json_auto('{src}')) TO '{d / 'keno.parquet'}' (FORMAT parquet)"], check=True, capture_output=True)
    (d / "coverage.json").write_text("{}", encoding="utf-8")
    (d / "exclusions.json").write_text(json.dumps({"keno": [{"product": "keno", "draw_id": "0275986", "draw_status": "not_confirmed", "effective_date": "2026-04-02"}]}), encoding="utf-8")
    got = import_v130(tmp_path, ProductCode.KENO)
    assert [r["id"] for r in got.rows] == [297759] and got.not_confirmed[0]["draw_id"] == 275986 and got.sources["xosominhngoc_net_vn"] == 1
    assert v130_exclusions(tmp_path)[0]["draw_id"] == 275986


def test_v130_missing_snapshot_is_reported(tmp_path: Path) -> None:
    from vietlott_engine.crawler.sources.vqe130 import products_dir

    with pytest.raises(SourceError, match="1.3.0"):
        products_dir(tmp_path)


# ------------------------------------------------------------------ merge layers
def _load_builder():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("bod", ROOT / "scripts" / "build_official_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_merge_layers_priority_and_conflicts() -> None:
    mod = _load_builder()
    hi = [{"id": 1, "date": "2026-01-01", "result": [1, 2, 3]}, {"id": 2, "date": "2026-01-01", "result": [4, 5, 6]}]
    lo = [{"id": 2, "date": "2026-01-01", "result": [6, 6, 6], "data_source": "xoso"}, {"id": 3, "date": "2026-01-02", "result": [1, 1, 2], "data_source": "xoso"}]
    out, stats = mod.merge_layers(ProductCode.BINGO18, [("vietlott.vn/list", hi), ("nhanaz", lo)])
    assert [r["result"] for r in out] == [[1, 2, 3], [4, 5, 6], [1, 1, 2]]  # higher layer wins on id 2
    L = stats["layers"][1]
    assert (L["overlap_with_higher_layers"], L["identical"], L["conflicts"], L["added"]) == (1, 0, 1, 1)
    assert out[2]["src"] == "nhanaz:xoso" and stats["origin_mix"] == {"vietlott.vn/list": 2, "nhanaz:xoso": 1}


# ------------------------------------------------------------------ Max 4D + digits
def test_max4d_digit_tests_size_and_power() -> None:
    rng = np.random.default_rng(2)
    fair = rng.integers(0, 10000, (2000, 6))
    h = ProductHistory(ProductCode.MAX4D, np.arange(1, 2001), np.datetime64("2017-01-01") + np.arange(2000), fair.astype(np.int16), "sim")
    rep = product_randomness(h)
    assert rep.min_q_value > 0.05 and any(t.name == "Chữ số hàng nghìn" for t in rep.tests)
    w = np.ones(10)
    w[6] = 1.3
    units = rng.choice(10, size=(2000, 6), p=w / w.sum())
    biased = ProductHistory(ProductCode.MAX4D, h.draw_ids, h.dates, (rng.integers(0, 1000, (2000, 6)) * 10 + units).astype(np.int16), "sim")
    assert next(t for t in digit_tests(biased, 4) if t.name == "Chữ số hàng đơn vị").p_value < 1e-4
    cell = digit_cell_test(biased, 4, 0, 6, "greater")
    assert cell.position == "đơn vị" and cell.p_value < 1e-4 and cell.share > 0.11
    assert digit_cell_test(h, 4, 0, 6, "greater").p_value > 0.01


def test_bundled_max4d_history() -> None:
    h = load_product_history(ProductCode.MAX4D, SEED)
    assert len(h) == 722 and h.values.shape == (722, 6) and h.values.max() <= 9999
    assert str(h.dates.min()) == "2016-11-19" and str(h.dates.max()) == "2021-08-31"


# ------------------------------------------------------------------ official Keno table
def test_keno_table_matches_official_detail_page() -> None:
    rules = json.loads((SEED / "keno_rules_official.json").read_text(encoding="utf-8"))
    for bac, cells in rules["bac"].items():
        for hits, value in cells.items():
            if (int(bac), int(hits)) == (5, 4):
                assert value == 0 and KENO_TABLE[5][4] == 150_000  # page shows "0 đ"; advertised 150.000 kept
                continue
            assert KENO_TABLE[int(bac)].get(int(hits)) == value, (bac, hits)
    side = {sb.name: sb for sb in KENO_SIDE_BETS}
    assert side["Lớn (hoặc Nhỏ)"].tiers == [(11, 12, 10_000), (13, 20, 26_000)] and rules["side_bets"]["Lớn từ 13 số trở lên từ 41-80"] == 26_000
    assert side["Chẵn (hoặc Lẻ)"].tiers == [(13, 14, 40_000), (15, 20, 200_000)] and rules["side_bets"]["15 số trở lên là số chẵn"] == 200_000
    assert all(sb.verified for sb in KENO_SIDE_BETS)


# ------------------------------------------------------------------ CLI / API
def test_cli_sync_from_local_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from vietlott_engine.cli import main
    from vietlott_engine.core.config import get_settings

    arch = make_archive(tmp_path / "arch")
    monkeypatch.setenv("VQE_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    try:
        assert main(["products", "sync", "--product", "keno", "--source", "nhanaz", "--path", str(arch)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out[0]["source_used"] == "nhanaz" and out[0]["inserted"] >= 0 and out[0]["exclusions_added"] >= 0
        assert main(["products", "odds", "--product", "max4d"]) == 0
        assert "đã ngừng" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_api_max4d(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from vietlott_engine.api.main import create_app
    from vietlott_engine.core.config import Settings

    s = Settings(storage_backend="memory", seed_file_dir=SEED, product_seed_dir=SEED, data_dir=tmp_path)
    with TestClient(create_app(s)) as c:
        rows = {r["code"]: r for r in c.get("/products").json()}
        assert rows["max4d"]["stored_draws"] == 722
        d = c.get("/products/max4d/draws", params={"limit": 2}).json()
        assert all(len(x) == 4 for x in d["draws"][0]["result"])
        assert c.get("/products/max4d/randomness").json()["draws"] == 722
