from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vietlott_engine.api.main import create_app
from vietlott_engine.core.config import Settings
from vietlott_engine.core.games import LOTTO_535, MEGA_645, POWER_655
from vietlott_engine.core.models import Draw
from vietlott_engine.crawler.storage import InMemoryRepository
from tests.conftest import make_history

CAL_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration"


def _repo() -> InMemoryRepository:
    repo = InMemoryRepository()
    for spec, seed in ((MEGA_645, 21), (POWER_655, 22), (LOTTO_535, 23)):
        h = make_history(spec, 300, seed=seed)
        repo.upsert(
            Draw(game=spec.code, draw_id=i + 1, draw_date=str(h.dates[i]), numbers=tuple(int(x) for x in h.numbers[i]), bonus=int(h.bonus[i]) or None)
            for i in range(len(h))
        )
    return repo


@pytest.fixture(scope="module")
def bare(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """No calibration artefacts: prior crowd model, typical sales fallback."""
    settings = Settings(storage_backend="memory", calibration_dir=tmp_path_factory.mktemp("cal"))
    with TestClient(create_app(settings, repository=_repo())) as c:
        yield c


@pytest.fixture(scope="module")
def calibrated() -> TestClient:
    if not (CAL_DIR / "market_lotto535.json").exists():
        pytest.skip("data/calibration not built")
    with TestClient(create_app(Settings(storage_backend="memory", calibration_dir=CAL_DIR), repository=_repo())) as c:
        yield c


@pytest.mark.parametrize("game,p_any", [("mega645", 0.0238), ("power655", 0.0133), ("lotto535", 0.0960)])
def test_odds(bare: TestClient, game: str, p_any: float) -> None:
    r = bare.get(f"/games/{game}/odds").json()
    assert r["p_any_prize"] == pytest.approx(p_any, abs=5e-4)
    assert sum(t["probability"] for t in r["tiers"]) == pytest.approx(r["p_any_prize"], rel=1e-12)
    assert r["one_in_any_prize"] == pytest.approx(1 / r["p_any_prize"])


def test_lotto_ev_special_and_validation(bare: TestClient) -> None:
    ok = bare.post("/games/lotto535/ev", json={"ticket": [3, 14, 22, 29, 33], "special": 7, "jackpot1": 9e9, "tickets_sold": 200_000})
    assert ok.status_code == 200, ok.text
    assert ok.json()["special"] == 7 and 0 < ok.json()["return_to_player"] < 1
    assert bare.post("/games/lotto535/ev", json={"ticket": [3, 14, 22, 29, 33]}).status_code == 422
    assert bare.post("/games/mega645/ev", json={"ticket": [3, 14, 22, 29, 33]}).status_code == 422
    dec = bare.post("/games/lotto535/ev/decision", json={"ticket": [3, 14, 22, 29, 33], "special": 2, "tickets_sold": 200_000})
    assert dec.status_code == 200, dec.text


def test_bao_coverage_rolldown(bare: TestClient) -> None:
    r = bare.post("/games/mega645/bao", json={"numbers": [5, 12, 19, 26, 33, 40, 44]})
    assert r.status_code == 200 and r.json()["analysis"]["plays"] == 7 and r.json()["analysis"]["kind"] == "Bao 7"
    assert bare.post("/games/mega645/bao", json={"numbers": [1, 2, 3, 4]}).status_code == 422
    r = bare.post("/games/lotto535/bao", json={"numbers": [1, 2, 3, 4, 5, 6], "specials": [1, 2], "include_outcomes": True})
    a = r.json()["analysis"]
    assert r.status_code == 200 and a["plays"] == 12 and a["outcomes"] and not a["documented"]
    assert bare.post("/games/lotto535/bao", json={"numbers": [1, 2, 3, 4, 5, 6], "specials": [1, 2], "strict": True}).status_code == 422
    r = bare.post("/games/power655/bao", json={"numbers": [4, 15, 26, 37, 48, 50, 53], "jackpot1": 60e9, "tickets_sold": 1_500_000})
    assert r.status_code == 200 and 0 < r.json()["ev_with_co_winners"]["return_to_player"] < 1
    r = bare.post("/games/power655/coverage", json={"budget": 5, "sim_draws": 2000, "eval_draws": 10000, "candidates": 200, "local_rounds": 0})
    assert r.status_code == 200, r.text
    assert len(r.json()["tickets"]) == 5
    r = bare.post("/games/lotto535/rolldown/ev", json={"jackpot": 20e9, "tickets_sold": 1_500_000})
    assert r.status_code == 200 and r.json()["return_to_player"] > 1
    assert bare.post("/games/mega645/rolldown/ev", json={"jackpot": 20e9}).status_code == 422


def test_missing_artefacts_are_409(bare: TestClient) -> None:
    assert bare.get("/games/power655/market").status_code == 409
    assert bare.get("/games/lotto535/rolldown/history").status_code == 409
    assert bare.get("/games/lotto535/prizes").status_code == 409


def test_calibrated_models_are_used(calibrated: TestClient) -> None:
    r = calibrated.post("/games/power655/ev", json={"ticket": [1, 2, 3, 4, 5, 6], "jackpot1": 100e9}).json()
    assert r["tickets_sold_source"] == "sales_model"
    assert r["popularity_ratio"] > 1  # a pattern ticket is popular under the calibrated crowd
    m = calibrated.get("/games/power655/market").json()
    assert 0.3 < m["payout_share_used"] < 0.5
    h = calibrated.get("/games/lotto535/rolldown/history").json()
    assert h["announced"] >= 15 and h["executed"] <= h["announced"]
    r = calibrated.post("/games/lotto535/rolldown/ev", json={"jackpot": 20e9}).json()
    assert r["tickets_sold"] > 1_000_000  # default: recent rolldown sales


def test_bao_catalog_table_compare_and_max3d(bare: TestClient) -> None:
    cat = bare.get("/games/mega645/bao/catalog").json()
    assert [c["plays"] for c in cat] == [40, 7, 28, 84, 210, 462, 924, 1716, 3003, 5005, 18564]
    t = bare.get("/games/power655/bao/table?level=7").json()["rows"]
    assert {r["vietlott_style"] for r in t} >= {"200.000", "1.700.000", "82.500.000", "Jackpot 2 + 42.500.000", "Jackpot 1 + 240.000.000"}
    c = bare.post("/games/mega645/bao/compare", json={"numbers": [3, 9, 14, 22, 31, 38, 41], "sims": 5000}).json()
    assert c["rows"][0]["strategy"] == "Bao 7" and len(c["rows"]) >= 3
    prods = bare.get("/max3d").json()
    assert {p["code"] for p in prods} == {"max3d", "max3dplus", "max3dpro"}
    r = bare.post("/max3d/max3dpro/analyse", json={"kind": "bao_bo_so", "numbers": ["123", "456"], "sims": 10000}).json()
    assert r["plays"] == 36 and r["cost"] == 360_000
    assert bare.post("/max3d/keno/analyse", json={"numbers": ["123"]}).status_code == 404
    assert bare.post("/max3d/max3dpro/analyse", json={"kind": "bao_nhieu_bo_so", "numbers": ["123", "456"]}).status_code == 422
