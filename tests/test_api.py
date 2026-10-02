from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vietlott_engine.api.main import create_app
from vietlott_engine.core.config import Settings
from vietlott_engine.core.games import MEGA_645, POWER_655
from vietlott_engine.core.models import Draw
from vietlott_engine.crawler.storage import InMemoryRepository
from tests.conftest import make_history


@pytest.fixture(scope="module")
def client() -> TestClient:
    repo = InMemoryRepository()
    for spec, seed in ((MEGA_645, 11), (POWER_655, 12)):
        h = make_history(spec, 400, seed=seed)
        repo.upsert(
            Draw(game=spec.code, draw_id=i + 1, draw_date=str(h.dates[i]), numbers=tuple(int(x) for x in h.numbers[i]), bonus=int(h.bonus[i]) or None)
            for i in range(len(h))
        )
    app = create_app(Settings(storage_backend="memory"), repository=repo)
    with TestClient(app) as c:
        yield c


def test_health_and_games(client: TestClient) -> None:
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["draws"] == {"mega645": 400, "power655": 400, "lotto535": 0}
    games = client.get("/games").json()
    assert {g["code"] for g in games} == {"mega645", "power655", "lotto535"}


def test_draws_and_unknown_game(client: TestClient) -> None:
    r = client.get("/games/power655/draws?limit=3").json()
    assert r["total"] == 400 and len(r["draws"]) == 3 and r["draws"][0]["bonus"] is not None
    assert client.get("/games/keno/draws").status_code == 404
    assert client.get("/games/lotto535/draws").status_code == 409  # known game, no data loaded


@pytest.mark.parametrize(
    "url",
    [
        "/games/mega645/analytics/randomness?mc_sims=100",
        "/games/mega645/analytics/gaps",
        "/games/power655/analytics/cooccurrence?top=5&mc_sims=50",
        "/games/mega645/probability/bayesian?decay=0.99",
        "/games/mega645/probability/bayesian/decay-scan",
        "/games/power655/probability/markov?sims=50",
    ],
)
def test_analytics_endpoints(client: TestClient, url: str) -> None:
    r = client.get(url)
    assert r.status_code == 200, r.text


def test_ev_and_validation(client: TestClient) -> None:
    ok = client.post("/games/mega645/ev", json={"ticket": [3, 17, 22, 35, 41, 44], "jackpot1": 45e9})
    assert ok.status_code == 200 and 0 < ok.json()["return_to_player"] < 1
    bad = client.post("/games/mega645/ev", json={"ticket": [3, 17, 22, 35, 41, 50]})
    assert bad.status_code == 422 and bad.json()["error"] == "DataValidationError"


def test_optimize_wheel_backtest(client: TestClient) -> None:
    opt = client.post("/games/power655/ev/optimize", json={"n_tickets": 2, "iterations": 500, "seed": 1})
    assert opt.status_code == 200 and len(opt.json()["tickets"]) == 2
    wheel = client.post("/games/mega645/wheel", json={"pool": [1, 5, 9, 13, 17, 21, 25, 29, 33], "guarantee": 3, "condition": 3, "simulate": False, "time_limit": 5})
    assert wheel.status_code == 200 and wheel.json()["verified"] and wheel.json()["lower_bound"] <= wheel.json()["n_tickets"]
    bt = client.post("/games/mega645/backtest", json={"strategies": ["random", "bayes"], "start": 300, "tickets_per_draw": 2})
    assert bt.status_code == 200 and len(bt.json()["results"]) == 2
    too_many = client.post("/games/mega645/backtest", json={"tickets_per_draw": 500})
    assert too_many.status_code == 422


@pytest.mark.parametrize(
    "url",
    [
        "/games/mega645/inference/per-number?sims=200",
        "/games/mega645/inference/power",
        "/games/power655/inference/hierarchical",
        "/games/power655/inference/sequential",
        "/games/mega645/inference/changepoints?sims=100",
        "/games/mega645/inference/report?sims=200",
    ],
)
def test_inference_endpoints(client: TestClient, url: str) -> None:
    r = client.get(url)
    assert r.status_code == 200, r.text


def test_decision_and_uncertainty(client: TestClient) -> None:
    d = client.post("/games/mega645/ev/decision", json={"ticket": [33, 35, 37, 40, 42, 45], "jackpot1": 100e9, "tickets_sold": 4_000_000})
    assert d.status_code == 200
    body = d.json()
    assert body["kelly"]["optimal_fraction"] > 0 and body["payout"]["p_any_prize"] == pytest.approx(0.0238, abs=1e-3)
    u = client.post("/games/mega645/ev/uncertainty", json={"ticket": [33, 35, 37, 40, 42, 45], "jackpot1": 30e9, "sims": 50})
    assert u.status_code == 200 and u.json()["prob_rtp_above_one"] == 0.0
    bad = client.post("/games/mega645/ev/uncertainty", json={"ticket": [1, 2, 3, 4, 5, 6], "jackpot1": 30e9, "tickets_sold_low": 5, "tickets_sold_high": 1})
    assert bad.status_code == 422
