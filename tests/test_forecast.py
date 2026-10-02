"""v3.5: self-learning forecaster — proper laws, no look-ahead, chunk/reload invariance,
size of the anytime-valid evidence under fair draws, power on planted biases, ledger, CLI, API."""

from __future__ import annotations

import json
from itertools import combinations, product
from math import log
from pathlib import Path

import numpy as np
import pytest

from vietlott_engine.core.games import MEGA_645, POWER_655
from vietlott_engine.forecast.data import DigitSpec, Series, SetSpec, digit_counts, load_series, matrix_series
from vietlott_engine.forecast.engine import Component, Forecaster, inclusion_probabilities, load_or_new, match_distribution, record, refresh, score_ledger, scoreboard, state_path
from vietlott_engine.forecast.experts import DigitExperts, DigitTilts, SetExperts, SetTilts, set_loglik
from vietlott_engine.game_theory.popularity import sample_product_subsets
from tests.conftest import make_history

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"
LOG20 = log(20)


def _set_obs(nums: np.ndarray, n: int, bonus: np.ndarray | None = None) -> dict:
    x = np.zeros((len(nums), n), dtype=bool)
    x[np.repeat(np.arange(len(nums)), nums.shape[1]), nums.ravel() - 1] = True
    return {"X": x, "bonus": np.zeros(len(nums), dtype=np.int16) if bonus is None else bonus.astype(np.int16)}


def _run(comp: Component, obs: dict, chunk: int | None = None) -> tuple[np.ndarray, dict]:
    st = comp.init_state()
    t = len(next(iter(obs.values())))
    ids = np.arange(1, t + 1)
    out = []
    step = chunk or t
    for a in range(0, t, step):
        incr, st = comp.process(st, {k: v[a : a + step] for k, v in obs.items()}, ids[a : a + step])
        out.append(incr)
    return np.concatenate(out), st


# ------------------------------------------------------------------ proper laws
def test_every_expert_is_a_probability_law_over_all_outcomes() -> None:
    rng = np.random.default_rng(0)
    n, k = 7, 3
    spec = SetSpec(n, k, bonus_same_drum=True)
    hist = np.array([rng.choice(n, k, replace=False) + 1 for _ in range(40)])
    bon = np.array([rng.choice(np.setdiff1d(np.arange(1, n + 1), r)) for r in hist])
    ad = SetExperts(spec, batch=5)
    _, pred, _ = ad.run(ad.init_state(), _set_obs(hist, n, bon))
    outs = [(s, b) for s in combinations(range(1, n + 1), k) for b in range(1, n + 1) if b not in s]
    o_nums = np.array([s for s, _ in outs])
    o_obs = _set_obs(o_nums, n, np.array([b for _, b in outs]))
    w_next = np.repeat(pred[-1][None], len(outs), axis=0)
    total = np.exp(set_loglik(w_next, o_obs["X"], o_obs["bonus"], k, True)).sum(axis=0)
    assert np.allclose(total, 1.0)
    ll_t, _, _ = SetTilts(spec).run({}, o_obs)
    assert np.allclose(np.exp(ll_t).sum(axis=0), 1.0)
    # digit games: A=3 symbols, L=2 positions, m=2 numbers per draw
    dspec = DigitSpec(3, 2, 2, ("a", "b"))
    draws = np.array(list(product(range(3), repeat=4))).reshape(-1, 2, 2)  # every (m, L) outcome
    c = digit_counts(draws, 3)
    dex = DigitExperts(dspec)
    _, q, _ = dex.run(dex.init_state(), {"C": digit_counts(rng.integers(0, 3, (30, 2, 2)), 3)})
    lq = np.log(q[-1].reshape(len(dex.names), 2, 3))
    assert np.allclose(np.exp(np.einsum("jla,tla->tj", lq, c)).sum(axis=0), 1.0)
    ll_d, _, _ = DigitTilts(dspec).run({}, {"C": c})
    assert np.allclose(np.exp(ll_d).sum(axis=0), 1.0)


def test_inclusion_and_match_probabilities_match_enumeration() -> None:
    rng = np.random.default_rng(1)
    n, k = 8, 3
    w = rng.uniform(0.5, 2.0, (2, n))
    subsets = list(combinations(range(n), k))
    p = np.array([[np.prod(row[list(s)]) for s in subsets] for row in w])
    p /= p.sum(axis=1, keepdims=True)
    incl = np.array([[sum(pp for pp, s in zip(prow, subsets) if i in s) for i in range(n)] for prow in p])
    assert np.allclose(inclusion_probabilities(w, k), incl)
    ticket = np.array([0, 3, 5, 6])
    md = np.array([[sum(pp for pp, s in zip(prow, subsets) if len(set(s) & set(ticket)) == r) for r in range(len(ticket) + 1)] for prow in p])
    assert np.allclose(match_distribution(w, ticket, k), md)


# ------------------------------------------------------------------ learning mechanics
def test_no_lookahead() -> None:
    comp = Component("main", SetSpec(20, 4), batch=7, switch_rate=1e-3)
    nums = np.array([np.sort(np.random.default_rng(i).choice(20, 4, replace=False) + 1) for i in range(120)])
    obs = _set_obs(nums, 20)
    other = {k: v.copy() for k, v in obs.items()}
    other["X"][80:] = other["X"][80:][::-1]  # rewrite the future
    a, _ = _run(comp, obs)
    b, _ = _run(comp, other)
    assert np.array_equal(a[:80], b[:80]) and not np.allclose(a[80:], b[80:])


def test_chunking_and_save_reload_do_not_change_learning(tmp_path: Path) -> None:
    h = make_history(POWER_655, 300, seed=4)
    series = matrix_series(h)
    whole = Forecaster("power655")
    whole.update(series)
    parts = Forecaster("power655")
    for cut in (37, 38, 150, 151, 300):  # odd cuts split the logistic mini-batches
        parts.update(series.head(cut), chunk=23)
        parts.save(state_path(tmp_path, parts.product))
        parts = load_or_new(tmp_path, parts.product)
    a, b = whole.state, parts.state
    assert a["last_id"] == b["last_id"] and a["draws"] == b["draws"] == 300
    assert np.isclose(a["log_wealth"], b["log_wealth"], rtol=0, atol=1e-9)
    ca, cb = a["components"]["main"], b["components"]["main"]
    assert np.allclose(ca["v"], cb["v"], atol=1e-12) and np.allclose(ca["groups"][0]["theta"], cb["groups"][0]["theta"], atol=1e-12)
    assert ca["path"] == cb["path"] and ca["hits"] == cb["hits"]


# ------------------------------------------------------------------ evidence: size and power
def test_evidence_is_rarely_found_on_fair_draws() -> None:
    rejections, uniform_w = 0, []
    for seed in range(60):
        rng = np.random.default_rng(100 + seed)
        nums = np.argpartition(rng.random((250, 20)), 4, axis=1)[:, :4] + 1
        comp = Component("main", SetSpec(20, 4), batch=10, switch_rate=1 / 1500)
        incr, st = _run(comp, _set_obs(nums, 20))
        rejections += np.cumsum(incr).max() >= LOG20
        uniform_w.append(st["v"][0])
    assert rejections <= 7  # α = 5 % → 3 expected; Ville's bound holds for every sample size
    dig = 0
    for seed in range(30):
        rng = np.random.default_rng(500 + seed)
        comp = Component("digits", DigitSpec(10, 3, 20, ("h", "t", "u")), batch=10, switch_rate=1 / 1500)
        incr, _ = _run(comp, {"C": digit_counts(rng.integers(0, 10, (250, 20, 3)), 10)})
        dig += np.cumsum(incr).max() >= LOG20
    assert dig <= 4


def test_planted_biases_are_learned_and_proven() -> None:
    rng = np.random.default_rng(7)
    probs = np.full(10, 0.87 / 9)
    probs[6] = 0.13  # units digit 6 comes up 30 % more often
    d = rng.integers(0, 10, (400, 20, 3))
    d[:, :, 2] = rng.choice(10, size=(400, 20), p=probs)
    comp = Component("digits", DigitSpec(10, 3, 20, ("trăm", "chục", "đơn vị")), batch=10, switch_rate=1 / 1500)
    incr, st = _run(comp, {"C": digit_counts(d, 10)})
    assert np.cumsum(incr).max() >= LOG20
    fc = comp.forecast(st)
    assert fc.evidence.found and fc.picks[2] == "6" and fc.digit.positions[2].symbols[0].symbol == "6"  # type: ignore[union-attr]
    # k-of-n: number 7 has 1.8× the odds of the others in a 6/45 game
    w = np.ones(45)
    w[6] = 1.8
    nums = sample_product_subsets(w, 6, 900, np.random.default_rng(8))
    comp = Component("main", SetSpec(45, 6), batch=20, switch_rate=1 / 1560)
    incr, st = _run(comp, _set_obs(nums, 45))
    fc = comp.forecast(st)
    assert np.cumsum(incr).max() >= LOG20 and fc.set.numbers[0].symbol == "7" and 7 in fc.picks  # type: ignore[union-attr]
    assert fc.pick_score.ratio > 1.0


def test_real_data_findings() -> None:
    """The bundled history: the only signal the forecaster finds by itself is the units digit 6 of Max 3D."""
    f = Forecaster("max3d")
    f.update(load_series("max3d", seed_dir=SEED))
    rep = f.forecast()
    assert rep.evidence.found and rep.components[0].digit.positions[2].symbols[0].symbol == "6"  # type: ignore[union-attr]
    assert rep.components[0].digit.bets[0].rtp_model < 1  # type: ignore[union-attr]
    m = Forecaster("mega645")
    m.update(load_series("mega645", seed_dir=SEED))
    rm = m.forecast()
    assert not rm.evidence.found and "CHƯA" in rm.verdict and rm.components[0].uniform_weight > 0.3


# ------------------------------------------------------------------ ledger, CLI, API
def test_ledger_records_once_then_scores(tmp_path: Path) -> None:
    series = load_series("max3dpro", seed_dir=SEED)
    cut = int(np.flatnonzero(np.diff(series.draw_ids) == 1)[-5]) + 1  # a draw whose next id exists
    f = Forecaster("max3dpro")
    f.update(series.head(cut))
    rep = f.forecast()
    record(tmp_path, rep)
    record(tmp_path, rep)
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1
    scored = score_ledger(tmp_path, series)
    assert len(scored) == 1 and scored[0]["target_id"] == int(series.draw_ids[cut])
    board = scoreboard(tmp_path)
    assert board[0]["scored"] == 1 and board[0]["expected"] == pytest.approx(20 * 3 / 10)
    f2, learned, again = refresh("max3dpro", series, tmp_path)
    assert learned == len(series) and again == []  # no fresh state saved yet → learns all; nothing left to score


def test_cli_and_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from fastapi.testclient import TestClient

    from vietlott_engine.api.main import create_app
    from vietlott_engine.cli import main
    from vietlott_engine.core.config import Settings, get_settings

    monkeypatch.setenv("VQE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VQE_STORAGE_BACKEND", "memory")
    monkeypatch.setenv("VQE_FORECAST_NOW", "2026-09-30T20:00:00+07:00")  # Max 3D #1139 drawn at 18:00; next draw Fri 02/10 18:00
    get_settings.cache_clear()
    try:
        assert main(["forecast", "next", "--product", "max3d", "--json"]) == 0
        rep = json.loads(capsys.readouterr().out)[0]
        assert rep["product"] == "max3d" and len(rep["components"][0]["picks"]) == 3 and rep["recorded"] is True
        assert main(["forecast", "next", "--product", "max4d"]) == 0
        assert "không ghi vào sổ" in capsys.readouterr().out  # discontinued: nothing to pre-register
        assert main(["forecast", "update", "--product", "lotto535"]) == 0
        assert "học thêm" in capsys.readouterr().out
        assert main(["forecast", "scoreboard"]) == 0
        board = capsys.readouterr().out
        assert "max3d" in board and "max4d" not in board
        assert main(["doctor", "--offline"]) == 0
        assert "Cài đặt dùng được" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()
    s = Settings(storage_backend="memory", seed_file_dir=SEED, product_seed_dir=SEED, data_dir=tmp_path / "api")
    with TestClient(create_app(s)) as c:
        r = c.get("/forecast/max3dpro", params={"record": True}).json()  # VQE_FORECAST_NOW: Pro #786 (Thu 01/10) is in, next Sat 03/10
        assert r["recorded"] is True and r["components"][0]["digit"]["bets"]
        assert c.get("/forecast/max3dpro/scoreboard").json()[0]["recorded"] == 1
        b = c.get("/forecast/bingo18", params={"record": True}).json()  # Bingo18 data end 02/10 but "now" is 30/09: refused
        assert b["recorded"] is False and b["evidence"]["threshold_log10"] == pytest.approx(1.30103, abs=1e-4)
        path = c.get("/forecast/max3dpro/evidence").json()[0]
        assert path["path"] and max(p[1] for p in path["path"]) > 1.30
        assert c.post("/forecast/mega645/fit", params={"last": 300}).json()["draws"] == 300
        assert c.get("/forecast/nope").status_code == 404


def test_series_after_and_tail() -> None:
    s = matrix_series(make_history(MEGA_645, 50, seed=9))
    assert len(s.after(int(s.draw_ids[39]))) == 10 and len(s.tail(5)) == 5 and isinstance(s.head(3), Series)


def test_chosen_window_is_never_evidence_and_ticket_uses_exact_probabilities(tmp_path: Path) -> None:
    series = load_series("lotto535", seed_dir=SEED)
    f, learned, _ = refresh("lotto535", series, tmp_path, refit=True, last=300)
    rep = f.forecast()
    assert learned == 300 and rep.evidence.max_log10_wealth > 1.30  # this window happens to cross 20× …
    assert not rep.evidence.valid and not rep.evidence.found and "--last" in rep.verdict  # … and is not reported as evidence
    assert all(not c.evidence.found for c in rep.components)
    again, _, _ = refresh("lotto535", series, tmp_path)  # later updates keep the flag until a full refit
    assert not again.forecast().evidence.valid
    p = Forecaster("power655")
    p.update(load_series("power655", seed_dir=SEED))
    c = p.forecast().components[0]
    top = sorted(int(x.symbol) for x in c.set.numbers[:6])  # type: ignore[union-attr]
    assert c.set.ticket == top == c.picks  # type: ignore[union-attr]


def test_ledger_only_takes_forecasts_made_before_their_draw() -> None:
    from datetime import datetime, timedelta, timezone

    from vietlott_engine.core.products import ProductCode as P
    from vietlott_engine.forecast.schedule import record_window

    vn = timezone(timedelta(hours=7))
    fri_evening = datetime(2026, 10, 2, 19, 29, tzinfo=vn)

    def ok(code, dates, now=fri_evening):  # type: ignore[no-untyped-def]
        return record_window(code, np.array(dates, dtype="datetime64[D]"), now)[0]

    assert not ok(P.MAX3D, ["2026-09-30"])  # Fri 18:00 draw already held, data stops Wed
    assert not ok(P.MEGA_645, ["2026-09-30"])
    assert ok(P.POWER_655, ["2026-10-01"])  # next Power draw is Sat 18:00
    assert not ok(P.LOTTO_535, ["2026-10-01", "2026-10-01"])  # 02/10 13:00 already held
    assert ok(P.LOTTO_535, ["2026-10-02"], datetime(2026, 10, 2, 16, 0, tzinfo=vn))  # 13:00 in, 21:00 ahead
    assert not ok(P.MAX4D, ["2021-08-31"])  # discontinued
    keno_day = ["2026-10-01"] * 100 + ["2026-10-02"] * 100
    assert not ok(P.KENO, keno_day)  # 19:29: today's draws still running
    assert ok(P.KENO, keno_day, datetime(2026, 10, 2, 23, 30, tzinfo=vn))  # overnight gap, day complete
    assert not ok(P.KENO, ["2026-10-01"] * 100 + ["2026-10-02"] * 60, datetime(2026, 10, 2, 23, 30, tzinfo=vn))  # day incomplete
    assert not ok(P.KENO, keno_day, datetime(2026, 10, 3, 6, 10, tzinfo=vn))  # next morning, draws started
