"""v3: Lotto 5/35 rules, prize data, market accounting, crowd calibration, bao tickets,
coverage portfolios and rolldowns."""

from __future__ import annotations

import json
from datetime import date, timedelta
from itertools import combinations
from math import comb

import numpy as np
import pytest

from vietlott_engine.core.exceptions import DataValidationError
from vietlott_engine.core.games import LOTTO_535, MEGA_645, POWER_655, GameSpec
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.models import Draw
from vietlott_engine.core.prizes import PrizeHistory, PrizeRecord
from vietlott_engine.crawler.prize_sources import parse_compal_winners, parse_leoodz_power, reconcile
from vietlott_engine.game_theory.bao import analyse_bao, bao_tickets
from vietlott_engine.game_theory.coverage import optimise_coverage
from vietlott_engine.game_theory.ev import EVCalculator
from vietlott_engine.game_theory.popularity import PopularityModel, PopularityParams, sample_product_subsets
from vietlott_engine.game_theory.rolldown import detect_rolldowns, realised_tier_prizes, rolldown_ev, rolldown_history
from vietlott_engine.game_theory.sales import estimate_payout_share, tickets_sold_from_accounting
from tests.conftest import make_history


# ------------------------------------------------------------------- helpers
def _masks(arr: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(arr).astype(np.uint64)
    return np.bitwise_or.reduce(np.left_shift(np.uint64(1), arr - np.uint64(1)), axis=1)


def _tier_table(spec: GameSpec) -> tuple[np.ndarray, np.ndarray]:
    """index[m, hit] of the tier (−1 = no prize) and its position in spec.tiers."""
    names = [t.name for t in spec.tiers]
    idx = np.full((spec.pick + 1, 2), -1)
    for m in range(spec.pick + 1):
        for hit in (0, 1):
            t = spec.classify(m, bool(hit))
            idx[m, hit] = names.index(t.name) if t else -1
    return idx, np.array(names)


def _simulate_winners(spec: GameSpec, h: DrawHistory, sold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Winner counts of uniformly random tickets: multinomial over the exact tier probabilities."""
    probs = np.array([spec.tier_probabilities[t.name] for t in spec.tiers])
    out = np.zeros((len(h), len(probs)), dtype=np.int64)
    for t in range(len(h)):
        out[t] = rng.multinomial(int(sold[t]), np.append(probs, 1 - probs.sum()))[:-1]
    return out


def _records(spec: GameSpec, h: DrawHistory, winners: np.ndarray, pots: np.ndarray | None = None) -> list[PrizeRecord]:
    names = [t.name for t in spec.tiers]
    jps = [t.name for t in spec.tiers if t.is_jackpot]
    recs = []
    for i in range(len(h)):
        recs.append(
            PrizeRecord(
                game=spec.code,
                draw_id=int(h.draw_ids[i]),
                draw_date=h.dates[i].astype("datetime64[D]").astype(date),
                winners={nm: int(winners[i, j]) for j, nm in enumerate(names)},
                jackpot_pots={nm: int(pots[i, j]) for j, nm in enumerate(jps)} if pots is not None else None,
            )
        )
    return recs


# ------------------------------------------------------------------- rules
def test_lotto_tier_probabilities_by_enumeration() -> None:
    spec = LOTTO_535
    win = np.array([3, 9, 17, 25, 33])
    combos = np.array(list(combinations(range(1, 36), 5)))
    m = np.bitwise_count(_masks(combos) & _masks(win)[0])
    counts = np.bincount(m, minlength=6) / comb(35, 5)
    probs = spec.tier_probabilities
    s_hit, s_miss = 1 / 12, 11 / 12
    assert probs["jackpot1"] == pytest.approx(counts[5] * s_hit)
    assert probs["first"] == pytest.approx(counts[5] * s_miss)
    assert probs["second"] == pytest.approx(counts[4] * s_hit)
    assert probs["third"] == pytest.approx(counts[4] * s_miss)
    assert probs["fourth"] == pytest.approx(counts[3] * s_hit)
    assert probs["fifth"] == pytest.approx(counts[3] * s_miss)
    assert probs["consolation"] == pytest.approx(counts[:3].sum() * s_hit)
    assert spec.p_any_prize == pytest.approx(0.0960, abs=5e-4)
    assert MEGA_645.p_any_prize == pytest.approx(0.0238, abs=2e-4)
    assert POWER_655.p_any_prize == pytest.approx(0.0133, abs=2e-4)


def test_lotto_ev_needs_special_and_uniform_crowd_is_textbook() -> None:
    spec = LOTTO_535
    uniform = PopularityModel(spec, PopularityParams(quick_pick_share=1.0))
    calc = EVCalculator(spec, uniform)
    with pytest.raises(DataValidationError):
        calc.evaluate([1, 2, 3, 4, 5], 9e9, tickets_sold=200_000)
    with pytest.raises(DataValidationError):
        calc.evaluate([1, 2, 3, 4, 5], 9e9, tickets_sold=200_000, special=13)
    ev = calc.evaluate([1, 2, 3, 4, 5], 9e9, tickets_sold=200_000, special=4)
    assert ev.popularity_ratio == pytest.approx(1.0, rel=1e-6)
    fixed = sum(spec.tier_probabilities[t.name] * calc.tax.after_tax(t.fixed_amount) for t in spec.tiers if t.fixed_amount)
    assert ev.expected_value > fixed and 0 < ev.return_to_player < 1


def test_power_cap_spills_into_jackpot2() -> None:
    calc = EVCalculator(POWER_655, PopularityModel(POWER_655, PopularityParams(quick_pick_share=1.0)))
    t = [1, 12, 23, 34, 45, 55]
    below = {x.tier: x for x in calc.evaluate(t, 290e9, 5e9, tickets_sold=1_000_000).tiers}
    above = {x.tier: x for x in calc.evaluate(t, 350e9, 5e9, tickets_sold=1_000_000).tiers}
    assert below["jackpot2"].gross_prize == pytest.approx(5e9)
    assert above["jackpot2"].gross_prize > 5e9 + 40e9  # ≈ (350 − 300) tỷ × P(JP1 not won)


# ------------------------------------------------------------------- prize data
def test_reconcile_prefers_verified_counts_and_realigns_pots() -> None:
    spec = POWER_655
    draws = [
        Draw(game=spec.code, draw_id=i, draw_date=date(2024, 1, 2) + timedelta(days=2 * i), numbers=(1 + i, 10, 20, 30, 40, 50), bonus=55)
        for i in range(1, 5)
    ]
    compal = "\n".join(
        json.dumps({"id": d.draw_id, "date": d.draw_date.isoformat(), "result": [*d.numbers, 55], "winners": {"Jackpot 1": 0, "Jackpot 2": 0, "Giải nhất": d.draw_id, "Giải nhì": 500, "Giải ba": 9000}})
        for d in draws
    )
    # leoodz: same counts but the pot of draw k is published one position later
    leo = []
    for d in draws:
        prev = d.draw_id - 1 if d.draw_id > 1 else 1
        leo.append(
            {
                "date": d.draw_date.strftime("%d-%m-%Y"),
                "numbers": list(d.numbers),
                "bonus": 55,
                "prizes": {
                    "jackpot1": {"count": 0, "amount": f"{30 + prev},000,000,000đ"},
                    "jackpot2": {"count": 0, "amount": "3,000,000,000đ"},
                    "giai_nhat": {"count": prev, "amount": "40,000,000đ"},
                    "giai_nhi": {"count": 500, "amount": "500,000đ"},
                    "giai_ba": {"count": 9000, "amount": "50,000đ"},
                },
            }
        )
    records, rep = reconcile(spec, draws, parse_compal_winners(compal, spec.code), parse_leoodz_power(json.dumps(leo)))
    assert rep.with_winners == 4 and not rep.number_mismatches
    assert all(r.source == "compal123" for r in records)
    pots = {r.draw_id: r.jackpot_pots for r in records}
    assert pots[2]["jackpot1"] == 32_000_000_000  # matched to the leoodz row with the same counts (realigned)
    assert rep.realigned >= 1


def test_payout_share_and_sales_identity_recovered() -> None:
    spec = POWER_655
    rng = np.random.default_rng(5)
    h = make_history(spec, 400, seed=3)
    sold = np.exp(rng.normal(np.log(800_000), 0.25, len(h))).round()
    w = _simulate_winners(spec, h, sold, rng)
    s_true = 0.41
    fixed = (w * np.array([t.fixed_amount or 0 for t in spec.tiers])).sum(axis=1)
    pots = np.zeros((len(h), 2))
    cur = np.array([30e9, 3e9])
    names = [t.name for t in spec.tiers]
    for t in range(len(h)):
        growth = s_true * sold[t] * spec.ticket_price - fixed[t]
        cur = cur + growth * np.array([0.9, 0.1])
        pots[t] = cur
        won = w[t, [names.index("jackpot1"), names.index("jackpot2")]] > 0
        cur = np.where(won, [30e9, 3e9], cur)
    ph = PrizeHistory.align(h, _records(spec, h, w, pots))
    share = estimate_payout_share(ph, h.dates)
    assert share.overall == pytest.approx(s_true, rel=0.03)
    assert share.ci95[0] < s_true < share.ci95[1]
    n_hat = tickets_sold_from_accounting(ph, s_true)
    ok = np.isfinite(n_hat)
    assert ok.mean() > 0.95
    np.testing.assert_allclose(n_hat[ok], sold[ok], rtol=1e-6)


# ------------------------------------------------------------------- calibration
@pytest.mark.slow
def test_behaviour_calibration_recovers_planted_crowd() -> None:
    from vietlott_engine.game_theory.calibration import calibrate, fitted_tickets_sold

    spec = MEGA_645
    rng = np.random.default_rng(11)
    n, k = spec.pool_size, spec.pick
    beta = np.where(np.arange(1, n + 1) <= 31, 0.45, -0.35) + rng.normal(0, 0.15, n)
    beta -= beta.mean()
    q = 0.4
    h = make_history(spec, 260, seed=9)
    sold = np.exp(rng.normal(np.log(40_000), 0.2, len(h))).round().astype(int)
    names = [t.name for t in spec.tiers]
    tier_idx, _ = _tier_table(spec)
    w = np.zeros((len(h), len(names)), dtype=np.int64)
    wm = _masks(h.numbers)
    for t in range(len(h)):
        n_qp = rng.binomial(sold[t], q)
        qp = np.argpartition(rng.random((n_qp, n)), k, axis=1)[:, :k] + 1
        manual = sample_product_subsets(np.exp(beta), k, sold[t] - n_qp, rng)
        m = np.bitwise_count(_masks(np.vstack([qp, manual])) & wm[t])
        tiers = tier_idx[m, 0]
        w[t] = np.bincount(tiers[tiers >= 0], minlength=len(names))
    ph = PrizeHistory.align(h, _records(spec, h, w))
    cal = calibrate(h, ph, sold.astype(float), holdout_fraction=0.0, compute_se=False)
    assert np.corrcoef(cal.beta, beta)[0, 1] > 0.8
    assert cal.quick_pick_share == pytest.approx(q, abs=0.15)
    assert set(cal.most_popular_numbers[:5]) <= set(range(1, 32))
    cal_unknown = calibrate(h, ph, None, holdout_fraction=0.0, compute_se=False)
    n_hat = fitted_tickets_sold(cal_unknown, h, ph)
    ratio = n_hat / sold
    assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.12)


# ------------------------------------------------------------------- bao
def _mc_bao(spec: GameSpec, numbers: list[int], specials: list[int] | None, draws: int, rng: np.random.Generator) -> float:
    plays = bao_tickets(spec, numbers, specials)
    pm = _masks(np.array([p[0] for p in plays]))
    ps = np.array([p[1] or 0 for p in plays])
    n, k = spec.pool_size, spec.pick
    tier_idx, _ = _tier_table(spec)
    hits = 0
    for _ in range(draws // 20_000):
        if spec.bonus_mode == "same_drum":
            perm = np.argsort(rng.random((20_000, n)), axis=1)[:, : k + 1] + 1
            wm, bonus = _masks(perm[:, :k]), perm[:, k]
            hit = (pm[:, None] >> (bonus.astype(np.uint64) - np.uint64(1))[None, :]) & np.uint64(1)
            hit = hit.astype(bool)
        else:
            wm = _masks(np.argpartition(rng.random((20_000, n)), k, axis=1)[:, :k] + 1)
            sp = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, 20_000)
            hit = ps[:, None] == sp[None, :] if spec.separate_special else np.zeros((len(pm), 20_000), bool)
        m = np.bitwise_count(pm[:, None] & wm[None, :])
        won = tier_idx[m, hit.astype(int)] >= 0
        hits += int(won.any(axis=0).sum())
    return hits / (draws // 20_000 * 20_000)


@pytest.mark.parametrize(
    "spec,numbers,specials",
    [
        (MEGA_645, [2, 9, 14, 23, 31, 38, 44], None),  # Bao 7
        (POWER_655, [4, 15, 26, 37, 48], None),  # Bao 5 (k−1)
        (POWER_655, [1, 8, 15, 22, 29, 36, 43, 50], None),  # Bao 8, bonus inside the set possible
        (LOTTO_535, [3, 8, 13, 21, 34, 35], [2, 7]),  # Lotto bao 6 with two specials
        (LOTTO_535, [5, 10, 20, 30], [11]),  # Lotto bao 4 (k−1)
    ],
)
def test_bao_exact_matches_simulation(spec: GameSpec, numbers: list[int], specials: list[int] | None) -> None:
    res = analyse_bao(spec, numbers, specials)
    assert res.plays == len(bao_tickets(spec, numbers, specials))
    assert sum(o.probability for o in res.outcomes) == pytest.approx(1.0, abs=1e-12)
    # linearity of the fixed prizes: E[fixed] = plays × single-ticket E[fixed]
    single_fixed = sum(spec.tier_probabilities[t.name] * t.fixed_amount for t in spec.tiers if t.fixed_amount is not None)
    exp_fixed = sum(o.probability * o.fixed_prizes for o in res.outcomes)
    assert exp_fixed == pytest.approx(res.plays * single_fixed, rel=1e-9)
    # jackpots: own plays share one pot, so the bao gets at most the singles' jackpot EV
    single_jp = sum(spec.tier_probabilities[t.name] * spec.min_jackpots[t.name] for t in spec.tiers if t.is_jackpot)
    assert res.expected_payout - exp_fixed <= res.plays * single_jp * (1 + 1e-9)
    p_mc = _mc_bao(spec, numbers, specials, 200_000, np.random.default_rng(1))
    se = np.sqrt(res.p_any_prize * (1 - res.p_any_prize) / 200_000)
    assert abs(p_mc - res.p_any_prize) < 4.5 * se + 1e-4
    assert res.p_any_prize <= res.p_any_prize_same_budget_single_tickets + 1e-12


def test_bao_validation() -> None:
    with pytest.raises(ValueError):
        analyse_bao(MEGA_645, [1, 2, 3, 4])
    with pytest.raises(ValueError):
        analyse_bao(LOTTO_535, [1, 2, 3, 4, 5, 6])  # Lotto needs specials
    with pytest.raises(ValueError):
        analyse_bao(MEGA_645, [1, 2, 3, 4, 5, 6, 7], specials=[1])


# ------------------------------------------------------------------- coverage
def test_coverage_beats_random_and_respects_bound() -> None:
    res = optimise_coverage(MEGA_645, 12, sim_draws=6000, eval_draws=40_000, candidates=600, local_rounds=1, seed=3)
    assert len(res.tickets) == 12 and all(len(set(t)) == 6 for t in res.tickets)
    assert res.p_at_least_one >= res.p_random_tickets - 0.01
    assert res.p_at_least_one <= res.upper_bound + 0.01
    assert res.expected_prizes_per_draw == pytest.approx(12 * MEGA_645.p_any_prize)
    lotto = optimise_coverage(LOTTO_535, 12, sim_draws=4000, eval_draws=20_000, candidates=400, local_rounds=0, seed=1)
    assert sorted(lotto.specials) == list(range(1, 13))  # one ticket per special number covers the consolation tier
    assert lotto.p_at_least_one > 0.99
    with pytest.raises(ValueError):
        optimise_coverage(MEGA_645, 3, min_tier="nope", sim_draws=2000, eval_draws=10_000, candidates=100)


# ------------------------------------------------------------------- rolldown
def test_rolldown_ev_monotone_and_break_even() -> None:
    lo = rolldown_ev(LOTTO_535, 20e9, 500_000)
    hi = rolldown_ev(LOTTO_535, 20e9, 4_000_000)
    assert lo.return_to_player > hi.return_to_player
    assert lo.base_return_to_player < 1
    assert lo.break_even_tickets_sold is not None
    at = rolldown_ev(LOTTO_535, 20e9, int(lo.break_even_tickets_sold))
    assert at.return_to_player == pytest.approx(1.0, abs=0.01)
    shares = LOTTO_535.rolldown.shares
    # fractions are conditional on the tier having a winner: empty tiers top up the others
    assert all(lo.tier_fractions[t] >= f for t, f in shares.items())
    assert sum(hi.tier_fractions.values()) == pytest.approx(1.0, abs=0.01)  # big sales: every tier is won
    with pytest.raises(ValueError):
        rolldown_ev(MEGA_645, 20e9, 1_000_000)


def test_realised_tier_prizes_redistribute_empty_tiers() -> None:
    w = {"first": 0, "second": 24, "third": 400, "fourth": 700, "fifth": 11000}
    p = realised_tier_prizes(LOTTO_535, w, 16.5e9)
    assert "first" not in p
    paid = sum((p[t] - LOTTO_535.tier(t).fixed_amount) * w[t] for t in p)
    assert paid == pytest.approx(16.5e9)
    assert p["second"] == pytest.approx(5e6 + 16.5e9 / 4 / 24)  # 1/6 + (1/3)/4 = 1/4 of the pot


def _simulate_lotto_market(c: float, c_r: float, days: int, seed: int) -> tuple[DrawHistory, PrizeHistory, np.ndarray, list[int], dict[int, float]]:
    spec = LOTTO_535
    rng = np.random.default_rng(seed)
    d0 = date(2025, 7, 1)
    dates = [d0 + timedelta(days=i // 2) for i in range(2 * days)]
    sold = np.exp(rng.normal(np.log(180_000), 0.15, len(dates)))
    j, pending = 6e9, None
    rolls: list[int] = []
    won = np.zeros(len(dates), bool)
    pot: dict[int, float] = {}
    for t, d in enumerate(dates):
        rd_day = pending is not None and d == pending
        if rd_day and (t + 1 == len(dates) or dates[t + 1] != d):
            sold[t] *= 8.0  # players pile into the rolldown draw
        j += (c_r if rd_day else c) * sold[t] * spec.ticket_price
        pot[t + 1] = j
        if rng.random() < 0.02:
            won[t], j, pending = True, 6e9, None
            continue
        if rd_day and (t + 1 == len(dates) or dates[t + 1] != d):
            rolls.append(t)
            j, pending = 6e9, None
            continue
        if pending is None and j > spec.rolldown.threshold:
            pending = d + timedelta(days=1)
    nums = np.argpartition(rng.random((len(dates), 35)), 5, axis=1)[:, :5] + 1
    h = DrawHistory.from_arrays(spec, nums, dates=np.array(dates, dtype="datetime64[D]"), bonus=rng.integers(1, 13, len(dates)))
    w = _simulate_winners(spec, h, sold, rng)
    w[:, 0] = won.astype(int)
    return h, PrizeHistory.align(h, _records(spec, h, w)), sold, rolls, pot


def test_rolldown_history_recovers_events_and_accrual() -> None:
    h, ph, sold, rolls, pot = _simulate_lotto_market(c=0.15, c_r=0.45, days=320, seed=4)
    assert len(rolls) >= 8
    assert set(rolls) <= set(detect_rolldowns(h, sold))
    anchors = [(int(h.draw_ids[r]), pot[int(h.draw_ids[r])]) for r in rolls[:3]]
    rh = rolldown_history(LOTTO_535, h, ph, sold, anchors)
    executed = [int(h.draw_ids[r]) for r in rolls]
    assert [e.draw_id for e in rh.events if e.executed] == executed
    assert rh.accrual_rate == pytest.approx(0.15, rel=0.12)
    assert rh.rolldown_day_rate == pytest.approx(0.45, rel=0.2)
    for a in rh.anchors:
        assert a["ratio"] == pytest.approx(1.0, abs=0.1)
    assert rh.timing_consistency > 0.5
    for e in rh.events:
        if e.executed:
            assert sum(e.tier_prizes.values()) > 0 and e.realised_rtp_pre_tax > 0
