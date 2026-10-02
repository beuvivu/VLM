"""Chơi bao: official catalogue, published prize examples, jackpot sharing, tax basis,
crowd-aware EV, strategy comparison, and the Max 3D family."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from vietlott_engine.core.games import DEFAULT_TAX, LOTTO_535, MEGA_645, POWER_655
from vietlott_engine.game_theory import max3d
from vietlott_engine.game_theory.bao import (
    analyse_bao,
    bao_catalog,
    bao_ev,
    bao_prize_table,
    bao_tickets,
    compare_bao_strategies,
    expected_share,
    simulate_plays,
    ticket_payout,
)


def _styles(spec, level, specials=1):  # type: ignore[no-untyped-def]
    return {(r["hits_in_set"], r["bonus"]): r["vietlott_style"] for r in bao_prize_table(spec, level, specials)}


# ------------------------------------------------------------------ catalogue
def test_catalog_matches_vietlott_price_list() -> None:
    mega = [(o.kind, o.plays, o.cost) for o in bao_catalog(MEGA_645)]
    assert mega == [
        ("Bao 5", 40, 400_000), ("Bao 7", 7, 70_000), ("Bao 8", 28, 280_000), ("Bao 9", 84, 840_000),
        ("Bao 10", 210, 2_100_000), ("Bao 11", 462, 4_620_000), ("Bao 12", 924, 9_240_000), ("Bao 13", 1716, 17_160_000),
        ("Bao 14", 3003, 30_030_000), ("Bao 15", 5005, 50_050_000), ("Bao 18", 18564, 185_640_000),
    ]
    power = {o.kind: o.cost for o in bao_catalog(POWER_655)}
    assert power["Bao 5"] == 500_000 and power["Bao 18"] == 185_640_000
    lotto = {o.kind: o.plays for o in bao_catalog(LOTTO_535)}
    assert lotto["Bao 4"] == 31 and lotto["Bao 6"] == 6 and lotto["Bao 15"] == comb(15, 5)
    assert lotto["Bao 2 số đặc biệt"] == 2 and lotto["Bao 12 số đặc biệt"] == 12
    assert all(o.documented for o in bao_catalog(LOTTO_535))


def test_undocumented_options_are_flagged_or_rejected() -> None:
    r = analyse_bao(MEGA_645, list(range(1, 17)))
    assert not r.documented and "Bao 16" in r.notes[0]
    with pytest.raises(ValueError):
        analyse_bao(MEGA_645, list(range(1, 17)), strict=True)
    r = analyse_bao(LOTTO_535, [1, 2, 3, 4, 5, 6], [1, 2])
    assert r.plays == 12 and not r.documented
    assert analyse_bao(LOTTO_535, [1, 2, 3, 4, 5], [1, 2, 3]).documented  # bao số đặc biệt


# ------------------------------------------------------------- published examples
def test_power_bao5_and_bao7_match_published_tables() -> None:
    b5 = _styles(POWER_655, 5)
    assert b5[(2, "not_in_fixed")] == "200.000"
    assert b5[(3, "not_in_fixed")] == "3.850.000"
    assert b5[(4, "in_fixed")] == "Jackpot 2 (×2 phần) + 24.000.000"  # published: "Jackpot 2 + 24.000.000"
    assert b5[(5, "not_in_fixed")] == "Jackpot 1 + Jackpot 2 + 1.920.000.000"
    b7 = _styles(POWER_655, 7)
    assert b7[(3, "not_in_set")] == "200.000"
    assert b7[(4, "not_in_set")] == "1.700.000"
    assert b7[(5, "not_in_set")] == "82.500.000"
    assert b7[(5, "in_set")] == "Jackpot 2 + 42.500.000"
    assert b7[(6, "not_in_set")] == "Jackpot 1 + 240.000.000"
    assert b7[(6, "in_set")] == "Jackpot 1 + Jackpot 2 (×6 phần)"


def test_mega_bao_tables() -> None:
    b5 = _styles(MEGA_645, 5)
    assert b5[(2, "-")] == "120.000"  # published Bao 5 example
    b7 = _styles(MEGA_645, 7)
    assert b7[(3, "-")] == "120.000" and b7[(4, "-")] == "1.020.000" and b7[(5, "-")] == "21.500.000"
    assert b7[(6, "-")] == "Jackpot + 60.000.000"


def test_lotto_bao4_returns_the_stake_whenever_the_special_hits() -> None:
    r = analyse_bao(LOTTO_535, [3, 8, 13, 21], [7])
    matched = [o for o in r.outcomes if o.bonus == "matched"]
    assert all(o.payout_gross >= r.cost for o in matched)
    assert sum(o.probability for o in matched) == pytest.approx(1 / 12)


# ------------------------------------------------------------- rules that matter
def test_own_jackpot_plays_share_one_pot() -> None:
    plays = {"jackpot1": 1, "jackpot2": 6}
    jp = {"jackpot1": 50e9, "jackpot2": 5e9}
    _, gross, _ = ticket_payout(POWER_655, plays, jp)
    assert gross == pytest.approx(55e9)  # not 50 + 6 × 5 tỷ
    _, gross, _ = ticket_payout(POWER_655, plays, jp, co_winners={"jackpot2": 6})
    assert gross == pytest.approx(50e9 + 5e9 * 6 / 12)
    assert expected_share(6, 0.0) == 1.0 and expected_share(1, 2.0) == pytest.approx((1 - np.exp(-2)) / 2, rel=1e-6)


def test_tax_basis_ticket_vs_play() -> None:
    plays = {"first": 2, "second": 5}  # Mega Bao 7 with 5 hits: 21.5 tr
    _, gross, net_ticket = ticket_payout(MEGA_645, plays, {}, tax=DEFAULT_TAX, tax_basis="ticket")
    _, _, net_play = ticket_payout(MEGA_645, plays, {}, tax=DEFAULT_TAX, tax_basis="play")
    assert gross == 21_500_000
    assert net_ticket == pytest.approx(gross - DEFAULT_TAX.rate * (gross - DEFAULT_TAX.threshold))
    assert net_play == gross  # every single prize is below the threshold


def test_bao_ev_reduces_to_exact_without_other_players() -> None:
    jp = {"jackpot1": 60e9, "jackpot2": 5e9}
    nums = [4, 15, 26, 37, 48, 50, 53]
    exact = analyse_bao(POWER_655, nums, None, jp)
    ev0 = bao_ev(POWER_655, nums, None, jp, tickets_sold=1)
    assert ev0.expected_payout_net == pytest.approx(exact.expected_payout_net, rel=1e-4)
    crowded = bao_ev(POWER_655, nums, None, jp, tickets_sold=5_000_000)
    assert crowded.expected_payout_net < ev0.expected_payout_net
    assert crowded.fixed_part == pytest.approx(ev0.fixed_part, rel=1e-9)


def test_simulation_matches_exact_bao() -> None:
    nums = [3, 9, 14, 22, 31, 38, 41, 44]
    exact = analyse_bao(MEGA_645, nums)
    plays = bao_tickets(MEGA_645, nums)
    pay = simulate_plays(MEGA_645, [p[0] for p in plays], None, 150_000, 3, {"jackpot1": 0.0})["payout"]
    se = np.sqrt(exact.p_any_prize * (1 - exact.p_any_prize) / 150_000)
    assert abs((pay > 0).mean() - exact.p_any_prize) < 4.5 * se


def test_compare_strategies_shapes() -> None:
    c = compare_bao_strategies(MEGA_645, [3, 9, 14, 22, 31, 38, 41, 44], sims=20_000, seed=1)
    rows = {r.strategy: r for r in c.rows}
    bao, wheel = rows["Bao 8"], rows["Bao rút gọn (3 nếu về 3)"]
    spread = rows["Vé lẻ dàn đều (cùng ngân sách)"]
    # a 3-if-3 wheel wins exactly when the bao wins (both need ≥ 3 drawn numbers in the set)
    assert wheel.p_any_prize == pytest.approx(bao.p_any_prize) and wheel.cost < bao.cost
    assert spread.p_any_prize > 5 * bao.p_any_prize
    assert bao.p_profit > spread.p_profit  # bao: rarely wins, but a win usually covers the stake


# ------------------------------------------------------------------ Max 3D family
def test_max3d_published_figures() -> None:
    pro = max3d.product_summary(max3d.MAX3D_PRO)
    assert pro.rtp_distinct_numbers == pytest.approx(0.55, abs=0.01)  # "tỷ lệ trả thưởng lên đến 55%"
    assert pro.p_any_prize == pytest.approx(0.04, abs=0.002)  # "tỷ lệ có giải lên tới 4%"
    plus = max3d.product_summary(max3d.MAX3D_PLUS)
    assert plus.top_prize_odds == pytest.approx(500_000, rel=1e-3)  # "xác suất 1/500.000"
    basic = max3d.product_summary(max3d.MAX3D)
    assert basic.rtp_distinct_numbers == pytest.approx(0.545, abs=0.005)
    assert max3d.play_distribution(max3d.MAX3D_PLUS, (333, 333)).return_to_player == pytest.approx(plus.rtp_distinct_numbers, abs=0.01)


def test_max3d_bao_counts() -> None:
    pro = max3d.MAX3D_PRO
    assert len(max3d.bao_plays(pro, "bao_bo_so", ["123", "456"])) == 36
    assert len(max3d.bao_plays(pro, "bao_bo_so", ["112", "456"])) == 18
    assert len(max3d.bao_plays(pro, "bao_bo_so", ["111", "456"])) == 6
    assert len(max3d.bao_plays(pro, "bao_nhieu_bo_so", ["123", "456", "789"])) == 6
    assert len(max3d.bao_plays(pro, "bao_nhieu_bo_so", [f"{i:03d}" for i in range(20)])) == 380
    assert len(max3d.bao_plays(max3d.MAX3D, "bao_vi_tri", ["1*3"])) == 10
    assert len(max3d.bao_plays(max3d.MAX3D, "bao_vi_tri", ["**3"])) == 100
    with pytest.raises(ValueError):
        max3d.bao_plays(pro, "bao_nhieu_bo_so", ["123", "456"])
    with pytest.raises(ValueError):
        max3d.parse_number("1000")


def test_max3d_exact_vs_simulation() -> None:
    pro = max3d.MAX3D_PRO
    d = max3d.play_distribution(pro, (123, 456))
    pay = max3d.simulate_plays(pro, [(123, 456)], 300_000, 5)
    assert (pay > 0).mean() == pytest.approx(d.p_any_prize, abs=0.002)
    low = {"nam": 100_000, "sau": 40_000}
    ev_low = sum(d.tier_probabilities[t] * v for t, v in low.items())
    sim_low = np.clip(pay, 0, 140_000).mean()  # low tiers only (higher tiers are rarer than 1/2,500)
    assert sim_low == pytest.approx(ev_low, rel=0.05)
    r = max3d.analyse_bao(pro, "bao_bo_so", ["123", "456"], sims=50_000)
    assert r.p_any_prize == pytest.approx(1 - (1 - 12 / 1000) ** 20)  # 6 + 6 distinct numbers
    assert r.return_to_player == pytest.approx(d.return_to_player)
    assert r.p_top_prize == pytest.approx(36 / 1_000_000)
