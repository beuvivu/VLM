"""Keno and Bingo18: prize tables, exact odds and return to player.

Keno — 20 different numbers from 1–80 per draw. "Bậc b": choose b numbers (1–10); the
prize depends on how many of them are drawn. Side bets on the 20 drawn numbers: Lớn/Nhỏ
(41–80 vs 1–40) and Chẵn/Lẻ. Stake 10,000 đ (×2 … ×50). Bậc-10 top prize capped at 10 tỷ
per draw in total.

Bingo18 — three numbers 1–6, drawn independently (with repetition), every ~6 minutes.
Bets: one number (paid by how many times it appears), a double, a specific triple, any
triple, the sum (3–18), Lớn (12–18) / Hòa (10–11) / Nhỏ (3–9).

Keno prize cells (v3.3) are those of the vietlott.vn Keno detail page (draw #0284640,
13/06/2026, as stored by the community archive NhanAZ-Data/vietlott-research —
``data/seed/keno_rules_official.json``) and agree with the product page used by Vietlott
Quant Engine 1.3.0. Values printed by agent / press sites (VTC Pay 2020, xosovip 2025,
xoso.mobi) are kept as variants with their exact return to player, so outdated or mis-copied
tables stand out. One cell is unresolved: bậc 5 / 4 hits is advertised at 150.000 đ while the
2026 detail pages show "0 đ" — 150.000 is used, the other value is a variant.
Bingo18: xskt, onbit and the same detail pages.
"""

from __future__ import annotations

from itertools import product
from math import comb

from pydantic import BaseModel

KENO_POOL, KENO_DRAWN, KENO_PRICE = 80, 20, 10_000

# prize per 10,000 đ for (bậc → {matches: prize}) — VTC Pay (Vietlott distribution partner)
KENO_TABLE: dict[int, dict[int, int]] = {
    1: {1: 20_000},
    2: {2: 90_000},
    3: {2: 20_000, 3: 200_000},
    4: {2: 10_000, 3: 50_000, 4: 400_000},
    5: {3: 10_000, 4: 150_000, 5: 4_400_000},
    6: {3: 10_000, 4: 40_000, 5: 450_000, 6: 12_500_000},
    7: {3: 10_000, 4: 20_000, 5: 100_000, 6: 1_200_000, 7: 40_000_000},
    8: {0: 10_000, 4: 10_000, 5: 50_000, 6: 500_000, 7: 5_000_000, 8: 200_000_000},
    9: {0: 10_000, 4: 10_000, 5: 30_000, 6: 150_000, 7: 1_500_000, 8: 12_000_000, 9: 800_000_000},
    10: {0: 10_000, 5: 20_000, 6: 80_000, 7: 710_000, 8: 8_000_000, 9: 150_000_000, 10: 2_000_000_000},
}
# values other sites print (RTP of each variant is reported so the outlier stands out)
KENO_TABLE_VARIANTS: dict[int, dict[str, dict[int, int]]] = {
    5: {"xoso.mobi, xosovip (4 số = 50.000)": {4: 50_000}, "trang chi tiết kỳ quay 2026 ghi 4 số = 0 đ": {4: 0}},
    8: {"xosovip (4 số = 20.000)": {4: 20_000}},
    10: {"VTC Pay 2020, bảng cũ (7 số = 600.000, 8 số = 7.400.000)": {7: 600_000, 8: 7_400_000}},
}


class SideBet(BaseModel):
    name: str
    condition: str
    tiers: list[tuple[int, int, int | None]]  # (min count, max count, payout) among the 20 drawn numbers
    alternatives: dict[str, list[tuple[int, int, int]]] = {}  # other published tables (label → tiers)
    verified: bool
    source: str


# Lớn/Nhỏ: numbers 41–80 vs 1–40; Chẵn/Lẻ: 40 even, 40 odd → the same hypergeometric law.
# Payouts: vietlott.vn Keno detail page (draw #0284640, 13/06/2026); older printed tables kept
# as alternatives.
OFFICIAL_KENO_PAGE = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0284640"
KENO_SIDE_BETS: list[SideBet] = [
    SideBet(
        name="Lớn (hoặc Nhỏ)",
        condition="13–20 / 11–12 trong 20 số thuộc 41–80 (1–40)",
        tiers=[(11, 12, 10_000), (13, 20, 26_000)],
        alternatives={"xosovip 2025: chỉ ≥13 = 55.000": [(13, 20, 55_000)], "VTC Pay 2020: chỉ ≥13 = 56.000": [(13, 20, 56_000)]},
        verified=True,
        source="vietlott.vn, trang chi tiết Keno #0284640: ≥13 số = 26.000, 11–12 số = 10.000",
    ),
    SideBet(name="Hòa Lớn–Nhỏ", condition="10 số mỗi bên", tiers=[(10, 10, 26_000)], verified=True, source="vietlott.vn #0284640; VnExpress, VietnamNet (05/04/2021)"),
    SideBet(
        name="Chẵn (hoặc Lẻ)",
        condition="13–14 / ≥ 15 số chẵn (lẻ)",
        tiers=[(13, 14, 40_000), (15, 20, 200_000)],
        alternatives={"xosovip 2025: 30.000 / 300.000": [(13, 14, 30_000), (15, 20, 300_000)], "VTC Pay 2020: 40.000 / 210.000": [(13, 14, 40_000), (15, 20, 210_000)]},
        verified=True,
        source="vietlott.vn, trang chi tiết Keno #0284640: 13–14 số = 40.000, ≥15 số = 200.000",
    ),
    SideBet(name="Chẵn 11–12 (Lẻ 11–12)", condition="11–12 số chẵn (lẻ)", tiers=[(11, 12, 20_000)], verified=True, source="vietlott.vn #0284640: 20.000"),
    SideBet(name="Hòa Chẵn–Lẻ", condition="10 chẵn, 10 lẻ", tiers=[(10, 10, 20_000)], verified=True, source="vietlott.vn #0284640: 20.000"),
]


def keno_hits_distribution(bac: int) -> dict[int, float]:
    """P(m of the b chosen numbers are among the 20 drawn), hypergeometric."""
    return {m: comb(KENO_DRAWN, m) * comb(KENO_POOL - KENO_DRAWN, bac - m) / comb(KENO_POOL, bac) for m in range(bac + 1)}


def keno_split_distribution() -> dict[int, float]:
    """P(x of the 20 drawn numbers are 'large' (or even)), x = 0…20."""
    return {x: comb(40, x) * comb(40, KENO_DRAWN - x) / comb(KENO_POOL, KENO_DRAWN) for x in range(KENO_DRAWN + 1)}


class KenoBacOdds(BaseModel):
    bac: int
    prizes: dict[int, int]
    probabilities: dict[int, float]
    p_any_prize: float
    one_in_top_prize: float
    return_to_player: float
    variants: dict[str, float] = {}


class SideBetOdds(BaseModel):
    name: str
    condition: str
    probability: float  # P(any payout)
    payouts: list[int | None]
    return_to_player: float | None
    alternatives_rtp: dict[str, float]
    verified: bool
    source: str


def keno_odds() -> tuple[list[KenoBacOdds], list[SideBetOdds]]:
    bacs = []
    for b, table in KENO_TABLE.items():
        dist = keno_hits_distribution(b)
        rtp = sum(dist[m] * v for m, v in table.items()) / KENO_PRICE
        variants = {}
        for label, change in KENO_TABLE_VARIANTS.get(b, {}).items():
            alt = table | change
            variants[label] = sum(dist[x] * y for x, y in alt.items()) / KENO_PRICE
        bacs.append(
            KenoBacOdds(
                bac=b,
                prizes=table,
                probabilities={m: dist[m] for m in table},
                p_any_prize=sum(dist[m] for m in table),
                one_in_top_prize=1 / dist[b],
                return_to_player=rtp,
                variants=variants,
            )
        )
    split = keno_split_distribution()
    sides = []
    def rtp_of(tiers: list) -> float | None:  # type: ignore[type-arg]
        if any(pay is None for _, _, pay in tiers):
            return None
        return sum(sum(split[x] for x in range(lo, hi + 1)) * pay for lo, hi, pay in tiers) / KENO_PRICE

    for sb in KENO_SIDE_BETS:
        probs = [sum(split[x] for x in range(lo, hi + 1)) for lo, hi, _ in sb.tiers]
        pays = [pay for _, _, pay in sb.tiers]
        rtp = rtp_of(sb.tiers)
        alts = {label: rtp_of(t) for label, t in sb.alternatives.items()}
        sides.append(SideBetOdds(name=sb.name, condition=sb.condition, probability=sum(probs), payouts=pays, return_to_player=rtp, alternatives_rtp=alts, verified=sb.verified, source=sb.source))
    return bacs, sides


# ------------------------------------------------------------------ Bingo18
BINGO_PRICE = 10_000
BINGO_SUM_PAYOUT = {3: 1_200_000, 18: 1_200_000, 4: 400_000, 17: 400_000, 5: 200_000, 16: 200_000, 6: 120_000, 15: 120_000, 7: 80_000, 14: 80_000, 8: 55_000, 13: 55_000, 9: 47_000, 12: 47_000, 10: 44_000, 11: 44_000}


class BingoBetOdds(BaseModel):
    bet: str
    condition: str
    probability: float  # P(any prize)
    payout: str
    return_to_player: float


def _outcomes() -> list[tuple[int, int, int]]:
    return list(product(range(1, 7), repeat=3))


def bingo18_odds() -> list[BingoBetOdds]:
    """Exact odds of every Bingo18 bet (216 equally likely ordered outcomes)."""
    outs = _outcomes()
    n = len(outs)
    rows = []
    # one number, paid by multiplicity (12k / 20k / 30k)
    pay_single = {1: 12_000, 2: 20_000, 3: 30_000}
    ev = sum(pay_single.get(o.count(1), 0) for o in outs) / n
    rows.append(BingoBetOdds(bet="Một số (vd. 1)", condition="số chọn xuất hiện 1 / 2 / 3 lần", probability=sum(o.count(1) > 0 for o in outs) / n, payout="12.000 / 20.000 / 30.000", return_to_player=ev / BINGO_PRICE))
    p2 = sum(o.count(1) >= 2 for o in outs) / n
    rows.append(BingoBetOdds(bet="Bộ đôi (vd. 1-1)", condition="số chọn xuất hiện ≥ 2 lần", probability=p2, payout="75.000", return_to_player=p2 * 75_000 / BINGO_PRICE))
    p3 = 1 / n
    rows.append(BingoBetOdds(bet="Bộ ba cụ thể (vd. 1-1-1)", condition="ba số giống nhau đúng số chọn", probability=p3, payout="1.200.000", return_to_player=p3 * 1_200_000 / BINGO_PRICE))
    pany3 = 6 / n
    rows.append(BingoBetOdds(bet="Bộ ba bất kỳ", condition="ba số giống nhau", probability=pany3, payout="200.000", return_to_player=pany3 * 200_000 / BINGO_PRICE))
    for s in range(3, 19):
        p = sum(sum(o) == s for o in outs) / n
        rows.append(BingoBetOdds(bet=f"Tổng {s}", condition=f"tổng ba số = {s}", probability=p, payout=f"{BINGO_SUM_PAYOUT[s]:,}".replace(",", "."), return_to_player=p * BINGO_SUM_PAYOUT[s] / BINGO_PRICE))
    for name, lo, hi, pay in (("Nhỏ", 3, 9, 15_000), ("Hòa", 10, 11, 20_000), ("Lớn", 12, 18, 15_000)):
        p = sum(lo <= sum(o) <= hi for o in outs) / n
        rows.append(BingoBetOdds(bet=name, condition=f"tổng {lo}–{hi}", probability=p, payout=f"{pay:,}".replace(",", "."), return_to_player=p * pay / BINGO_PRICE))
    return rows


def bingo18_sum_distribution() -> dict[int, float]:
    outs = _outcomes()
    return {s: sum(sum(o) == s for o in outs) / len(outs) for s in range(3, 19)}
