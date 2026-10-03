"""Max 3D, Max 3D+ and Max 3D Pro: exact prize mathematics and "bao" (system) plays.

Draw (all three products): 20 three-digit numbers 000–999, drawn independently, in four
groups — 2 + 4 + 6 + 8. Max 3D / Max 3D+ call them giải Nhất / Nhì / Ba / Khuyến khích;
Max 3D Pro calls them Đặc biệt / Nhất / Nhì / Ba. A ticket wins **every** tier whose
condition it meets (prizes are cumulative). One play costs 10,000 đ; prizes scale with
the stake.

Prize tables (per 10,000 đ; sources in README §Chơi bao):

Max 3D (one number)            Max 3D+ (two numbers)                  Max 3D Pro (two numbers)
 Nhất  in the 2 Nhất  1,000,000  Nhất  = both Nhất (any order) 1 tỷ     ĐB    = both ĐB, in order    2 tỷ
 Nhì   in the 4 Nhì     350,000  Nhì   both in 4 Nhì          40 tr     Phụ ĐB both ĐB, reversed   400 tr
 Ba    in the 6 Ba      210,000  Ba    both in 6 Ba           10 tr     Nhất  both in 4 Nhất       30 tr
 KK    in the 8 KK      100,000  Tư    both in 8 KK            5 tr     Nhì   both in 6 Nhì        10 tr
                                 Năm   both among all 20       1 tr     Ba    both in 8 Ba          4 tr
                                 Sáu   one in the 2 Nhất     150,000    Tư    both among all 20     1 tr
                                 Bảy   one in the other 18    40,000    Năm   one in the 2 ĐB     100,000
                                                                        Sáu   one in the other 18  40,000

Checks against published figures (tests): Max 3D Pro pays back ≈ 55 % and ≈ 4 % of plays
win something; Max 3D+ first prize has probability 1/500,000.

Bao (system) plays — each generated play costs one stake:
* Max 3D Pro "bao bộ ba số": every digit permutation of the first number × every
  permutation of the second (123-456 → 36 plays).
* Max 3D Pro "bao nhiều bộ ba số": 3–20 numbers → every ordered pair of two different
  numbers, n(n−1) plays (3 → 6, 20 → 380).
* Max 3D / Max 3D+ "đảo số" (digit permutations) and "bao vị trí" (a wildcard digit
  ``*`` → 10 plays, two wildcards → 100) — documented by resellers, not by a Vietlott page
  we could read; flagged in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Literal

import numpy as np
from pydantic import BaseModel

Condition = Literal["one", "both", "ordered", "reversed", "pair_exact"]
GROUPS = ("D", "G1", "G2", "G3")
GROUP_SIZE = {"D": 2, "G1": 4, "G2": 6, "G3": 8}
ALL = GROUPS
P_NUM = 1 / 1000


@dataclass(frozen=True)
class Max3DTier:
    name: str
    condition: Condition
    groups: tuple[str, ...]
    value: int
    rule: str


@dataclass(frozen=True)
class Max3DProduct:
    code: str
    display_name: str
    numbers_per_play: int
    group_labels: dict[str, str]
    tiers: tuple[Max3DTier, ...]
    price: int = 10_000
    identical_pair_multiplier: float = 1.0
    top_prize_cap: int | None = None  # total payable for the top tier per draw
    draw_weekdays: tuple[int, ...] = ()
    bao_types: tuple[str, ...] = ()
    bao_documented: bool = True
    max_stake_multiple: int = 20


MAX3D = Max3DProduct(
    code="max3d",
    display_name="Max 3D",
    numbers_per_play=1,
    group_labels={"D": "Nhất", "G1": "Nhì", "G2": "Ba", "G3": "Khuyến khích"},
    tiers=(
        Max3DTier("nhat", "one", ("D",), 1_000_000, "trùng 1 trong 2 số giải Nhất"),
        Max3DTier("nhi", "one", ("G1",), 350_000, "trùng 1 trong 4 số giải Nhì"),
        Max3DTier("ba", "one", ("G2",), 210_000, "trùng 1 trong 6 số giải Ba"),
        Max3DTier("khuyen_khich", "one", ("G3",), 100_000, "trùng 1 trong 8 số giải Khuyến khích"),
    ),
    draw_weekdays=(0, 2, 4),
    bao_types=("dao_so", "bao_vi_tri"),
    bao_documented=False,
)

MAX3D_PLUS = Max3DProduct(
    code="max3dplus",
    display_name="Max 3D+",
    numbers_per_play=2,
    group_labels=MAX3D.group_labels,
    tiers=(
        Max3DTier("nhat", "pair_exact", ("D",), 1_000_000_000, "trùng 2 số giải Nhất"),
        Max3DTier("nhi", "both", ("G1",), 40_000_000, "trùng 2 trong 4 số giải Nhì"),
        Max3DTier("ba", "both", ("G2",), 10_000_000, "trùng 2 trong 6 số giải Ba"),
        Max3DTier("tu", "both", ("G3",), 5_000_000, "trùng 2 trong 8 số giải Khuyến khích"),
        Max3DTier("nam", "both", ALL, 1_000_000, "trùng 2 số bất kỳ trong 20 số"),
        Max3DTier("sau", "one", ("D",), 150_000, "trùng 1 trong 2 số giải Nhất"),
        Max3DTier("bay", "one", ("G1", "G2", "G3"), 40_000, "trùng 1 số giải Nhì, Ba hoặc Khuyến khích"),
    ),
    identical_pair_multiplier=2.0,  # "chọn hai số giống nhau, giải thưởng gấp đôi"
    draw_weekdays=(0, 2, 4),
    bao_types=("dao_so", "bao_vi_tri"),
    bao_documented=False,
)

MAX3D_PRO = Max3DProduct(
    code="max3dpro",
    display_name="Max 3D Pro",
    numbers_per_play=2,
    group_labels={"D": "Đặc biệt", "G1": "Nhất", "G2": "Nhì", "G3": "Ba"},
    tiers=(
        Max3DTier("dac_biet", "ordered", ("D",), 2_000_000_000, "trùng 2 số giải Đặc biệt theo đúng thứ tự"),
        Max3DTier("phu_dac_biet", "reversed", ("D",), 400_000_000, "trùng 2 số giải Đặc biệt ngược thứ tự"),
        Max3DTier("nhat", "both", ("G1",), 30_000_000, "trùng 2 trong 4 số giải Nhất"),
        Max3DTier("nhi", "both", ("G2",), 10_000_000, "trùng 2 trong 6 số giải Nhì"),
        Max3DTier("ba", "both", ("G3",), 4_000_000, "trùng 2 trong 8 số giải Ba"),
        Max3DTier("tu", "both", ALL, 1_000_000, "trùng 2 số bất kỳ trong 20 số"),
        Max3DTier("nam", "one", ("D",), 100_000, "trùng 1 trong 2 số giải Đặc biệt"),
        Max3DTier("sau", "one", ("G1", "G2", "G3"), 40_000, "trùng 1 trong 18 số giải Nhất, Nhì, Ba"),
    ),
    top_prize_cap=30_000_000_000,
    draw_weekdays=(1, 3, 5),
    bao_types=("bao_bo_so", "bao_nhieu_bo_so"),
)

PRODUCTS: dict[str, Max3DProduct] = {p.code: p for p in (MAX3D, MAX3D_PLUS, MAX3D_PRO)}


def get_product(code: str) -> Max3DProduct:
    key = code.lower().replace(" ", "").replace("+", "plus").replace("_", "")
    aliases = {"3d": "max3d", "max3d": "max3d", "3dplus": "max3dplus", "max3dplus": "max3dplus", "3dpro": "max3dpro", "max3dpro": "max3dpro", "pro": "max3dpro"}
    if key not in aliases:
        raise KeyError(f"unknown Max 3D product {code!r}; expected one of {list(PRODUCTS)}")
    return PRODUCTS[aliases[key]]


# ------------------------------------------------------------------ plays & bao
def parse_number(x: int | str) -> int:
    v = int(str(x))
    if not 0 <= v <= 999:
        raise ValueError(f"{x!r} is not a three-digit number 000–999")
    return v


def digit_permutations(x: int | str) -> list[int]:
    """Distinct numbers obtained by permuting the three digits (123 → 6, 112 → 3, 111 → 1)."""
    s = f"{parse_number(x):03d}"
    return sorted({int("".join(p)) for p in permutations(s)})


def expand_pattern(pattern: str) -> list[int]:
    """'1*3' → 103, 113, …, 193 (``*`` = any digit)."""
    pat = str(pattern).strip()
    if len(pat) != 3 or any(c not in "0123456789*" for c in pat):
        raise ValueError(f"pattern {pattern!r} must be three characters of digits or '*'")
    slots = [list("0123456789") if c == "*" else [c] for c in pat]
    return [int("".join(p)) for p in product(*slots)]


Play = tuple[int, ...]


def bao_plays(prod: Max3DProduct, kind: str, numbers: list[str | int]) -> list[Play]:
    """Expand a bao into individual plays.

    * ``bao_bo_so`` / ``dao_so``: digit permutations of each number (pair products: product).
    * ``bao_nhieu_bo_so`` (Pro): ordered pairs of two different numbers from 3–20 numbers.
    * ``bao_vi_tri``: numbers may contain ``*`` wildcards (10 / 100 variants each).
    """
    k = prod.numbers_per_play
    if kind in ("bao_bo_so", "dao_so"):
        if len(numbers) != k:
            raise ValueError(f"{prod.display_name} {kind} needs {k} number(s)")
        sets = [digit_permutations(x) for x in numbers]
        return [tuple(p) for p in product(*sets)]
    if kind == "bao_vi_tri":
        if len(numbers) != k:
            raise ValueError(f"{prod.display_name} bao vị trí needs {k} pattern(s)")
        sets = [expand_pattern(str(x)) for x in numbers]
        return [tuple(p) for p in product(*sets)]
    if kind == "bao_nhieu_bo_so":
        if k != 2:
            raise ValueError("bao nhiều bộ số is a two-number play")
        nums = sorted({parse_number(x) for x in numbers})
        if not 3 <= len(nums) <= 20 or len(nums) != len(numbers):
            raise ValueError("bao nhiều bộ số takes 3–20 different numbers")
        return [(a, b) for a in nums for b in nums if a != b]
    if kind == "co_ban":
        if len(numbers) != k:
            raise ValueError(f"{prod.display_name} needs {k} number(s) per play")
        return [tuple(parse_number(x) for x in numbers)]
    raise ValueError(f"unknown bao kind {kind!r}")


# ------------------------------------------------------------------ exact single play
def _group_states(n: int, same: bool) -> list[tuple[int, int, float]]:
    """Full occurrence counts and their probabilities for n independent draws."""
    p = P_NUM
    out: dict[tuple[int, int], float] = {}
    if same:
        for a in range(n + 1):
            pr = factorial(n) / (factorial(a) * factorial(n - a)) * p**a * (1 - p) ** (n - a)
            key = (a, a)
            out[key] = out.get(key, 0.0) + pr
    else:
        for a in range(n + 1):
            for b in range(n + 1 - a):
                pr = factorial(n) / (factorial(a) * factorial(b) * factorial(n - a - b)) * p ** (a + b) * (1 - 2 * p) ** (n - a - b)
                key = (a, b)
                out[key] = out.get(key, 0.0) + pr
    return [(a, b, pr) for (a, b), pr in out.items()]


def _d_states(same: bool) -> list[tuple[tuple[str, str], float]]:
    """Ordered outcomes of the two D draws over symbols x, y, o (other)."""
    p = P_NUM
    sym = {"x": p, "o": 1 - p} if same else {"x": p, "y": p, "o": 1 - 2 * p}
    return [((a, b), sym[a] * sym[b]) for a in sym for b in sym]


def _tier_won(t: Max3DTier, d: tuple[str, str], cx: dict[str, int], cy: dict[str, int], single: bool, same: bool) -> bool:
    if t.condition == "one":
        return any(cx[g] > 0 or (not single and cy[g] > 0) for g in t.groups)
    tx = sum(cx[g] for g in t.groups)
    ty = sum(cy[g] for g in t.groups)
    if t.condition == "both":
        return tx >= 2 if same else (tx >= 1 and ty >= 1)
    if t.condition == "ordered":
        return d == ("x", "x") if same else d == ("x", "y")
    if t.condition == "reversed":
        return (not same) and d == ("y", "x")
    if t.condition == "pair_exact":
        return d == ("x", "x") if same else d in (("x", "y"), ("y", "x"))
    raise ValueError(t.condition)


def _tier_awards(prod: Max3DProduct, t: Max3DTier, d: tuple[str, str], cx: dict[str, int],
                 cy: dict[str, int], single: bool, same: bool) -> int:
    """Quantity won: each matching constituent number pays its single-number tier."""
    if not _tier_won(t, d, cx, cy, single, same):
        return 0
    if t.condition != "one":
        return 1
    awards = sum(cx[g] for g in t.groups)
    if not single and not same:
        awards += sum(cy[g] for g in t.groups)
    elif not single and prod.identical_pair_multiplier == 1:
        awards *= 2
    return awards


class PlayDistribution(BaseModel):
    product: str
    play: list[str]
    p_any_prize: float
    expected_payout: float
    return_to_player: float
    tier_probabilities: dict[str, float]  # P(tier won), tiers are cumulative
    distribution: list[tuple[float, float]]  # (payout, probability), payout > 0


def play_distribution(prod: Max3DProduct, play: Play | list[int | str], stake_multiple: int = 1) -> PlayDistribution:
    """Exact prize distribution of one play (all four groups, cumulative tiers)."""
    nums = tuple(parse_number(x) for x in play)
    if len(nums) != prod.numbers_per_play:
        raise ValueError(f"{prod.display_name} plays have {prod.numbers_per_play} number(s)")
    single = prod.numbers_per_play == 1
    same = single or nums[0] == nums[1]
    mult = stake_multiple * (prod.identical_pair_multiplier if (not single and same) else 1.0)
    g_states = {g: _group_states(GROUP_SIZE[g], same) for g in ("G1", "G2", "G3")}
    dist: dict[float, float] = {}
    tier_p = {t.name: 0.0 for t in prod.tiers}
    for d, pd in _d_states(same):
        cxd = sum(1 for s in d if s == "x")
        cyd = cxd if same else sum(1 for s in d if s == "y")
        for (a1, b1, p1) in g_states["G1"]:
            for (a2, b2, p2) in g_states["G2"]:
                for (a3, b3, p3) in g_states["G3"]:
                    pr = pd * p1 * p2 * p3
                    cx = {"D": cxd, "G1": a1, "G2": a2, "G3": a3}
                    cy = {"D": cyd, "G1": b1, "G2": b2, "G3": b3}
                    pay = 0.0
                    for t in prod.tiers:
                        awards = _tier_awards(prod, t, d, cx, cy, single, same)
                        if awards:
                            pay += t.value * mult * awards
                            tier_p[t.name] += pr
                    dist[pay] = dist.get(pay, 0.0) + pr
    p_any = float(sum(p for v, p in dist.items() if v > 0))
    ev = float(sum(v * p for v, p in dist.items()))
    cost = prod.price * stake_multiple
    return PlayDistribution(
        product=prod.code,
        play=[f"{x:03d}" for x in nums],
        p_any_prize=p_any,
        expected_payout=ev,
        return_to_player=ev / cost,
        tier_probabilities=tier_p,
        distribution=sorted(((v, p) for v, p in dist.items() if v > 0), reverse=True),
    )


# ------------------------------------------------------------------ bao analysis
class Max3DBaoAnalysis(BaseModel):
    product: str
    kind: str
    numbers: list[str]
    plays: int
    cost: int
    documented: bool
    p_any_prize: float  # exact: some chosen number appears among the 20
    p_any_prize_same_budget_random: float  # same number of independent random plays
    expected_payout: float  # exact, by linearity
    return_to_player: float
    p_profit: float  # Monte Carlo
    p_top_prize: float  # exact: P(at least one play wins the top tier)
    payout_quantiles: dict[str, float]  # Monte Carlo
    simulations: int
    sample_plays: list[list[str]]
    notes: list[str] = []


def simulate_draws(sims: int, rng: np.random.Generator) -> np.ndarray:
    """(sims, 20) numbers 0–999: columns 0–1 D, 2–5 G1, 6–11 G2, 12–19 G3."""
    return rng.integers(0, 1000, (sims, 20))


_SLICES = {"D": slice(0, 2), "G1": slice(2, 6), "G2": slice(6, 12), "G3": slice(12, 20)}


def simulate_plays(prod: Max3DProduct, plays: list[Play], sims: int = 200_000, seed: int | None = 0, stake_multiple: int = 1) -> np.ndarray:
    """Monte Carlo total payout per draw of a set of plays (cumulative tiers, no cap)."""
    rng = np.random.default_rng(seed)
    uniq = sorted({x for p in plays for x in p})
    pos = {u: i for i, u in enumerate(uniq)}
    single = prod.numbers_per_play == 1
    a_num = np.array([p[0] for p in plays])
    b_num = np.array([p[-1] for p in plays])
    xi = np.array([pos[x] for x in a_num])
    yi = np.array([pos[x] for x in b_num])
    same = np.ones(len(plays), bool) if single else a_num == b_num
    mult = stake_multiple * np.where(same & (not single), prod.identical_pair_multiplier, 1.0)
    out = np.empty(sims)
    chunk = max(500, min(50_000, 2_000_000 // max(len(plays), len(uniq), 1)))
    done = 0
    while done < sims:
        c = min(chunk, sims - done)
        dr_all = simulate_draws(c, rng)
        eq = dr_all[:, :, None] == np.asarray(uniq)[None, None, :]  # (c, 20, U)
        rows = eq.any(axis=(1, 2))  # draws where none of the chosen numbers appears pay 0
        out[done : done + c] = 0.0
        dr, eq = dr_all[rows], eq[rows]
        c_all, c = c, int(rows.sum())
        if c == 0:
            done += c_all
            continue
        cnt = {g: eq[:, _SLICES[g], :].sum(axis=1, dtype=np.int8) for g in GROUPS}  # (c, U)
        cx = {g: cnt[g][:, xi] for g in GROUPS}  # (c, plays)
        cy = cx if single else {g: cnt[g][:, yi] for g in GROUPS}
        total = np.zeros(c)
        for t in prod.tiers:
            if t.condition == "one":
                hit = sum(cx[g] for g in t.groups).astype(np.int64)
                if not single:
                    additional = sum(cy[g] for g in t.groups)
                    # Max 3D+ already doubles identical selections through mult.
                    additional = np.where(same[None, :] & (prod.identical_pair_multiplier == 2), 0, additional)
                    hit += additional
            elif t.condition == "both":
                tx = sum(cx[g] for g in t.groups)
                ty = sum(cy[g] for g in t.groups)
                hit = np.where(same[None, :], tx >= 2, (tx >= 1) & (ty >= 1))
            else:
                fwd = (dr[:, 0:1] == a_num[None, :]) & (dr[:, 1:2] == b_num[None, :])
                rev = (dr[:, 0:1] == b_num[None, :]) & (dr[:, 1:2] == a_num[None, :])
                if t.condition == "ordered":
                    hit = fwd
                elif t.condition == "reversed":
                    hit = rev & ~same[None, :]
                else:  # pair_exact
                    hit = fwd | rev
            total += hit @ (t.value * mult)
        out[done : done + c_all][rows] = total
        done += c_all
    return out


def analyse_bao(prod: Max3DProduct, kind: str, numbers: list[str | int], sims: int = 200_000, seed: int | None = 0, stake_multiple: int = 1) -> Max3DBaoAnalysis:
    plays = bao_plays(prod, kind, numbers)
    n_plays = len(plays)
    cost = n_plays * prod.price * stake_multiple
    uniq = {x for p in plays for x in p}
    p_any = 1 - (1 - len(uniq) / 1000) ** 20
    # every play's distribution depends only on whether its two numbers are equal (symmetry)
    ev = 0.0
    cache: dict[bool, PlayDistribution] = {}
    for p in plays:
        same = prod.numbers_per_play == 1 or p[0] == p[1]
        if same not in cache:
            cache[same] = play_distribution(prod, p, stake_multiple)
        ev += cache[same].expected_payout
    top = prod.tiers[0]
    # top tier: D (or the D pair) is fully determined by the two D draws → exact count
    if top.condition in ("ordered", "pair_exact"):
        winners = {(a, b) for a, b in plays} if prod.numbers_per_play == 2 else set()
        if top.condition == "pair_exact":
            winners |= {(b, a) for a, b in winners}
        p_top = len(winners) / 1_000_000
    else:  # Max 3D: a number among the 2 Nhất numbers
        p_top = 1 - (1 - len(uniq) / 1000) ** 2
    sims = int(min(sims, max(20_000, 20_000_000 // n_plays)))  # keep large bao affordable
    pay = simulate_plays(prod, plays, sims, seed, stake_multiple)
    notes = []
    if kind in prod.bao_types and not prod.bao_documented:
        notes.append(f"cách bao '{kind}' của {prod.display_name} lấy từ hướng dẫn của đại lý, chưa đối chiếu được với trang Vietlott")
    if kind not in prod.bao_types and kind != "co_ban":
        notes.append(f"{prod.display_name} không công bố kiểu bao '{kind}'; phân tích như mua lẻ từng lượt")
    if prod.top_prize_cap:
        notes.append(f"tổng giải {top.name} mỗi kỳ tối đa {prod.top_prize_cap:,} đ (chia đều khi vượt) — không tính trong EV")
    return Max3DBaoAnalysis(
        product=prod.code,
        kind=kind,
        numbers=[str(x) for x in numbers],
        plays=n_plays,
        cost=cost,
        documented=kind == "co_ban" or (kind in prod.bao_types and prod.bao_documented),
        p_any_prize=float(p_any),
        p_any_prize_same_budget_random=float(1 - (1 - _single_p_any(prod)) ** n_plays),
        expected_payout=float(ev),
        return_to_player=float(ev / cost),
        p_profit=float((pay >= cost).mean()),
        p_top_prize=float(p_top),
        payout_quantiles={q: float(np.quantile(pay, float(q))) for q in ("0.5", "0.9", "0.99", "0.999")},
        simulations=sims,
        sample_plays=[[f"{x:03d}" for x in p] for p in plays[:12]],
        notes=notes,
    )


def _single_p_any(prod: Max3DProduct) -> float:
    k = prod.numbers_per_play
    return 1 - (1 - k / 1000) ** 20 if k == 2 else 1 - (1 - 1 / 1000) ** 20


class ProductSummary(BaseModel):
    code: str
    name: str
    numbers_per_play: int
    price: int
    groups: dict[str, str]
    tiers: list[dict]
    bao_types: list[str]
    rtp_distinct_numbers: float
    p_any_prize: float
    top_prize_odds: float


def product_summary(prod: Max3DProduct) -> ProductSummary:
    play = (123,) if prod.numbers_per_play == 1 else (123, 456)
    d = play_distribution(prod, play)
    return ProductSummary(
        code=prod.code,
        name=prod.display_name,
        numbers_per_play=prod.numbers_per_play,
        price=prod.price,
        groups={g: f"{prod.group_labels[g]} ({GROUP_SIZE[g]} số)" for g in GROUPS},
        tiers=[{"tier": t.name, "rule": t.rule, "value": t.value, "probability": d.tier_probabilities[t.name]} for t in prod.tiers],
        bao_types=list(prod.bao_types),
        rtp_distinct_numbers=d.return_to_player,
        p_any_prize=d.p_any_prize,
        top_prize_odds=1 / d.tier_probabilities[prod.tiers[0].name],
    )
