"""Exact mathematics of Vietlott "chơi bao" (system) tickets.

Official options (Vietlott; see README §Chơi bao for sources)
-------------------------------------------------------------
* **Mega 6/45, Power 6/55** — Bao 5 (5 numbers + every other number: 40 / 50 plays),
  Bao 7 … Bao 15 and Bao 18 (every 6-subset: C(v, 6) plays). No Bao 6, 16, 17.
* **Lotto 5/35** — Bao 4 (4 main numbers + each of the other 31: 31 plays), Bao 6 … Bao 15
  (C(v, 5) plays) and "bao số đặc biệt" (one set of 5 main numbers with 2 … 12 special
  numbers: one play per special number). Combining a main-number bao with several
  special numbers is not described in public material; it is analysed (plays × specials)
  but flagged as undocumented.

Every play costs one ticket price (10,000 đ).

Exact distribution
------------------
For Bao v, the number j of winning numbers inside the chosen set is hypergeometric and the
number of plays with exactly m matches is C(j, m)·C(v−j, k−m). The Power bonus ball (same
drum) and the Lotto special number (separate drum) are handled by conditioning on where
they fall. Nothing is simulated.

Rules that matter for a bao
---------------------------
* **Jackpots are shared per winning play.** If several plays of one bao hit the same
  jackpot (Power Bao 7, six numbers + bonus ⇒ six Jackpot-2 plays), the bao holds that many
  *shares* of one pot — it does not receive the pot several times. Payout = pot·n/(n + K),
  K = other winning plays.
* **Tax** (10 % above the threshold) applies per winning ticket. A bao is one ticket, so
  its prizes are added before tax (``tax_basis="ticket"``); ``"play"`` taxes each play
  separately, as if the same plays had been bought as single tickets.
* Expected value per đồng is the same as for single tickets (linearity of expectation);
  what changes is the *shape*: prizes arrive in correlated bundles, so the probability of
  winning anything is much lower than for the same number of independent tickets.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.core.games import DEFAULT_TAX, GameSpec, TaxRule

TaxBasis = Literal["ticket", "play"]


# ------------------------------------------------------------------ catalogue
class BaoOption(BaseModel):
    game: str
    kind: str
    main_numbers: int
    special_numbers: int
    plays: int
    cost: int
    documented: bool
    note: str = ""


def bao_kind(spec: GameSpec, v: int, s: int = 1) -> tuple[str, int, bool, str]:
    """(name, plays, documented by Vietlott, note) for v main numbers and s special numbers."""
    n, k = spec.pool_size, spec.pick
    if v == k - 1:
        main_plays, name = n - k + 1, f"Bao {k - 1}"
    elif v == k:
        main_plays, name = 1, "Cơ bản"
    elif v > k:
        main_plays, name = comb(v, k), f"Bao {v}"
    else:
        raise ValueError(f"choose k−1 = {k - 1} numbers or at least {k}")
    if not spec.separate_special:
        documented = v in (k, k - 1) or v in spec.bao_levels
        note = "" if documented else f"Vietlott không bán {name} cho {spec.display_name} (chỉ Bao {k - 1}, " + ", ".join(f"Bao {x}" for x in spec.bao_levels if x > k) + ")"
        return name, main_plays, documented, note
    plays = main_plays * s
    if s > 1:
        name = f"{name} × {s} số ĐB" if v != k else f"Bao {s} số đặc biệt"
    if v == k:
        documented = 1 <= s <= spec.special_bao_max
    else:
        documented = s == 1 and v in spec.bao_levels
    note = ""
    if not documented and s > 1 and v != k:
        note = "kết hợp bao số chính với nhiều số đặc biệt chưa thấy trong tài liệu công khai; tính theo nguyên tắc mọi tổ hợp × mọi số đặc biệt"
    elif not documented:
        note = f"ngoài các mức Vietlott công bố (Bao {k - 1}, Bao 6–15, bao số đặc biệt 2–{spec.special_bao_max})"
    return name, plays, documented, note


def bao_catalog(spec: GameSpec) -> list[BaoOption]:
    """Every bao option Vietlott sells for the game, with plays and cost."""
    k = spec.pick
    out = []
    levels = [k - 1] + [v for v in spec.bao_levels if v > k]
    for v in levels:
        name, plays, doc, note = bao_kind(spec, v, 1)
        out.append(BaoOption(game=spec.code.value, kind=name, main_numbers=v, special_numbers=1 if spec.separate_special else 0, plays=plays, cost=plays * spec.ticket_price, documented=doc, note=note))
    for s in range(2, spec.special_bao_max + 1):
        name, plays, doc, note = bao_kind(spec, k, s)
        out.append(BaoOption(game=spec.code.value, kind=name, main_numbers=k, special_numbers=s, plays=plays, cost=plays * spec.ticket_price, documented=doc, note=note))
    return out


# ------------------------------------------------------------------ results
class BaoOutcome(BaseModel):
    hits_in_set: int
    bonus: str  # "-", "in_set"/"not_in_set" (Power), "in_fixed"/"not_in_fixed" (Bao 5), "matched"/"missed" (Lotto)
    probability: float
    plays_per_tier: dict[str, int]
    fixed_prizes: float
    jackpot_plays: dict[str, int]
    payout_gross: float  # fixed + one share-weighted pot per jackpot tier won
    payout_net: float
    vietlott_style: str  # "JP2 + 24.000.000"


class BaoAnalysis(BaseModel):
    game: str
    kind: str
    numbers: list[int]
    specials: list[int] | None = None
    plays: int
    cost: int
    documented: bool
    notes: list[str] = []
    p_any_prize: float
    p_any_prize_same_budget_single_tickets: float  # B independent random tickets
    p_profit: float  # payout ≥ cost (gross)
    expected_payout: float  # gross, jackpots at the given values, no other winners
    expected_payout_net: float
    return_to_player: float
    tax_basis: TaxBasis
    jackpot_values: dict[str, float]
    outcomes: list[BaoOutcome]
    prize_table: list[dict]  # Vietlott-style: what the ticket wins when j drawn numbers are in the set


def _hyper(total: int, good: int, draws: int, j: int) -> float:
    if j < 0 or j > min(good, draws) or draws - j > total - good:
        return 0.0
    return comb(good, j) * comb(total - good, draws - j) / comb(total, draws)


def _value(spec: GameSpec, tier: str, jackpots: dict[str, float]) -> float:
    t = spec.tier(tier)
    return float(t.fixed_amount) if t.fixed_amount is not None else float(jackpots.get(tier, spec.min_jackpots.get(tier, 0)))


def fmt_vnd(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def ticket_payout(
    spec: GameSpec,
    plays: dict[str, int],
    jackpots: dict[str, float],
    co_winners: dict[str, float] | None = None,
    tax: TaxRule | None = None,
    tax_basis: TaxBasis = "ticket",
) -> tuple[float, float, float]:
    """(fixed prizes, gross payout, net payout) of one bao ticket for given plays per tier.

    A jackpot won by n plays of the ticket pays pot·n/(n + K), K = other winning plays."""
    co = co_winners or {}
    fixed = 0.0
    jp_shares: list[float] = []
    for tier, cnt in plays.items():
        if cnt <= 0:
            continue
        t = spec.tier(tier)
        if t.fixed_amount is not None:
            fixed += cnt * t.fixed_amount
        else:
            pot = _value(spec, tier, jackpots)
            k_other = float(co.get(tier, 0.0))
            jp_shares.extend([pot / (cnt + k_other)] * cnt)
    gross = fixed + sum(jp_shares)
    if tax is None:
        return fixed, gross, gross
    if tax_basis == "ticket":
        return fixed, gross, tax.after_tax(gross)
    net = sum(tax.after_tax(v) for v in jp_shares)
    for tier, cnt in plays.items():
        t = spec.tier(tier)
        if cnt > 0 and t.fixed_amount is not None:
            net += cnt * tax.after_tax(t.fixed_amount)
    return fixed, gross, net


def vietlott_style(spec: GameSpec, plays: dict[str, int]) -> str:
    parts = []
    labels = {"jackpot1": "Jackpot 1" if "jackpot2" in spec.min_jackpots else "Jackpot", "jackpot2": "Jackpot 2"}
    if spec.separate_special:
        labels["jackpot1"] = "Độc đắc"
    for t in spec.tiers:
        cnt = plays.get(t.name, 0)
        if t.is_jackpot and cnt:
            parts.append(labels.get(t.name, t.name) + (f" (×{cnt} phần)" if cnt > 1 else ""))
    fixed = sum(plays.get(t.name, 0) * (t.fixed_amount or 0) for t in spec.tiers if not t.is_jackpot)
    if fixed:
        parts.append(fmt_vnd(fixed))
    return " + ".join(parts) if parts else "0"


def _add(plays: dict[str, int], tier, count: int) -> None:  # type: ignore[no-untyped-def]
    if tier is None or count <= 0:
        return
    plays[tier.name] = plays.get(tier.name, 0) + count


def _outcome_plays(spec: GameSpec, v: int, n_specials: int) -> list[tuple[int, str, float, dict[str, int]]]:
    """(hits, bonus label, probability, plays per tier) for every outcome of a bao."""
    n, k = spec.pool_size, spec.pick
    out: list[tuple[int, str, float, dict[str, int]]] = []
    if v == k - 1:
        main_plays = n - (k - 1)
        for j0 in range(k):  # winners among the k−1 fixed numbers
            pj = _hyper(n, k - 1, k, j0)
            if pj == 0:
                continue
            up = k - j0  # plays whose added number is a winner → j0 + 1 matches
            rest = main_plays - up  # → j0 matches
            if spec.bonus_mode == "same_drum":
                p_in_f = (k - 1 - j0) / (n - k)  # bonus is one of the fixed non-winners
                for label, pb in (("in_fixed", p_in_f), ("not_in_fixed", 1 - p_in_f)):
                    if pb <= 0:
                        continue
                    pl: dict[str, int] = {}
                    if label == "in_fixed":
                        _add(pl, spec.classify(j0 + 1, True), up)
                        _add(pl, spec.classify(j0, True), rest)
                    else:
                        _add(pl, spec.classify(j0 + 1, False), up)
                        _add(pl, spec.classify(j0, True), 1)  # the play whose added number is the bonus
                        _add(pl, spec.classify(j0, False), rest - 1)
                    out.append((j0, label, pj * pb, pl))
            elif spec.separate_special:
                p_hit = n_specials / float(spec.bonus_pool_size or 1)
                for label, pb in (("matched", p_hit), ("missed", 1 - p_hit)):
                    if pb <= 0:
                        continue
                    hit = 1 if label == "matched" else 0
                    pl = {}
                    for m, cnt in ((j0 + 1, up), (j0, rest)):
                        _add(pl, spec.classify(m, True), cnt * hit)
                        _add(pl, spec.classify(m, False), cnt * (n_specials - hit))
                    out.append((j0, label, pj * pb, pl))
            else:
                pl = {}
                for m, cnt in ((j0 + 1, up), (j0, rest)):
                    _add(pl, spec.classify(m), cnt)
                out.append((j0, "-", pj, pl))
        return out
    for j in range(min(v, k) + 1):
        pj = _hyper(n, v, k, j)
        if pj == 0:
            continue
        counts = {m: comb(j, m) * comb(v - j, k - m) for m in range(k + 1)}
        if spec.bonus_mode == "same_drum":
            p_in = (v - j) / (n - k)
            for label, pb in (("in_set", p_in), ("not_in_set", 1 - p_in)):
                if pb <= 0:
                    continue
                pl = {}
                for m, cnt in counts.items():
                    if cnt == 0:
                        continue
                    # plays holding the bonus: m winners + the bonus + (k−m−1) other non-winners
                    with_b = comb(j, m) * comb(v - j - 1, k - m - 1) if label == "in_set" and m < k and v - j - 1 >= 0 and k - m - 1 >= 0 else 0
                    _add(pl, spec.classify(m, True), with_b)
                    _add(pl, spec.classify(m, False), cnt - with_b)
                out.append((j, label, pj * pb, pl))
        elif spec.separate_special:
            p_hit = n_specials / float(spec.bonus_pool_size or 1)
            for label, pb in (("matched", p_hit), ("missed", 1 - p_hit)):
                if pb <= 0:
                    continue
                hit = 1 if label == "matched" else 0
                pl = {}
                for m, cnt in counts.items():
                    _add(pl, spec.classify(m, True), cnt * hit)
                    _add(pl, spec.classify(m, False), cnt * (n_specials - hit))
                out.append((j, label, pj * pb, pl))
        else:
            pl = {}
            for m, cnt in counts.items():
                _add(pl, spec.classify(m), cnt)
            out.append((j, "-", pj, pl))
    return out


def _validate(spec: GameSpec, numbers: list[int], specials: list[int] | None) -> tuple[list[int], list[int]]:
    n = spec.pool_size
    nums = sorted(set(int(x) for x in numbers))
    if len(nums) != len(numbers) or any(not 1 <= x <= n for x in nums):
        raise ValueError(f"numbers must be distinct values in 1..{n}")
    if spec.separate_special:
        sp = sorted(set(int(s) for s in (specials or [])))
        if not sp or len(sp) != len(specials or []) or any(not 1 <= s <= int(spec.bonus_pool_size or 0) for s in sp):
            raise ValueError(f"{spec.display_name} needs 1..{spec.bonus_pool_size} distinct special numbers")
        return nums, sp
    if specials:
        raise ValueError(f"{spec.display_name} has no special number")
    return nums, []


def analyse_bao(
    spec: GameSpec,
    numbers: list[int],
    specials: list[int] | None = None,
    jackpots: dict[str, float] | None = None,
    after_tax: bool = False,
    tax_basis: TaxBasis = "ticket",
    co_winners: dict[str, float] | None = None,
    strict: bool = False,
    tax: TaxRule = DEFAULT_TAX,
) -> BaoAnalysis:
    """Exact payout distribution of a bao ticket.

    ``len(numbers) == k−1`` → Bao k−1 (fixed numbers + every other number);
    ``len(numbers) >= k`` → Bao v (``== k`` with several specials: Lotto bao số đặc biệt).
    ``strict`` rejects options Vietlott does not sell. ``co_winners`` = other winning
    plays per jackpot tier (default 0). ``after_tax`` selects which payout the RTP uses.
    """
    nums, sp = _validate(spec, numbers, specials)
    jackpots = dict(jackpots or {})
    v = len(nums)
    s = len(sp) if spec.separate_special else 1
    kind, plays, documented, note = bao_kind(spec, v, s)
    if strict and not documented:
        raise ValueError(note or f"{kind} is not offered by Vietlott")
    notes = [note] if note else []
    outcomes: list[BaoOutcome] = []
    for hits, label, prob, pl in _outcome_plays(spec, v, s):
        fixed, gross, net = ticket_payout(spec, pl, jackpots, co_winners, tax, tax_basis)
        outcomes.append(
            BaoOutcome(
                hits_in_set=hits,
                bonus=label,
                probability=prob,
                plays_per_tier=pl,
                fixed_prizes=fixed,
                jackpot_plays={t.name: pl[t.name] for t in spec.tiers if t.is_jackpot and pl.get(t.name)},
                payout_gross=gross,
                payout_net=net,
                vietlott_style=vietlott_style(spec, pl),
            )
        )
    probs = np.array([o.probability for o in outcomes])
    gross = np.array([o.payout_gross for o in outcomes])
    net = np.array([o.payout_net for o in outcomes])
    cost = plays * spec.ticket_price
    exp_g = float(probs @ gross)
    exp_n = float(probs @ net)
    if any(len(o.jackpot_plays) and max(o.jackpot_plays.values()) > 1 for o in outcomes):
        notes.append("có kết cục nhiều lượt cùng trúng một Jackpot: vé nhận nhiều phần của cùng một pot, không nhận pot nhiều lần")
    table = [
        {
            "hits_in_set": o.hits_in_set,
            "bonus": o.bonus,
            "plays_per_tier": o.plays_per_tier,
            "fixed_prizes": o.fixed_prizes,
            "jackpot_plays": o.jackpot_plays,
            "vietlott_style": o.vietlott_style,
            "payout": o.payout_gross,
            "probability": o.probability,
        }
        for o in outcomes
        if o.payout_gross > 0
    ]
    return BaoAnalysis(
        game=spec.code.value,
        kind=kind,
        numbers=nums,
        specials=sp or None,
        plays=plays,
        cost=cost,
        documented=documented,
        notes=notes,
        p_any_prize=float(probs[gross > 0].sum()),
        p_any_prize_same_budget_single_tickets=float(1 - (1 - spec.p_any_prize) ** plays),
        p_profit=float(probs[gross >= cost].sum()),
        expected_payout=exp_g,
        expected_payout_net=exp_n,
        return_to_player=(exp_n if after_tax else exp_g) / cost,
        tax_basis=tax_basis,
        jackpot_values={t.name: _value(spec, t.name, jackpots) for t in spec.tiers if t.is_jackpot},
        outcomes=outcomes,
        prize_table=table,
    )


def bao_prize_table(spec: GameSpec, v: int, n_specials: int = 1) -> list[dict]:
    """Vietlott-style lookup table of a bao level (independent of the chosen numbers)."""
    nums = list(range(1, v + 1))
    sp = list(range(1, n_specials + 1)) if spec.separate_special else None
    return analyse_bao(spec, nums, sp).prize_table


def bao_tickets(spec: GameSpec, numbers: list[int], specials: list[int] | None = None) -> list[tuple[tuple[int, ...], int | None]]:
    """Expand a bao into its individual plays (for simulation / printing)."""
    n, k = spec.pool_size, spec.pick
    nums = sorted(numbers)
    if len(nums) == k - 1:
        mains = [tuple(sorted(nums + [x])) for x in range(1, n + 1) if x not in nums]
    else:
        mains = [tuple(c) for c in combinations(nums, k)]
    sps = specials if spec.separate_special else [None]
    return [(c, s) for c in mains for s in (sps or [None])]


# ------------------------------------------------------------- crowd-aware EV
def expected_share(n: int, lam: float) -> float:
    """E[n / (n + K)], K ~ Poisson(λ): the bao's expected fraction of a shared pot."""
    if lam <= 1e-12:
        return 1.0
    kmax = int(stats.poisson.ppf(1 - 1e-12, lam)) + 2
    ks = np.arange(kmax + 1)
    return float(np.sum(stats.poisson.pmf(ks, lam) * n / (n + ks)))


class BaoEV(BaseModel):
    kind: str
    plays: int
    cost: int
    tickets_sold: int
    mean_popularity_ratio: float
    expected_payout_net: float
    return_to_player: float
    jackpot_part: float
    fixed_part: float


def bao_ev(
    spec: GameSpec,
    numbers: list[int],
    specials: list[int] | None,
    jackpots: dict[str, float],
    tickets_sold: int,
    popularity=None,  # type: ignore[no-untyped-def]
    tax: TaxRule = DEFAULT_TAX,
    tax_basis: TaxBasis = "ticket",
) -> BaoEV:
    """Expected after-tax payout of a bao when other players may hold the jackpot combination.

    Other winners K ~ Poisson(N·ratio/C) with ratio = mean popularity of the bao's plays
    (1 without a crowd model). Fixed prizes and taxes come from the exact distribution."""
    nums, sp = _validate(spec, numbers, specials)
    plays_list = bao_tickets(spec, nums, sp or None)
    ratio = 1.0
    if popularity is not None:
        combos = np.array([p[0] for p in plays_list], dtype=np.int64)
        if spec.separate_special:
            pr = popularity.ticket_probability_joint(combos, np.array([p[1] for p in plays_list]))
            ratio = float(np.mean(pr) * spec.total_combinations)
        else:
            ratio = float(np.mean(popularity.popularity_ratio(combos)))
    lam_base = tickets_sold * ratio / spec.total_combinations
    res = analyse_bao(spec, nums, sp or None, jackpots, tax=tax, tax_basis=tax_basis)
    total = fixed_part = 0.0
    for o in res.outcomes:
        # expected net payout of this outcome with random co-winners on each jackpot won
        if not o.jackpot_plays:
            total += o.probability * o.payout_net
            fixed_part += o.probability * o.payout_net
            continue
        # JP1: others holding the drawn combination; JP2 (Power): any of the k winning
        # 5+bonus combinations, so ≈ k−1 more combinations at uniform popularity
        lam = {t: lam_base if t != "jackpot2" else tickets_sold * (ratio + spec.pick - 1) / spec.total_combinations for t in o.jackpot_plays}
        # integrate over K for the (at most two) jackpot tiers with a small Poisson grid
        grids = []
        for t, nown in o.jackpot_plays.items():
            kmax = int(stats.poisson.ppf(1 - 1e-9, lam[t])) + 2
            ks = np.arange(kmax + 1)
            grids.append((t, ks, stats.poisson.pmf(ks, lam[t])))
        exp = 0.0
        if len(grids) == 1:
            t, ks, pk = grids[0]
            for kk, p in zip(ks, pk):
                exp += p * ticket_payout(spec, o.plays_per_tier, jackpots, {t: float(kk)}, tax, tax_basis)[2]
        else:
            (t1, k1, p1), (t2, k2, p2) = grids[:2]
            for a, pa in zip(k1, p1):
                for b, pb in zip(k2, p2):
                    exp += pa * pb * ticket_payout(spec, o.plays_per_tier, jackpots, {t1: float(a), t2: float(b)}, tax, tax_basis)[2]
        total += o.probability * exp
        fixed_part += o.probability * o.fixed_prizes
    return BaoEV(
        kind=res.kind,
        plays=res.plays,
        cost=res.cost,
        tickets_sold=int(tickets_sold),
        mean_popularity_ratio=ratio,
        expected_payout_net=total,
        return_to_player=total / res.cost,
        jackpot_part=total - fixed_part,
        fixed_part=fixed_part,
    )


# ----------------------------------------------------------- simulation & comparison
def _masks(arr: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(arr)).astype(np.uint64)
    return np.bitwise_or.reduce(np.left_shift(np.uint64(1), arr - np.uint64(1)), axis=1)


def simulate_plays(
    spec: GameSpec,
    mains: list[tuple[int, ...]] | np.ndarray,
    specials: list[int] | np.ndarray | None = None,
    sims: int = 200_000,
    seed: int | None = 0,
    jackpots: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Monte Carlo payout of an arbitrary set of plays (own jackpot plays share one pot)."""
    rng = np.random.default_rng(seed)
    n, k = spec.pool_size, spec.pick
    mains = np.asarray(mains, dtype=np.int64)
    p = len(mains)
    pm = _masks(mains)
    sp = np.asarray(specials if specials is not None else np.zeros(p), dtype=np.int64)
    names = [t.name for t in spec.tiers]
    table = np.full((k + 1, 2), -1)
    for m in range(k + 1):
        for h in (0, 1):
            t = spec.classify(m, bool(h))
            table[m, h] = names.index(t.name) if t else -1
    fixed_val = np.array([float(t.fixed_amount or 0) for t in spec.tiers] + [0.0])
    jp_idx = [i for i, t in enumerate(spec.tiers) if t.is_jackpot]
    pots = {i: _value(spec, names[i], jackpots or {}) for i in jp_idx}
    pay = np.empty(sims)
    best = np.empty(sims, dtype=np.int64)
    chunk = max(500, min(50_000, 3_000_000 // max(p, 1)))
    done = 0
    while done < sims:
        c = min(chunk, sims - done)
        if spec.bonus_mode == "same_drum":
            perm = np.argpartition(rng.random((c, n)), k + 1, axis=1)[:, : k + 1] + 1
            dm, bonus = _masks(perm[:, :k]), perm[:, k]
            hit = ((pm[:, None] >> (bonus.astype(np.uint64) - np.uint64(1))[None, :]) & np.uint64(1)).astype(bool)
        else:
            dm = _masks(np.argpartition(rng.random((c, n)), k, axis=1)[:, :k] + 1)
            if spec.separate_special:
                dsp = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, c)
                hit = sp[:, None] == dsp[None, :]
            else:
                hit = np.zeros((p, c), dtype=bool)
        m = np.bitwise_count(pm[:, None] & dm[None, :])
        t = table[m, hit.astype(np.int64)]  # (plays, draws), −1 = no prize
        tt = np.where(t >= 0, t, len(spec.tiers))
        val = fixed_val[tt].sum(axis=0)
        for i in jp_idx:
            val += np.where((t == i).any(axis=0), pots[i], 0.0)
        pay[done : done + c] = val
        best[done : done + c] = np.where(t >= 0, t, 99).min(axis=0)
        done += c
    return {"payout": pay, "best_tier": best}


class StrategyStats(BaseModel):
    strategy: str
    plays: int
    cost: int
    p_any_prize: float
    p_payout_at_least_1m: float
    p_profit: float  # payout ≥ cost
    expected_fixed_payout: float
    sd_payout: float
    note: str = ""


class BaoComparison(BaseModel):
    game: str
    bao: str
    numbers: list[int]
    rows: list[StrategyStats]
    simulations: int
    note: str


def _stats(name: str, pay: np.ndarray, plays: int, price: int, note: str = "") -> StrategyStats:
    cost = plays * price
    return StrategyStats(
        strategy=name,
        plays=plays,
        cost=cost,
        p_any_prize=float((pay > 0).mean()),
        p_payout_at_least_1m=float((pay >= 1_000_000).mean()),
        p_profit=float((pay >= cost).mean()),
        expected_fixed_payout=float(pay.mean()),
        sd_payout=float(pay.std()),
        note=note,
    )


def compare_bao_strategies(
    spec: GameSpec,
    numbers: list[int],
    specials: list[int] | None = None,
    sims: int = 100_000,
    seed: int | None = 0,
    reduced_guarantee: tuple[int, int] = (3, 3),
) -> BaoComparison:
    """Same money, different shapes: the bao vs a reduced wheel ("bao rút gọn") on the same
    numbers vs the same number of spread-out single tickets vs random quick picks.
    Jackpots are excluded (identical tiny probabilities); fixed prizes only."""
    from vietlott_engine.game_theory.coverage import optimise_coverage
    from vietlott_engine.wheeling.cover import WheelRequest, build_wheel

    nums, sp = _validate(spec, numbers, specials)
    res = analyse_bao(spec, nums, sp or None)
    zero_jp = {t.name: 0.0 for t in spec.tiers if t.is_jackpot}
    plays = bao_tickets(spec, nums, sp or None)
    mains = [p[0] for p in plays]
    spc = [p[1] or 0 for p in plays]
    rows = [_stats(res.kind, simulate_plays(spec, mains, spc, sims, seed, zero_jp)["payout"], res.plays, spec.ticket_price, "các lượt chồng lên nhau")]
    k = spec.pick
    t, m = reduced_guarantee
    if len(nums) > k:
        try:
            w = build_wheel(WheelRequest(pool=nums, ticket_size=k, guarantee=t, condition=m), spec, seed=seed, simulate=False, exact=len(nums) <= 12, time_limit=5.0)
            wsp = [sp[0] if sp else 0] * w.n_tickets
            rows.append(
                _stats(
                    f"Bao rút gọn ({t} nếu về {m})",
                    simulate_plays(spec, w.tickets, wsp, sims, seed, zero_jp)["payout"],
                    w.n_tickets,
                    spec.ticket_price,
                    f"{w.n_tickets} vé đảm bảo ≥ {t} số trúng trên một vé khi ≥ {m} số về trong {len(nums)} số đã chọn",
                )
            )
        except ValueError:
            pass
    if res.plays <= 400:
        cov = optimise_coverage(spec, res.plays, sim_draws=min(20_000, sims), eval_draws=min(20_000, sims), candidates=2000, local_rounds=1, seed=seed)
        rows.append(_stats("Vé lẻ dàn đều (cùng ngân sách)", simulate_plays(spec, cov.tickets, cov.specials, sims, seed, zero_jp)["payout"], res.plays, spec.ticket_price, "tối đa P(≥ 1 giải)"))
        rng = np.random.default_rng(None if seed is None else seed + 7)
        rnd = np.sort(np.argpartition(rng.random((res.plays, spec.pool_size)), k, axis=1)[:, :k] + 1, axis=1)
        rsp = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, res.plays) if spec.separate_special else None
        rows.append(_stats("Vé lẻ ngẫu nhiên (cùng ngân sách)", simulate_plays(spec, rnd, rsp, sims, seed, zero_jp)["payout"], res.plays, spec.ticket_price, "một danh mục quick pick"))
    return BaoComparison(
        game=spec.code.value,
        bao=res.kind,
        numbers=nums,
        rows=rows,
        simulations=sims,
        note=(
            "Cùng số lượt ⇒ cùng số giải kỳ vọng và cùng tiền thưởng kỳ vọng. Bao dồn giải vào ít kỳ (P trúng thấp, "
            "trúng thì nhiều giải cùng lúc); dàn đều trải ra nhiều kỳ. Bao rút gọn giữ một bảo đảm có điều kiện với chi phí "
            "thấp hơn nhiều."
        ),
    )
