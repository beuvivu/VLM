"""Exact, pre-tax VND settlement. Shared pots use Fraction, never float.

No unobserved jackpot winner count is assumed. For capped fixed prizes, a
missing global count is explicitly reported as a nominal (uncapped) quote.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any

from vietlott_engine.core.games import get_game
from vietlott_engine.game_theory.bao import bao_kind
from vietlott_engine.game_theory.max3d import bao_plays, get_product, GROUP_SIZE, _tier_awards
from vietlott_engine.game_theory.fastgames import KENO_TABLE, BINGO_SUM_PAYOUT
from vlm.database.schema import DrawRecord, game_code


@dataclass(frozen=True)
class TicketCost:
    game: str
    plays: int
    cost: int
    documented: bool = True


@dataclass(frozen=True)
class Settlement:
    game: str
    plays: int
    cost: int
    prize_counts: dict[str, int]
    fixed_payout: Fraction
    jackpot_payout: Fraction | None = Fraction(0)
    notes: tuple[str, ...] = ()
    nominal: bool = False

    @property
    def gross_payout(self) -> Fraction | None:
        return None if self.jackpot_payout is None else self.fixed_payout + self.jackpot_payout


def _natural(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _selection(numbers: list[int], upper: int) -> list[int]:
    if any(type(n) is not int or not 1 <= n <= upper for n in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError(f"numbers must be distinct integers in 1..{upper}")
    return sorted(numbers)


def matrix_cost(game: str, numbers: list[int], specials: list[int] | None = None) -> TicketCost:
    spec = get_game(game)
    nums = _selection(numbers, spec.pool_size)
    sp = _selection(specials or [], spec.bonus_pool_size or 1) if spec.separate_special else []
    if spec.separate_special and not sp:
        raise ValueError("Lotto needs at least one special number")
    if not spec.separate_special and specials:
        raise ValueError("this game does not sell a separate special selection")
    _, plays, documented, note = bao_kind(spec, len(nums), len(sp) if sp else 1)
    # The existing engine can analyze research combinations; operational pricing is stricter.
    if len(nums) not in (spec.pick, *spec.bao_levels) or not documented or note:
        raise ValueError(note or "unsupported system size")
    return TicketCost(spec.code.value, plays, plays * spec.ticket_price)


@dataclass(frozen=True)
class PowerJackpotDistribution:
    transfer: int
    payable1: int
    payable2: int
    per_winner1: Fraction | None
    per_winner2: Fraction | None
    next_base1: int
    next_base2: int


def power_jackpot_distribution(jackpot1: int, jackpot2: int, winners1: int, winners2: int) -> PowerJackpotDistribution:
    """Inputs are pre-transfer pools and ALL winning units in this draw."""
    for value, label in [(jackpot1,"jackpot1"),(jackpot2,"jackpot2"),(winners1,"winners1"),(winners2,"winners2")]:
        _natural(value, label, 0)
    transfer = max(0, jackpot1 - 300_000_000_000) if winners1 == 0 and winners2 > 0 else 0
    p2 = jackpot2 + transfer
    return PowerJackpotDistribution(
        transfer, jackpot1, p2, Fraction(jackpot1,winners1) if winners1 else None,
        Fraction(p2,winners2) if winners2 else None,
        30_000_000_000 if winners1 else jackpot1 - transfer,
        3_000_000_000 if winners2 else jackpot2,
    )


def _choose(n: int, k: int) -> int:
    return comb(n,k) if 0 <= k <= n else 0


def settle_matrix(game: str, numbers: list[int], winning_numbers: list[int], *,
                  bonus_number: int | None = None, specials: list[int] | None = None,
                  jackpot_pots: dict[str,int] | None = None,
                  total_jackpot_winners: dict[str,int] | None = None,
                  jackpot_pots_basis: str = "payable") -> Settlement:
    """Settle against payable pools; pre-transfer Power pools require opt-in.

    Published detail/canonical pools already include any redistributed amount.
    """
    if jackpot_pots_basis not in ("payable", "pre_transfer"):
        raise ValueError("jackpot_pots_basis must be payable or pre_transfer")
    quote = matrix_cost(game,numbers,specials)
    spec = get_game(game)
    DrawRecord(game_type=spec.code.value, draw_id=1, draw_date="2000-01-01 00:00:00",
               winning_numbers=winning_numbers, bonus_number=bonus_number)
    selected, wins = set(numbers), set(winning_numbers)
    h, v, k = len(selected & wins), len(numbers), spec.pick
    counts: Counter[str] = Counter()

    def add(m: int, hit: bool, count: int) -> None:
        tier = spec.classify(m,hit)
        if tier and count > 0:
            counts[tier.name] += count

    if v == k - 1:
        candidates = [(h+1,k-h), (h,spec.pool_size-v-(k-h))]
        for m, count in candidates:
            if spec.bonus_mode == "same_drum":
                with_bonus = count if bonus_number in selected else (1 if m == h else 0)
                add(m,True,with_bonus)
                add(m,False,count-with_bonus)
            elif spec.separate_special:
                hit = int(bonus_number in (specials or []))
                add(m,True,count*hit)
                add(m,False,count*(len(specials or [])-hit))
            else:
                add(m,False,count)
    else:
        for m in range(k+1):
            count = _choose(h,m)*_choose(v-h,k-m)
            if spec.bonus_mode == "same_drum":
                with_bonus = _choose(h,m)*_choose(v-h-1,k-m-1) if bonus_number in selected else 0
                add(m,True,with_bonus)
                add(m,False,count-with_bonus)
            elif spec.separate_special:
                hit = int(bonus_number in (specials or []))
                add(m,True,count*hit)
                add(m,False,count*(len(specials or [])-hit))
            else:
                add(m,False,count)
    fixed = sum(spec.tier(t).fixed_amount * n for t,n in counts.items() if not spec.tier(t).is_jackpot)
    own_jp = {t:n for t,n in counts.items() if spec.tier(t).is_jackpot}
    payout: Fraction | None = Fraction(0)
    notes = []
    if own_jp:
        if jackpot_pots is None or total_jackpot_winners is None:
            payout = None
            notes.append("Actual jackpot settlement requires pools and total winning units.")
        else:
            for t,n in own_jp.items():
                _natural(jackpot_pots[t],t,0)
                if _natural(total_jackpot_winners[t],t,0) < n:
                    raise ValueError("global winners must include every winning play on this ticket")
            pots = dict(jackpot_pots)
            if spec.code.value == "power655" and jackpot_pots_basis == "pre_transfer":
                dist = power_jackpot_distribution(pots["jackpot1"],pots["jackpot2"],
                                                  total_jackpot_winners["jackpot1"],total_jackpot_winners["jackpot2"])
                pots["jackpot2"] = dist.payable2
            payout = sum((Fraction(pots[t]*n,total_jackpot_winners[t]) for t,n in own_jp.items()),Fraction(0))
    return Settlement(quote.game,quote.plays,quote.cost,dict(counts),Fraction(fixed),payout,tuple(notes))


def max3d_cost(game: str, kind: str, numbers: list[int | str], *, stake_multiple: int = 1,
               allow_unverified: bool = False) -> TicketCost:
    prod = get_product(game)
    _natural(stake_multiple,"stake_multiple")
    if stake_multiple > prod.max_stake_multiple:
        raise ValueError(f"{prod.code} maximum stake multiple is {prod.max_stake_multiple}")
    if any(type(x) not in (str,int) or (isinstance(x,str) and not x.isdigit()
           and not (kind=="bao_vi_tri" and len(x)==3 and all(c in "0123456789*" for c in x))) for x in numbers):
        raise ValueError("Max numbers must be integers or digit strings")
    documented = kind == "co_ban" or (prod.bao_documented and kind in prod.bao_types)
    if not documented and not allow_unverified:
        raise ValueError("this Max system option has not been verified as officially sold")
    plays = bao_plays(prod,kind,numbers)
    return TicketCost(prod.code,len(plays),len(plays)*prod.price*stake_multiple,documented)


def settle_max3d(game: str, kind: str, numbers: list[int | str], winning_numbers: list[int | str], *,
                 stake_multiple: int = 1, total_top_winning_units: int | None = None,
                 allow_unverified: bool = False) -> Settlement:
    quote=max3d_cost(game,kind,numbers,stake_multiple=stake_multiple,allow_unverified=allow_unverified)
    prod=get_product(game)
    row=DrawRecord(game_type=game_code(prod.code),draw_id=1,draw_date="2000-01-01 00:00:00",winning_numbers=winning_numbers)
    dr=[int(n) for n in row.winning_numbers]
    groups, cursor = {}, 0
    for g,n in GROUP_SIZE.items():
        groups[g], cursor=dr[cursor:cursor+n],cursor+n
    counts: Counter[str] = Counter()
    payout=Fraction(0)
    top_units=0
    for play in bao_plays(prod,kind,numbers):
        single=len(play)==1
        x,y=play[0],play[-1]
        same=x==y
        d=tuple("x" if n==x else "y" if n==y and not same else "o" for n in groups["D"])
        cx={g:xs.count(x) for g,xs in groups.items()}
        cy={g:xs.count(y) for g,xs in groups.items()}
        mult=stake_multiple * (int(prod.identical_pair_multiplier) if same and not single else 1)
        for tier in prod.tiers:
            awards=_tier_awards(prod,tier,d,cx,cy,single,same)
            if awards:
                counts[tier.name]+=mult*awards
                payout+=tier.value*mult*awards
                if tier == prod.tiers[0]:
                    top_units+=mult
    nominal=bool(top_units and prod.top_prize_cap and total_top_winning_units is None)
    if total_top_winning_units is not None:
        _natural(total_top_winning_units,"total_top_winning_units",0)
        if total_top_winning_units < top_units:
            raise ValueError("global top units must include this ticket")
        if prod.top_prize_cap and top_units:
            unit=min(Fraction(prod.tiers[0].value),Fraction(prod.top_prize_cap,total_top_winning_units))
            payout += top_units*(unit-prod.tiers[0].value)
    notes = ["Top-prize payout is nominal until global winning units are known."] if nominal else []
    if not quote.documented:
        notes.append("Research expansion; official sale format unverified.")
    return Settlement(quote.game,quote.plays,quote.cost,dict(counts),payout,notes=tuple(notes),nominal=nominal)


def settle_keno(numbers: list[int], winning_numbers: list[int], *, side_bet: str | None = None,
                stake_multiple: int = 1, total_top_winning_units: int | None = None,
                rules_variant: str = "product") -> Settlement:
    _natural(stake_multiple,"stake_multiple")
    if stake_multiple>50:
        raise ValueError("Keno stake multiple must be <= 50")
    draw=DrawRecord(game_type="keno",draw_id=1,draw_date="2000-01-01 00:00:00",winning_numbers=winning_numbers)
    dr=[int(x) for x in draw.winning_numbers]
    _selection(numbers,80)
    notes, nominal = [],False
    if rules_variant not in ("product","detail"):
        raise ValueError("rules_variant must be product or detail")
    if side_bet:
        if numbers:
            raise ValueError("side bets do not take a number selection")
        large=sum(x>40 for x in dr)
        even=sum(x%2==0 for x in dr)
        side_counts={"big":large,"small":20-large,"even":even,"odd":20-even,
                     "even_11_12":even,"odd_11_12":20-even,"big_small_tie":large,"even_odd_tie":even}
        if side_bet not in side_counts:
            raise ValueError("unknown Keno side bet")
        c=side_counts[side_bet]
        if side_bet in ("big","small"):
            pay=26000 if c>=13 else 10000 if c in (11,12) else 0
        elif side_bet in ("even","odd"):
            pay=200000 if c>=15 else 40000 if c in (13,14) else 0
        elif side_bet.endswith("11_12"):
            pay=20000 if c in (11,12) else 0
        else:
            pay=(26000 if side_bet=="big_small_tie" else 20000) if c==10 else 0
        tier=side_bet
    else:
        b=len(numbers)
        if not 1<=b<=10:
            raise ValueError("Keno requires 1..10 selected numbers")
        hits=len(set(numbers)&set(dr))
        pay=KENO_TABLE[b].get(hits,0)
        tier=f"bac_{b}_hits_{hits}"
        if (b,hits)==(5,4):
            pay=150000 if rules_variant=="product" else 0
            notes.append("Product page: 150000 VND; detail page: 0 VND. Explicit source variant required for reconciliation.")
        if b in (8,9,10) and hits==b:
            if total_top_winning_units is None:
                nominal=True
                notes.append("Top payout nominal; each top tier has a 10 billion VND draw cap.")
            else:
                _natural(total_top_winning_units,"total_top_winning_units")
                if total_top_winning_units<stake_multiple:
                    raise ValueError("global winning units must include this stake")
                pay=min(Fraction(pay),Fraction(10000000000,total_top_winning_units))
    return Settlement("keno",1,10000*stake_multiple,{tier:stake_multiple} if pay else {},
                      Fraction(pay)*stake_multiple,notes=tuple(notes),nominal=nominal)


def bingo18_catalog() -> list[tuple[str,int | None]]:
    return [(kind,n) for kind in ("single","double","triple") for n in range(1,7)] + [
        ("any_triple",None)] + [("sum",s) for s in range(3,19)] + [(kind,None) for kind in ("small","tie","big")]


def settle_bingo18(kind: str, selection: int | None, winning_numbers: list[int], *, stake_multiple: int = 1) -> Settlement:
    _natural(stake_multiple,"stake_multiple")
    if (kind,selection) not in bingo18_catalog() or (selection is not None and type(selection) is not int):
        raise ValueError("invalid Bingo18 bet")
    dr=DrawRecord(game_type="bingo18",draw_id=1,draw_date="2000-01-01 00:00:00",winning_numbers=winning_numbers).winning_numbers
    count=dr.count(selection)
    total=sum(dr)
    if kind=="single": pay={1:12000,2:20000,3:30000}.get(count,0)
    elif kind=="double": pay=75000 if count>=2 else 0
    elif kind=="triple": pay=1200000 if count==3 else 0
    elif kind=="any_triple": pay=200000 if len(set(dr))==1 else 0
    elif kind=="sum": pay=BINGO_SUM_PAYOUT[selection] if total==selection else 0
    elif kind=="small": pay=15000 if total<=9 else 0
    elif kind=="tie": pay=20000 if total in (10,11) else 0
    else: pay=15000 if total>=12 else 0
    return Settlement("bingo18",1,10000*stake_multiple,{kind:stake_multiple} if pay else {},Fraction(pay*stake_multiple))
