"""Game specifications for Vietlott matrix lotteries (Mega 6/45, Power 6/55, Lotto 5/35).

Everything that depends on the rules of a game — pool size, prize tiers, ticket
price, exact tier probabilities, bonus/special-number mechanics, jackpot rules, tax —
lives here so that every other module is game-agnostic.

Prize tables (VND, per 10,000 VND play) as published by Vietlott:

* Mega 6/45  : Jackpot (6) ≥ 12 tỷ (pari-mutuel) · 5 → 10,000,000 · 4 → 300,000 · 3 → 30,000
* Power 6/55 : Jackpot 1 (6) ≥ 30 tỷ · Jackpot 2 (5 + bonus) ≥ 3 tỷ (both pari-mutuel)
               · 5 → 40,000,000 · 4 → 500,000 · 3 → 50,000.
               The jackpot fund of a draw is split 90 % / 10 % between JP1 and JP2.
               When JP1 > 300 tỷ and only JP2 is won, the excess over 300 tỷ goes to the
               JP2 winner(s).
* Lotto 5/35 : 5 main numbers from 1–35 plus one special number from 1–12 (separate drum).
               Độc đắc (5 + special) ≥ 6 tỷ · 5 → 10,000,000 · 4 + special → 5,000,000
               · 4 → 500,000 · 3 + special → 100,000 · 3 → 30,000 · special only → 10,000.
               Two draws a day (13:00, 21:00). When the jackpot exceeds 12 tỷ without a
               winner, the 21:00 draw of the following day is a "chia giải" (rolldown) draw:
               the jackpot is split — 1/3 to the first-prize tier, 1/6 to each of tiers 2–5.

Personal income tax on lottery winnings (Luật Thuế TNCN 2025, effective 01/07/2026):
10 % on the part of each winning above 20,000,000 VND.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import comb
from typing import Literal, Mapping


class GameCode(str, Enum):
    """Supported games. String values are used in URLs, file names and the database."""

    MEGA_645 = "mega645"
    POWER_655 = "power655"
    LOTTO_535 = "lotto535"


BonusMode = Literal["none", "same_drum", "separate"]


@dataclass(frozen=True)
class PrizeTier:
    """One prize tier.

    Attributes:
        name: Stable identifier (``jackpot1``, ``jackpot2``, ``first``...).
        main_matches: Number of main numbers that must match (lower bound when
            ``main_matches_max`` is set).
        bonus_required: The ticket must hold the bonus ball (Power JP2) or match the
            special number (Lotto 5/35).
        fixed_amount: Fixed prize in VND, or ``None`` for pari-mutuel jackpots.
        main_matches_max: Upper bound of a main-match range (Lotto 5/35 "special only"
            consolation prize: 0–2 main numbers + special).
    """

    name: str
    main_matches: int
    bonus_required: bool = False
    fixed_amount: int | None = None
    main_matches_max: int | None = None

    @property
    def is_jackpot(self) -> bool:
        return self.fixed_amount is None

    def main_range(self) -> range:
        hi = self.main_matches if self.main_matches_max is None else self.main_matches_max
        return range(min(self.main_matches, hi), max(self.main_matches, hi) + 1)


@dataclass(frozen=True)
class TaxRule:
    """Flat tax on the part of a single winning above ``threshold``."""

    rate: float = 0.10
    threshold: int = 20_000_000

    def after_tax(self, amount: float) -> float:
        return amount - self.rate * max(0.0, amount - self.threshold)


@dataclass(frozen=True)
class RolldownRule:
    """Lotto 5/35 "chia giải Độc đắc": jackpot split among lower tiers above a threshold."""

    threshold: int
    shares: Mapping[str, float]


@dataclass(frozen=True)
class GameSpec:
    """Immutable rules of a k-of-n matrix lottery."""

    code: GameCode
    display_name: str
    pool_size: int
    pick: int
    bonus_mode: BonusMode
    ticket_price: int
    tiers: tuple[PrizeTier, ...]
    min_jackpots: Mapping[str, int] = field(default_factory=dict)
    draw_weekdays: tuple[int, ...] = ()  # Monday=0
    draws_per_day: int = 1
    bonus_pool_size: int | None = None  # separate drum size (Lotto 5/35: 12)
    jackpot_split: Mapping[str, float] = field(default_factory=dict)  # Power: JP1 90 %, JP2 10 %
    jackpot_cap: int | None = None  # Power: 300 tỷ rule
    rolldown: RolldownRule | None = None
    bao_levels: tuple[int, ...] = ()  # "chơi bao" sizes offered by Vietlott (k−1 = Bao 5 / Bao 4)
    special_bao_max: int = 0  # Lotto 5/35 "bao số đặc biệt": up to 12 special numbers on one main set

    # ------------------------------------------------------------------ basics
    @property
    def has_bonus(self) -> bool:
        return self.bonus_mode != "none"

    @property
    def separate_special(self) -> bool:
        return self.bonus_mode == "separate"

    @property
    def numbers(self) -> range:
        return range(1, self.pool_size + 1)

    @cached_property
    def total_combinations(self) -> int:
        """Distinct main-number combinations (× special numbers for separate-drum games)."""
        base = comb(self.pool_size, self.pick)
        return base * (self.bonus_pool_size or 1) if self.separate_special else base

    @cached_property
    def main_combinations(self) -> int:
        return comb(self.pool_size, self.pick)

    @property
    def inclusion_probability(self) -> float:
        """P(a given number is among the main numbers of a draw) = k/n."""
        return self.pick / self.pool_size

    def tier(self, name: str) -> PrizeTier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(f"{self.code.value} has no tier {name!r}")

    # ------------------------------------------------------------- probability
    def match_probability(self, m: int) -> float:
        """Hypergeometric P(exactly m of the k main numbers on a ticket are drawn)."""
        n, k = self.pool_size, self.pick
        if m < 0 or m > k:
            return 0.0
        return comb(k, m) * comb(n - k, k - m) / comb(n, k)

    def match_distribution(self) -> list[float]:
        return [self.match_probability(m) for m in range(self.pick + 1)]

    def bonus_hit_probability(self, main_matches: int) -> float:
        """P(bonus / special condition is met | m main matches) for a random ticket.

        Same drum (Power): the bonus is drawn from the n−k non-winning numbers; a ticket
        with m main matches holds k−m of them. Separate drum: 1 / bonus_pool_size.
        """
        if self.bonus_mode == "none":
            return 0.0
        if self.bonus_mode == "separate":
            return 1.0 / float(self.bonus_pool_size or 1)
        return (self.pick - main_matches) / (self.pool_size - self.pick)

    def outcome_tier(self, main_matches: int, bonus_hit: bool) -> PrizeTier | None:
        """Highest tier for an outcome (tiers are ordered from highest to lowest)."""
        for t in self.tiers:
            if main_matches not in t.main_range():
                continue
            if t.bonus_required and not bonus_hit:
                continue
            return t
        return None

    @cached_property
    def tier_probabilities(self) -> dict[str, float]:
        """Exact probability that one play wins each tier (highest tier only)."""
        probs = {t.name: 0.0 for t in self.tiers}
        for m in range(self.pick + 1):
            pm = self.match_probability(m)
            ph = self.bonus_hit_probability(m)
            for hit, p in ((True, ph), (False, 1.0 - ph)):
                if p <= 0:
                    continue
                t = self.outcome_tier(m, hit)
                if t is not None:
                    probs[t.name] += pm * p
        return probs

    @cached_property
    def p_any_prize(self) -> float:
        return float(sum(self.tier_probabilities.values()))

    def classify(self, main_matches: int, bonus_hit: bool = False) -> PrizeTier | None:
        """Return the highest tier won by a play, or ``None``."""
        return self.outcome_tier(main_matches, bonus_hit)


MEGA_645 = GameSpec(
    code=GameCode.MEGA_645,
    display_name="Mega 6/45",
    pool_size=45,
    pick=6,
    bonus_mode="none",
    ticket_price=10_000,
    tiers=(
        PrizeTier("jackpot1", 6),
        PrizeTier("first", 5, fixed_amount=10_000_000),
        PrizeTier("second", 4, fixed_amount=300_000),
        PrizeTier("third", 3, fixed_amount=30_000),
    ),
    min_jackpots={"jackpot1": 12_000_000_000},
    draw_weekdays=(2, 4, 6),
    jackpot_split={"jackpot1": 1.0},
    bao_levels=(5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18),
)

POWER_655 = GameSpec(
    code=GameCode.POWER_655,
    display_name="Power 6/55",
    pool_size=55,
    pick=6,
    bonus_mode="same_drum",
    ticket_price=10_000,
    tiers=(
        PrizeTier("jackpot1", 6),
        PrizeTier("jackpot2", 5, bonus_required=True),
        PrizeTier("first", 5, fixed_amount=40_000_000),
        PrizeTier("second", 4, fixed_amount=500_000),
        PrizeTier("third", 3, fixed_amount=50_000),
    ),
    min_jackpots={"jackpot1": 30_000_000_000, "jackpot2": 3_000_000_000},
    draw_weekdays=(1, 3, 5),
    jackpot_split={"jackpot1": 0.9, "jackpot2": 0.1},
    jackpot_cap=300_000_000_000,
    bao_levels=(5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18),
)

LOTTO_535 = GameSpec(
    code=GameCode.LOTTO_535,
    display_name="Lotto 5/35",
    pool_size=35,
    pick=5,
    bonus_mode="separate",
    bonus_pool_size=12,
    ticket_price=10_000,
    tiers=(
        PrizeTier("jackpot1", 5, bonus_required=True),
        PrizeTier("first", 5, fixed_amount=10_000_000),
        PrizeTier("second", 4, bonus_required=True, fixed_amount=5_000_000),
        PrizeTier("third", 4, fixed_amount=500_000),
        PrizeTier("fourth", 3, bonus_required=True, fixed_amount=100_000),
        PrizeTier("fifth", 3, fixed_amount=30_000),
        PrizeTier("consolation", 0, bonus_required=True, fixed_amount=10_000, main_matches_max=2),
    ),
    min_jackpots={"jackpot1": 6_000_000_000},
    draw_weekdays=(0, 1, 2, 3, 4, 5, 6),
    draws_per_day=2,
    jackpot_split={"jackpot1": 1.0},
    rolldown=RolldownRule(
        threshold=12_000_000_000,
        shares={"first": 1 / 3, "second": 1 / 6, "third": 1 / 6, "fourth": 1 / 6, "fifth": 1 / 6},
    ),
    bao_levels=(4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    special_bao_max=12,
)

GAMES: dict[GameCode, GameSpec] = {g.code: g for g in (MEGA_645, POWER_655, LOTTO_535)}
SIX_NUMBER_GAMES = (GameCode.MEGA_645, GameCode.POWER_655)
DEFAULT_TAX = TaxRule()


def get_game(code: str | GameCode) -> GameSpec:
    """Resolve a game from its code (``mega645`` / ``power655`` / ``lotto535``; aliases accepted)."""
    aliases = {
        "645": GameCode.MEGA_645,
        "mega": GameCode.MEGA_645,
        "power645": GameCode.MEGA_645,
        "655": GameCode.POWER_655,
        "power": GameCode.POWER_655,
        "535": GameCode.LOTTO_535,
        "lotto": GameCode.LOTTO_535,
        "power535": GameCode.LOTTO_535,
    }
    key = code.value if isinstance(code, GameCode) else str(code).lower().strip()
    try:
        return GAMES[GameCode(key)]
    except ValueError:
        if key in aliases:
            return GAMES[aliases[key]]
        raise KeyError(f"Unknown game {code!r}; expected one of {[g.value for g in GameCode]}") from None
