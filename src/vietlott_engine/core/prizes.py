"""Per-draw prize results: number of winning tickets per tier and jackpot pot values.

Two representations:

* ``PrizeRecord`` — validated Pydantic row (storage, API).
* ``PrizeHistory`` — arrays aligned 1:1 with a ``DrawHistory`` (NaN where a draw has no
  prize data), the input of the behaviour calibration and the sales model.

``jackpot_pots`` holds the *total* value of each pari-mutuel pot at that draw (the
amount shared by its winners, or carried forward when nobody wins). Published tables
quote the value *per winner* when several people share a jackpot; loaders convert.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vietlott_engine.core.games import GameCode, GameSpec, get_game
from vietlott_engine.core.history import DrawHistory


class PrizeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    game: GameCode
    draw_id: int = Field(ge=1)
    draw_date: date
    winners: dict[str, int]
    jackpot_pots: dict[str, int] | None = None
    source: str = "unknown"
    flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "PrizeRecord":
        spec = get_game(self.game)
        names = {t.name for t in spec.tiers}
        unknown = set(self.winners) - names
        if unknown:
            raise ValueError(f"unknown tiers {sorted(unknown)} for {spec.display_name}")
        if any(v < 0 for v in self.winners.values()):
            raise ValueError("negative winner count")
        if self.jackpot_pots:
            bad = set(self.jackpot_pots) - {t.name for t in spec.tiers if t.is_jackpot}
            if bad:
                raise ValueError(f"{sorted(bad)} are not jackpot tiers")
        return self

    def fixed_prizes_paid(self, spec: GameSpec | None = None) -> float:
        """Total value of fixed-amount prizes paid in this draw (VND, before tax)."""
        spec = spec or get_game(self.game)
        return float(sum(self.winners.get(t.name, 0) * (t.fixed_amount or 0) for t in spec.tiers if not t.is_jackpot))


@dataclass(frozen=True)
class PrizeHistory:
    """Prize data aligned with a ``DrawHistory``.

    ``winners``: (D, T) float, NaN when unknown; columns follow ``spec.tiers`` order.
    ``pots``:    (D, J) float, NaN when unknown; columns follow ``jackpot_tiers``.
    """

    spec: GameSpec
    draw_ids: np.ndarray
    winners: np.ndarray
    pots: np.ndarray
    tier_names: tuple[str, ...]
    jackpot_tiers: tuple[str, ...]

    @classmethod
    def align(cls, history: DrawHistory, records: list[PrizeRecord]) -> "PrizeHistory":
        spec = history.spec
        names = tuple(t.name for t in spec.tiers)
        jps = tuple(t.name for t in spec.tiers if t.is_jackpot)
        idx = {int(i): r for i, r in enumerate(history.draw_ids)}
        pos = {v: k for k, v in idx.items()}
        win = np.full((len(history), len(names)), np.nan)
        pot = np.full((len(history), len(jps)), np.nan)
        for rec in records:
            row = pos.get(rec.draw_id)
            if row is None:
                continue
            if len(rec.winners) == len(names):
                win[row] = [rec.winners[nm] for nm in names]
            if rec.jackpot_pots and all(j in rec.jackpot_pots for j in jps):
                pot[row] = [rec.jackpot_pots[j] for j in jps]
        for a in (win, pot):
            a.setflags(write=False)
        return cls(spec=spec, draw_ids=history.draw_ids, winners=win, pots=pot, tier_names=names, jackpot_tiers=jps)

    @property
    def has_winners(self) -> np.ndarray:
        return ~np.isnan(self.winners).any(axis=1)

    @property
    def has_pots(self) -> np.ndarray:
        return ~np.isnan(self.pots).any(axis=1)

    def column(self, name: str) -> np.ndarray:
        return self.winners[:, self.tier_names.index(name)]

    def fixed_prizes_paid(self) -> np.ndarray:
        amounts = np.array([t.fixed_amount or 0 for t in self.spec.tiers], dtype=np.float64)
        return np.nansum(self.winners * amounts, axis=1) + np.where(self.has_winners, 0.0, np.nan)

    def jackpot_won(self) -> np.ndarray:
        """(D, J) bool — pot j had at least one winner at that draw (False when unknown)."""
        cols = [self.tier_names.index(j) for j in self.jackpot_tiers]
        w = self.winners[:, cols]
        return np.nan_to_num(w, nan=0.0) > 0
