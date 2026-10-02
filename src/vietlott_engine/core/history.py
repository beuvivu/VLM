"""Vectorised, read-only draw history — the single input type of every model.

``DrawHistory`` stores draws in chronological order as NumPy arrays:

* ``numbers``   (D, k) int16 — sorted main numbers
* ``incidence`` (D, n) bool  — incidence[d, i-1] is True if number i was drawn in draw d
* ``bonus``     (D,)  int16 — Power 6/55 bonus ball (0 when absent)

All arrays are flagged non-writeable. ``upto(t)`` returns the strict prefix of the
first ``t`` draws; the backtest engine only ever hands strategies such prefixes, which
is what makes look-ahead bias structurally impossible (see ``vietlott_engine.backtest.engine``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np

from vietlott_engine.core.exceptions import InsufficientDataError
from vietlott_engine.core.games import GameSpec, get_game
from vietlott_engine.core.models import Draw


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class DrawHistory:
    """Chronologically ordered draws of one game (immutable)."""

    spec: GameSpec
    draw_ids: np.ndarray
    dates: np.ndarray  # datetime64[D]
    numbers: np.ndarray
    incidence: np.ndarray
    bonus: np.ndarray

    # ----------------------------------------------------------- construction
    @classmethod
    def from_draws(cls, draws: Iterable[Draw], spec: GameSpec | None = None) -> "DrawHistory":
        draws = sorted(draws, key=lambda d: (d.draw_date, d.draw_id))
        if spec is None:
            if not draws:
                raise InsufficientDataError("cannot infer game from an empty draw list")
            spec = get_game(draws[0].game)
        if any(d.game != spec.code for d in draws):
            raise ValueError("draws from several games cannot share one history")
        return cls.from_arrays(
            spec,
            numbers=np.array([d.numbers for d in draws], dtype=np.int16).reshape(-1, spec.pick),
            draw_ids=np.array([d.draw_id for d in draws], dtype=np.int64),
            dates=np.array([np.datetime64(d.draw_date, "D") for d in draws], dtype="datetime64[D]"),
            bonus=np.array([d.bonus or 0 for d in draws], dtype=np.int16),
        )

    @classmethod
    def from_arrays(
        cls,
        spec: GameSpec,
        numbers: np.ndarray,
        draw_ids: np.ndarray | None = None,
        dates: np.ndarray | None = None,
        bonus: np.ndarray | None = None,
    ) -> "DrawHistory":
        numbers = np.sort(np.asarray(numbers, dtype=np.int16).reshape(-1, spec.pick), axis=1)
        d = numbers.shape[0]
        if d and (numbers.min() < 1 or numbers.max() > spec.pool_size):
            raise ValueError(f"numbers outside 1..{spec.pool_size}")
        incidence = np.zeros((d, spec.pool_size), dtype=bool)
        if d:
            incidence[np.repeat(np.arange(d), spec.pick), numbers.ravel() - 1] = True
            if (incidence.sum(axis=1) != spec.pick).any():
                raise ValueError("a draw contains duplicate numbers")
        if draw_ids is None:
            draw_ids = np.arange(1, d + 1, dtype=np.int64)
        if dates is None:
            dates = np.datetime64("2000-01-01") + np.arange(d).astype("timedelta64[D]")
        if bonus is None:
            bonus = np.zeros(d, dtype=np.int16)
        return cls(
            spec=spec,
            draw_ids=_readonly(np.asarray(draw_ids, dtype=np.int64).copy()),
            dates=_readonly(np.asarray(dates, dtype="datetime64[D]").copy()),
            numbers=_readonly(numbers.copy()),
            incidence=_readonly(incidence),
            bonus=_readonly(np.asarray(bonus, dtype=np.int16).copy()),
        )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return int(self.numbers.shape[0])

    @property
    def n(self) -> int:
        return self.spec.pool_size

    @property
    def k(self) -> int:
        return self.spec.pick

    @property
    def last_date(self) -> date | None:
        return self.dates[-1].astype(date) if len(self) else None

    def upto(self, t: int) -> "DrawHistory":
        """Strict prefix: draws with index < t (views, no copy)."""
        if not 0 <= t <= len(self):
            raise IndexError(f"t={t} outside 0..{len(self)}")
        return DrawHistory(
            spec=self.spec,
            draw_ids=self.draw_ids[:t],
            dates=self.dates[:t],
            numbers=self.numbers[:t],
            incidence=self.incidence[:t],
            bonus=self.bonus[:t],
        )

    def detached(self) -> "DrawHistory":
        """Independent read-only copy. Slices share memory with the full history and
        ``ndarray.base`` would expose future draws; the backtest hands strategies
        detached prefixes so that is impossible."""
        return DrawHistory(
            spec=self.spec,
            draw_ids=_readonly(self.draw_ids.copy()),
            dates=_readonly(self.dates.copy()),
            numbers=_readonly(self.numbers.copy()),
            incidence=_readonly(self.incidence.copy()),
            bonus=_readonly(self.bonus.copy()),
        )

    def tail(self, m: int) -> "DrawHistory":
        start = max(0, len(self) - m)
        return DrawHistory(
            spec=self.spec,
            draw_ids=self.draw_ids[start:],
            dates=self.dates[start:],
            numbers=self.numbers[start:],
            incidence=self.incidence[start:],
            bonus=self.bonus[start:],
        )

    def require(self, min_draws: int) -> None:
        if len(self) < min_draws:
            raise InsufficientDataError(f"need at least {min_draws} draws, have {len(self)}")

    def counts(self, weights: np.ndarray | None = None) -> np.ndarray:
        """Per-number (optionally weighted) appearance counts, shape (n,)."""
        inc = self.incidence.astype(np.float64)
        return inc.sum(axis=0) if weights is None else weights @ inc

    def id_gaps(self) -> list[tuple[int, int]]:
        """Missing draw-id ranges ``(first_missing, last_missing)`` (data-quality check)."""
        ids = self.draw_ids
        out: list[tuple[int, int]] = []
        for a, b in zip(ids[:-1], ids[1:]):
            if b - a > 1:
                out.append((int(a + 1), int(b - 1)))
        return out

    def to_rows(self) -> list[dict]:
        return [
            {
                "draw_id": int(i),
                "draw_date": str(dt),
                "numbers": [int(x) for x in nums],
                "bonus": int(b) if b else None,
            }
            for i, dt, nums, b in zip(self.draw_ids, self.dates, self.numbers, self.bonus)
        ]


def decay_weights(length: int, decay: float) -> np.ndarray:
    """Exponential weights λ^(D-1-d): the most recent draw has weight 1."""
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if length == 0:
        return np.zeros(0)
    return decay ** np.arange(length - 1, -1, -1, dtype=np.float64)


def tickets_to_array(tickets: Sequence[Sequence[int]], k: int) -> np.ndarray:
    arr = np.sort(np.asarray(tickets, dtype=np.int16).reshape(-1, k), axis=1)
    return arr
