"""Pydantic domain models: validated draws and tickets."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vietlott_engine.core.games import GameCode, GameSpec, get_game

Number = Annotated[int, Field(ge=1, le=99)]


class Draw(BaseModel):
    """One official draw result.

    ``numbers`` are the six main numbers (sorted). ``bonus`` is the Power 6/55
    special ball and must be ``None`` for Mega 6/45. Jackpot values and winner counts
    are optional because not every source publishes them.
    """

    model_config = ConfigDict(frozen=True)

    game: GameCode
    draw_id: int = Field(ge=1)
    draw_date: date
    numbers: tuple[Number, ...]
    bonus: Number | None = None
    jackpot1_value: int | None = Field(default=None, ge=0)
    jackpot2_value: int | None = Field(default=None, ge=0)
    tier_winners: dict[str, int] | None = None
    source: str = "unknown"

    @field_validator("numbers", mode="before")
    @classmethod
    def _sort_numbers(cls, v: object) -> tuple[int, ...]:
        return tuple(sorted(int(x) for x in v))  # type: ignore[union-attr]

    @model_validator(mode="after")
    def _check_against_game(self) -> "Draw":
        spec = get_game(self.game)
        validate_numbers(self.numbers, spec)
        if spec.has_bonus:
            if self.bonus is None:
                raise ValueError(f"{spec.display_name} draw {self.draw_id} is missing the bonus/special number")
            top = spec.bonus_pool_size if spec.separate_special else spec.pool_size
            if not 1 <= self.bonus <= (top or spec.pool_size):
                raise ValueError(f"bonus {self.bonus} outside 1..{top}")
            if not spec.separate_special and self.bonus in self.numbers:
                raise ValueError(f"bonus {self.bonus} duplicates a main number")
        elif self.bonus is not None:
            raise ValueError(f"{spec.display_name} has no bonus ball")
        return self

    @property
    def spec(self) -> GameSpec:
        return get_game(self.game)


class Ticket(BaseModel):
    """One play: k distinct numbers in 1..n (sorted on construction)."""

    model_config = ConfigDict(frozen=True)

    game: GameCode
    numbers: tuple[Number, ...]

    @field_validator("numbers", mode="before")
    @classmethod
    def _sort_numbers(cls, v: object) -> tuple[int, ...]:
        return tuple(sorted(int(x) for x in v))  # type: ignore[union-attr]

    @model_validator(mode="after")
    def _check(self) -> "Ticket":
        validate_numbers(self.numbers, get_game(self.game))
        return self


def validate_numbers(numbers: tuple[int, ...] | list[int], spec: GameSpec) -> None:
    """Raise ``ValueError`` unless ``numbers`` is a valid k-subset of 1..n."""
    if len(numbers) != spec.pick:
        raise ValueError(f"{spec.display_name} needs exactly {spec.pick} numbers, got {len(numbers)}")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"duplicate numbers in {list(numbers)}")
    bad = [x for x in numbers if not 1 <= x <= spec.pool_size]
    if bad:
        raise ValueError(f"numbers {bad} outside 1..{spec.pool_size}")
