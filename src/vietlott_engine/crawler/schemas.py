"""Pydantic schemas for raw records coming from each source, and their mapping to ``Draw``."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vietlott_engine.core.games import GameSpec
from vietlott_engine.core.models import Draw


class MirrorRecord(BaseModel):
    """One line of the community JSONL mirror (github.com/vietvudanh/vietlott-data).

    ``result`` holds the six main numbers; Power 6/55 appends the bonus ball as a 7th value.
    """

    model_config = ConfigDict(extra="ignore")

    date: date
    id: int = Field(ge=1)
    result: list[int]

    @field_validator("id", mode="before")
    @classmethod
    def _parse_id(cls, v: object) -> int:
        return int(str(v).strip())

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> object:
        # the mirror occasionally stores epoch milliseconds
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
            return datetime.fromtimestamp(int(v) / 1000).date()
        return v

    def to_draw(self, spec: GameSpec, source: str = "github_mirror") -> Draw:
        expected = spec.pick + (1 if spec.has_bonus else 0)
        if len(self.result) != expected:
            raise ValueError(f"draw {self.id}: expected {expected} values, got {len(self.result)}")
        main = self.result[: spec.pick]
        bonus = self.result[spec.pick] if spec.has_bonus else None
        return Draw(
            game=spec.code,
            draw_id=self.id,
            draw_date=self.date,
            numbers=tuple(main),
            bonus=bonus,
            source=source,
        )


class OfficialRow(BaseModel):
    """One row of the results table returned by vietlott.vn's AjaxPro endpoint."""

    date_text: str
    id_text: str
    values: list[int]

    def to_draw(self, spec: GameSpec, source: str = "vietlott") -> Draw:
        draw_date = datetime.strptime(self.date_text.strip(), "%d/%m/%Y").date()
        expected = spec.pick + (1 if spec.has_bonus else 0)
        if len(self.values) != expected:
            raise ValueError(f"draw {self.id_text}: expected {expected} values, got {len(self.values)}")
        return Draw(
            game=spec.code,
            draw_id=int(self.id_text.strip().lstrip("#")),
            draw_date=draw_date,
            numbers=tuple(self.values[: spec.pick]),
            bonus=self.values[spec.pick] if spec.has_bonus else None,
            source=source,
        )
