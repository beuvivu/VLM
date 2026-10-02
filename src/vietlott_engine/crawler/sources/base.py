"""Data-source abstraction. Every source yields validated ``Draw`` objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vietlott_engine.core.games import GameSpec
from vietlott_engine.core.models import Draw


@dataclass
class FetchResult:
    """Validated draws plus a record of rows that failed validation (never silently dropped)."""

    draws: list[Draw] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (raw row, reason)


class DrawSource(ABC):
    """A provider of historical draw results."""

    name: str = "abstract"

    @abstractmethod
    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        """Return draws with ``draw_id > since_id`` (all draws when ``since_id`` is None)."""
