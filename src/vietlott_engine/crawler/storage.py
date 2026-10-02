"""Persistence: DuckDB (primary, with Parquet export) and an in-memory repository for tests.

DuckDB gives an embedded, columnar, SQL-queryable store; every sync also writes
``<parquet_dir>/<game>.parquet`` so the data can be consumed by other tools
(pandas, polars, Spark, BI) without touching the database file.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Protocol

from vietlott_engine.core.exceptions import StorageError
from vietlott_engine.core.games import GameCode, GameSpec, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.models import Draw
from vietlott_engine.core.prizes import PrizeHistory, PrizeRecord

log = get_logger(__name__)


class DrawRepository(Protocol):
    def upsert(self, draws: Iterable[Draw]) -> int: ...
    def load(self, game: GameCode | str) -> list[Draw]: ...
    def max_draw_id(self, game: GameCode | str) -> int | None: ...
    def count(self, game: GameCode | str) -> int: ...
    def log_sync(self, game: GameCode | str, source: str, fetched: int, inserted: int, rejected: int) -> None: ...

    def load_history(self, game: GameCode | str) -> DrawHistory: ...

    def upsert_prizes(self, records: Iterable[PrizeRecord]) -> int: ...
    def load_prizes(self, game: GameCode | str) -> list[PrizeRecord]: ...
    def load_prize_history(self, game: GameCode | str) -> tuple[DrawHistory, PrizeHistory]: ...


class _HistoryMixin:
    def load_history(self, game: GameCode | str) -> DrawHistory:
        spec: GameSpec = get_game(game)
        return DrawHistory.from_draws(self.load(spec.code), spec)  # type: ignore[attr-defined]

    def load_prize_history(self, game: GameCode | str) -> tuple[DrawHistory, PrizeHistory]:
        h = self.load_history(game)
        return h, PrizeHistory.align(h, self.load_prizes(game))  # type: ignore[attr-defined]


class InMemoryRepository(_HistoryMixin):
    """Dictionary-backed repository (tests, notebooks, API demos)."""

    def __init__(self, draws: Iterable[Draw] = ()) -> None:
        self._data: dict[tuple[str, int], Draw] = {}
        self._prizes: dict[tuple[str, int], PrizeRecord] = {}
        self.sync_log: list[dict] = []
        self.upsert(draws)

    def upsert_prizes(self, records: Iterable[PrizeRecord]) -> int:
        n = 0
        for r in records:
            self._prizes[(r.game.value, r.draw_id)] = r
            n += 1
        return n

    def load_prizes(self, game: GameCode | str) -> list[PrizeRecord]:
        code = get_game(game).code.value
        return sorted((r for (g, _), r in self._prizes.items() if g == code), key=lambda r: r.draw_id)

    def upsert(self, draws: Iterable[Draw]) -> int:
        n = 0
        for d in draws:
            self._data[(d.game.value, d.draw_id)] = d
            n += 1
        return n

    def load(self, game: GameCode | str) -> list[Draw]:
        code = get_game(game).code.value
        return sorted((d for (g, _), d in self._data.items() if g == code), key=lambda d: (d.draw_date, d.draw_id))

    def max_draw_id(self, game: GameCode | str) -> int | None:
        ids = [d.draw_id for d in self.load(game)]
        return max(ids) if ids else None

    def count(self, game: GameCode | str) -> int:
        return len(self.load(game))

    def log_sync(self, game: GameCode | str, source: str, fetched: int, inserted: int, rejected: int) -> None:
        self.sync_log.append(dict(game=str(game), source=source, fetched=fetched, inserted=inserted, rejected=rejected))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    game            VARCHAR   NOT NULL,
    draw_id         INTEGER   NOT NULL,
    draw_date       DATE      NOT NULL,
    numbers         INTEGER[] NOT NULL,
    bonus           INTEGER,
    jackpot1_value  BIGINT,
    jackpot2_value  BIGINT,
    tier_winners    VARCHAR,
    source          VARCHAR,
    ingested_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (game, draw_id)
);
CREATE TABLE IF NOT EXISTS prizes (
    game          VARCHAR NOT NULL,
    draw_id       INTEGER NOT NULL,
    draw_date     DATE    NOT NULL,
    winners       VARCHAR NOT NULL,
    jackpot_pots  VARCHAR,
    source        VARCHAR,
    flags         VARCHAR,
    ingested_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (game, draw_id)
);
CREATE TABLE IF NOT EXISTS sync_log (
    game      VARCHAR,
    source    VARCHAR,
    fetched   INTEGER,
    inserted  INTEGER,
    rejected  INTEGER,
    run_at    TIMESTAMP DEFAULT current_timestamp
);
"""

_UPSERT = """
INSERT INTO draws (game, draw_id, draw_date, numbers, bonus, jackpot1_value, jackpot2_value, tier_winners, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (game, draw_id) DO UPDATE SET
    draw_date = excluded.draw_date,
    numbers = excluded.numbers,
    bonus = excluded.bonus,
    jackpot1_value = COALESCE(excluded.jackpot1_value, draws.jackpot1_value),
    jackpot2_value = COALESCE(excluded.jackpot2_value, draws.jackpot2_value),
    tier_winners = COALESCE(excluded.tier_winners, draws.tier_winners),
    source = excluded.source
"""

_SELECT = """
SELECT game, draw_id, draw_date, numbers, bonus, jackpot1_value, jackpot2_value, tier_winners, source
FROM draws WHERE game = ? ORDER BY draw_date, draw_id
"""


def _as_date(v: object) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


class DuckDBRepository(_HistoryMixin):
    """DuckDB-backed repository. One connection guarded by a lock (DuckDB connections
    are not safe for concurrent use from several threads)."""

    def __init__(self, db_path: Path | str, parquet_dir: Path | str | None = None) -> None:
        try:
            import duckdb  # imported lazily so the rest of the engine works without it
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise StorageError("duckdb is not installed: pip install duckdb") from exc
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self._lock = threading.Lock()
        self._con = duckdb.connect(self.db_path)
        for stmt in filter(str.strip, _SCHEMA.split(";")):
            self._con.execute(stmt)

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ----------------------------------------------------------------- writes
    def upsert(self, draws: Iterable[Draw]) -> int:
        rows = [
            (
                d.game.value,
                d.draw_id,
                d.draw_date,
                list(d.numbers),
                d.bonus,
                d.jackpot1_value,
                d.jackpot2_value,
                json.dumps(d.tier_winners) if d.tier_winners else None,
                d.source,
            )
            for d in draws
        ]
        if not rows:
            return 0
        with self._lock:
            try:
                self._con.execute("BEGIN TRANSACTION")
                self._con.executemany(_UPSERT, rows)
                self._con.execute("COMMIT")
            except Exception as exc:
                self._con.execute("ROLLBACK")
                raise StorageError(f"upsert failed: {exc}") from exc
        return len(rows)

    def log_sync(self, game: GameCode | str, source: str, fetched: int, inserted: int, rejected: int) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO sync_log (game, source, fetched, inserted, rejected) VALUES (?, ?, ?, ?, ?)",
                [get_game(game).code.value, source, fetched, inserted, rejected],
            )

    def export_parquet(self, game: GameCode | str) -> Path:
        """Write ``<parquet_dir>/<game>.parquet`` (overwrites)."""
        if self.parquet_dir is None:
            raise StorageError("parquet_dir not configured")
        code = get_game(game).code.value  # validated enum value -> safe to inline
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        target = (self.parquet_dir / f"{code}.parquet").resolve()
        path_sql = str(target).replace("'", "''")
        with self._lock:
            self._con.execute(
                f"COPY (SELECT * FROM draws WHERE game = '{code}' ORDER BY draw_id) "
                f"TO '{path_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        log.info("exported %s", target)
        return target

    # ------------------------------------------------------------------ reads
    def load(self, game: GameCode | str) -> list[Draw]:
        code = get_game(game).code.value
        with self._lock:
            rows = self._con.execute(_SELECT, [code]).fetchall()
        return [
            Draw(
                game=r[0],
                draw_id=int(r[1]),
                draw_date=_as_date(r[2]),
                numbers=tuple(int(x) for x in r[3]),
                bonus=int(r[4]) if r[4] is not None else None,
                jackpot1_value=r[5],
                jackpot2_value=r[6],
                tier_winners=json.loads(r[7]) if r[7] else None,
                source=r[8] or "unknown",
            )
            for r in rows
        ]

    def upsert_prizes(self, records: Iterable[PrizeRecord]) -> int:
        rows = [
            (
                r.game.value,
                r.draw_id,
                r.draw_date,
                json.dumps(r.winners),
                json.dumps(r.jackpot_pots) if r.jackpot_pots else None,
                r.source,
                json.dumps(list(r.flags)),
            )
            for r in records
        ]
        if not rows:
            return 0
        sql = (
            "INSERT INTO prizes (game, draw_id, draw_date, winners, jackpot_pots, source, flags) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (game, draw_id) DO UPDATE SET draw_date = excluded.draw_date, winners = excluded.winners, "
            "jackpot_pots = COALESCE(excluded.jackpot_pots, prizes.jackpot_pots), source = excluded.source, flags = excluded.flags"
        )
        with self._lock:
            try:
                self._con.execute("BEGIN TRANSACTION")
                self._con.executemany(sql, rows)
                self._con.execute("COMMIT")
            except Exception as exc:
                self._con.execute("ROLLBACK")
                raise StorageError(f"prize upsert failed: {exc}") from exc
        return len(rows)

    def load_prizes(self, game: GameCode | str) -> list[PrizeRecord]:
        code = get_game(game).code.value
        with self._lock:
            rows = self._con.execute(
                "SELECT game, draw_id, draw_date, winners, jackpot_pots, source, flags FROM prizes WHERE game = ? ORDER BY draw_id", [code]
            ).fetchall()
        return [
            PrizeRecord(
                game=r[0],
                draw_id=int(r[1]),
                draw_date=_as_date(r[2]),
                winners=json.loads(r[3]) if isinstance(r[3], str) else r[3],
                jackpot_pots=(json.loads(r[4]) if isinstance(r[4], str) else r[4]) if r[4] else None,
                source=r[5] or "unknown",
                flags=tuple(json.loads(r[6]) if isinstance(r[6], str) else (r[6] or ())),
            )
            for r in rows
        ]

    def max_draw_id(self, game: GameCode | str) -> int | None:
        with self._lock:
            row = self._con.execute("SELECT max(draw_id) FROM draws WHERE game = ?", [get_game(game).code.value]).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def count(self, game: GameCode | str) -> int:
        with self._lock:
            row = self._con.execute("SELECT count(*) FROM draws WHERE game = ?", [get_game(game).code.value]).fetchone()
        return int(row[0]) if row else 0
