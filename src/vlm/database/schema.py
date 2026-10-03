"""Transactional storage. Date-only archives use midnight with time_precision=day.

Missing financial fields remain NULL. Midnight on a date-only record does not
assert the actual draw time. Conflicting results never silently overwrite data.
"""
from __future__ import annotations
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Integer, JSON, String, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

VN = timezone(timedelta(hours=7))


class GameType(str, Enum):
    MEGA645 = "mega645"
    POWER655 = "power655"
    LOTTO535 = "lotto535"
    MAX3D = "max3d"
    MAX3DPLUS = "max3dplus"
    MAX3DPRO = "max3dpro"
    KENO = "keno"
    BINGO18 = "bingo18"
    MAX4D = "max4d"


TABLE_BY_GAME = {g.value: "draws_" + g.value for g in GameType}
WIDTH = {"mega645": 6, "power655": 6, "lotto535": 5, "keno": 20, "bingo18": 3,
         "max3d": 20, "max3dplus": 20, "max3dpro": 20, "max4d": 6}
POOL = {"mega645": 45, "power655": 55, "lotto535": 35, "keno": 80, "bingo18": 6}


def game_code(game: str | GameType) -> str:
    key = str(game.value if isinstance(game, Enum) else game).lower().replace("_", "").replace(" ", "").replace("+", "plus")
    key = {"mega": "mega645", "power645": "mega645", "power": "power655", "power535": "lotto535", "lotto": "lotto535"}.get(key, key)
    return GameType(key).value


class DrawRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    game_type: GameType
    draw_id: int = Field(ge=1, strict=True)
    draw_date: datetime
    winning_numbers: tuple[int | str, ...]
    time_precision: Literal["second", "day"] = "second"
    bonus_number: int | None = Field(default=None, ge=1, strict=True)
    jackpot1_value: int | None = Field(default=None, ge=0, strict=True)
    jackpot2_value: int | None = Field(default=None, ge=0, strict=True)
    jackpot1_winners: int | None = Field(default=None, ge=0, strict=True)
    jackpot2_winners: int | None = Field(default=None, ge=0, strict=True)
    sub_prizes_json: dict[str, Any] | None = None
    source: str = "unknown"
    source_url: str | None = None
    source_sha256: str | None = None

    @field_validator("game_type", mode="before")
    @classmethod
    def _game(cls, value: Any) -> str:
        return game_code(value)

    @field_validator("draw_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> datetime:
        if isinstance(value, str):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?", value):
                raise ValueError("draw_date requires YYYY-MM-DD HH:mm:ss (optional ISO timezone)")
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise ValueError("draw_date must include a time; use from_legacy for date-only records")
        return (value if value.tzinfo else value.replace(tzinfo=VN)).astimezone(VN)

    @field_validator("winning_numbers", mode="before")
    @classmethod
    def _strict_numbers(cls, value: Any) -> tuple[int | str, ...]:
        if not isinstance(value, (list, tuple)) or any(type(x) not in (str, int) for x in value):
            raise ValueError("numbers must be integers or padded digit strings")
        return tuple(value)

    @field_validator("sub_prizes_json")
    @classmethod
    def _json(cls, value: Any) -> Any:
        if value is not None:
            json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def _validate_game(self) -> DrawRecord:
        g, nums = self.game_type.value, self.winning_numbers
        if len(nums) != WIDTH[g]:
            raise ValueError(f"{g} requires {WIDTH[g]} result numbers")
        if g.startswith("max"):
            width = 4 if g == "max4d" else 3
            if any((isinstance(x, str) and not re.fullmatch(rf"\d{{{width}}}", x))
                   or not 0 <= int(x) < 10**width for x in nums):
                raise ValueError(f"{g} requires {width}-digit numbers")
            object.__setattr__(self, "winning_numbers", tuple(f"{int(x):0{width}d}" for x in nums))
        else:
            if any(type(x) is not int or not 1 <= x <= POOL[g] for x in nums):
                raise ValueError(f"{g} numbers outside 1..{POOL[g]}")
            if g != "bingo18":
                if len(set(nums)) != len(nums):
                    raise ValueError("duplicate winning numbers")
                object.__setattr__(self, "winning_numbers", tuple(sorted(nums)))
        if g in ("power655", "lotto535"):
            upper = 55 if g == "power655" else 12
            if self.bonus_number is None or self.bonus_number > upper:
                raise ValueError(f"{g} requires bonus in 1..{upper}")
            if g == "power655" and self.bonus_number in nums:
                raise ValueError("Power bonus must differ from all main numbers")
        elif self.bonus_number is not None:
            raise ValueError(f"{g} has no bonus number")
        if self.time_precision == "day" and self.draw_date.time().isoformat() != "00:00:00":
            raise ValueError("date-only records must be stored at 00:00:00")
        return self

    @classmethod
    def from_legacy(cls, game: str, row: dict[str, Any]) -> DrawRecord:
        """Adapt seed/mirror/AjaxPro/canonical records without inventing values."""
        g = game_code(game)
        for field in ("game", "game_type", "product"):
            if row.get(field) is not None and game_code(row[field]) != g:
                # Max 3D+ uses the published Max 3D draw, with different tickets.
                if not (g == "max3dplus" and game_code(row[field]) == "max3d"):
                    raise ValueError("source row declares a different product")
        if row.get("draw_status", "confirmed") != "confirmed":
            raise ValueError("unconfirmed draws must be excluded, not settled")
        raw = row.get("winning_numbers", row.get("result"))
        bonus = row.get("bonus_number", row.get("bonus"))
        if isinstance(raw, dict):
            if "main_numbers" in raw:
                bonus_list = raw.get("bonus_numbers", [])
                bonus = bonus_list[0] if bonus_list else bonus
                raw = raw["main_numbers"]
            elif "tiers" in raw:
                tiers = raw["tiers"]
                keys = ["first", "second", "third"] if g == "max4d" else ["special", "first", "second", "third"]
                if isinstance(tiers, list):
                    tiers = {t["code"]: t["numbers"] for t in tiers}
                expected = [1, 2, 3] if g == "max4d" else [2, 4, 6, 8]
                if any(len(tiers[k]) != n for k, n in zip(keys, expected)):
                    raise ValueError("invalid prize group sizes")
                raw = [x for k in keys for x in tiers[k]]
            else:
                raw = [x for k in ["Giải Đặc biệt", "Giải Nhất", "Giải Nhì", "Giải ba"] for x in raw[k]]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("missing winning_numbers/result")
        if g in ("power655", "lotto535") and len(raw) == WIDTH[g] + 1:
            raw, bonus = raw[:-1], raw[-1]
        when = str(row.get("draw_date", row.get("date", "")))
        precision = row.get("time_precision", "day" if len(when) == 10 else "second")
        if precision == "day" and len(when) == 10:
            when += " 00:00:00"
        pots, winners = dict(row.get("jackpot_pots") or {}), dict(row.get("winners") or {})
        if any(type(n) is not int or n < 0 for n in [*pots.values(), *winners.values()]):
            raise ValueError("financial amounts/counts must be nonnegative integers")
        # Canonical/detail-page amounts are per winning unit when winners > 0.
        # Reuse the engine's tier mapping; never treat that amount as the pool.
        from vietlott_engine.crawler.official_data import CANONICAL_GAMES, tier_of
        if g in CANONICAL_GAMES:
            for prize in row.get("prizes") or []:
                tier = tier_of(CANONICAL_GAMES[g], prize)
                count = prize.get("winner_count")
                if tier is None or count is None:
                    continue
                if type(count) is not int or count < 0:
                    raise ValueError("invalid prize winner count")
                if tier in winners and winners[tier] != count:
                    raise ValueError("conflicting winner counts")
                winners[tier] = count
                amount = prize.get("jackpot_vnd")
                if amount is not None:
                    if type(amount) is not int or amount < 0:
                        raise ValueError("invalid jackpot amount")
                    pool = amount * max(1, count)
                    if tier in pots and pots[tier] != pool:
                        raise ValueError("conflicting jackpot pool")
                    pots[tier] = pool
        sub = row.get("sub_prizes_json")
        if sub is None and (winners or row.get("prizes")):
            sub = {"winners": winners, "prizes": row.get("prizes", [])}
        def finance_column(key: str, values: dict[str, int], tier: str) -> int | None:
            direct, nested = row.get(key), values.get(tier)
            if direct is not None and nested is not None and direct != nested:
                raise ValueError(f"conflicting {key} within source row")
            return direct if direct is not None else nested
        return cls(
            game_type=g, draw_id=int(str(row.get("draw_id", row.get("id"))).lstrip("#")), draw_date=when,
            time_precision=precision, winning_numbers=raw, bonus_number=bonus,
            jackpot1_value=finance_column("jackpot1_value",pots,"jackpot1"),
            jackpot2_value=finance_column("jackpot2_value",pots,"jackpot2"),
            jackpot1_winners=finance_column("jackpot1_winners",winners,"jackpot1"),
            jackpot2_winners=finance_column("jackpot2_winners",winners,"jackpot2"),
            sub_prizes_json=sub, source=row.get("source", row.get("src", "unknown")),
            source_url=row.get("source_url"), source_sha256=row.get("source_sha256"),
        )


class Base(DeclarativeBase):
    pass


class _DrawColumns:
    game_type: Mapped[str] = mapped_column(String(16), primary_key=True)
    draw_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    draw_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    winning_numbers: Mapped[list] = mapped_column(JSON, nullable=False)
    time_precision: Mapped[str] = mapped_column(String(8), nullable=False)
    bonus_number: Mapped[int | None] = mapped_column(Integer)
    jackpot1_value: Mapped[int | None] = mapped_column(BigInteger)
    jackpot2_value: Mapped[int | None] = mapped_column(BigInteger)
    jackpot1_winners: Mapped[int | None] = mapped_column(BigInteger)
    jackpot2_winners: Mapped[int | None] = mapped_column(BigInteger)
    sub_prizes_json: Mapped[dict | None] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String)
    source_sha256: Mapped[str | None] = mapped_column(String(64))


ORM_MODELS = {g: type("Draws" + g.title(), (_DrawColumns, Base), {
    "__tablename__": table,
    "__table_args__": (CheckConstraint("draw_id > 0"), CheckConstraint(f"game_type = '{g}'")),
}) for g, table in TABLE_BY_GAME.items()}
DrawsMega645, DrawsPower655 = ORM_MODELS["mega645"], ORM_MODELS["power655"]
DrawsMax3d, DrawsKeno, DrawsBingo18 = ORM_MODELS["max3d"], ORM_MODELS["keno"], ORM_MODELS["bingo18"]


class DataConflict(ValueError):
    """A source disagrees with an existing result."""


def _merge_metadata(old: Any, new: Any) -> Any:
    """Fill nested unknown values; retain observed data on partial updates."""
    if new is None or new == {} or new == []:
        return old
    if old is None or old == {} or old == []:
        return new
    if isinstance(old, dict) and isinstance(new, dict):
        return {key: _merge_metadata(old.get(key), value) for key, value in new.items()} | {
            key: value for key, value in old.items() if key not in new}
    if isinstance(old, list) and isinstance(new, list) and all(isinstance(x, dict) and x.get("code") for x in old + new):
        indexed = {x["code"]: x for x in old}
        for item in new:
            indexed[item["code"]] = _merge_metadata(indexed.get(item["code"]), item)
        return list(indexed.values())
    if type(old) in (int, float) and type(new) in (int, float) and old != new:
        raise DataConflict("conflicting observed financial metadata")
    return new


def _merge(old: DrawRecord, new: DrawRecord) -> DrawRecord:
    if (old.winning_numbers, old.bonus_number) != (new.winning_numbers, new.bonus_number):
        raise DataConflict(f"conflicting numbers: {old.game_type.value}/{old.draw_id}")
    if old.draw_date.date() != new.draw_date.date() or (
        old.time_precision == new.time_precision == "second" and old.draw_date != new.draw_date
    ):
        raise DataConflict(f"conflicting draw date: {old.game_type.value}/{old.draw_id}")
    data = old.model_dump()
    for key, value in new.model_dump().items():
        if value is not None and key not in {"source", "source_url", "source_sha256", "draw_date", "time_precision"}:
            if key.startswith("jackpot") and data[key] is not None and data[key] != value:
                raise DataConflict(f"conflicting {key}: {old.game_type.value}/{old.draw_id}")
            data[key] = _merge_metadata(data[key], value) if key == "sub_prizes_json" else value
    if new.time_precision == "second":
        data.update(draw_date=new.draw_date, time_precision="second")
    if new.source != "unknown" and (old.source == "unknown" or new.source.startswith("vietlott")):
        data.update(source=new.source, source_url=new.source_url or old.source_url,
                    source_sha256=new.source_sha256 or old.source_sha256)
    return DrawRecord.model_validate(data)


def _validated(records: Iterable[DrawRecord]) -> list[DrawRecord]:
    return [DrawRecord.model_validate(r.model_dump()) for r in records]


def _values(r: DrawRecord) -> dict[str, Any]:
    return r.model_dump() | {"game_type": r.game_type.value, "winning_numbers": list(r.winning_numbers)}


class _ParquetMixin:
    def export_parquet(self, path: Path | str) -> None:
        """Export queryable flat columns, using DuckDB without pandas/pyarrow."""
        import duckdb
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with duckdb.connect() as conn:
            _duck_create(conn, "all_draws")
            for game in GameType:
                for r in self.load(game.value):
                    _duck_insert(conn, "all_draws", r)
            escaped = str(temporary).replace("'", "''")
            conn.execute(f"COPY all_draws TO '{escaped}' (FORMAT PARQUET)")
        temporary.replace(target)

    def import_parquet(self, path: Path | str) -> int:
        import duckdb
        with duckdb.connect() as conn:
            cursor = conn.execute("SELECT * FROM read_parquet(?)", [str(path)])
            names = [c[0] for c in cursor.description]
            rows = [DrawRecord.model_validate(_decode(dict(zip(names, row)))) for row in cursor.fetchall()]
        return self.upsert(rows)


class SQLRepository(_ParquetMixin):
    def __init__(self, url: str = "sqlite:///vlm.sqlite") -> None:
        self.engine = create_engine(url)
        if self.engine.dialect.name not in ("sqlite", "postgresql"):
            self.engine.dispose()
            raise ValueError("use SQLRepository for SQLite/PostgreSQL; DuckRepository for DuckDB")
        self._lock = threading.RLock()
        Base.metadata.create_all(self.engine)

    def upsert(self, records: Iterable[DrawRecord]) -> int:
        rows, inserted = _validated(records), 0
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        insert = sqlite_insert if self.engine.dialect.name == "sqlite" else pg_insert
        with self._lock, self.engine.begin() as conn:
            for r in rows:
                table = ORM_MODELS[r.game_type.value].__table__
                condition = (table.c.game_type == r.game_type.value) & (table.c.draw_id == r.draw_id)
                stmt = insert(table).values(**_values(r)).on_conflict_do_nothing().returning(table.c.draw_id)
                if conn.execute(stmt).first() is not None:
                    inserted += 1
                    continue
                query = select(table).where(condition)
                if self.engine.dialect.name == "postgresql":
                    query = query.with_for_update()
                old = DrawRecord.model_validate(dict(conn.execute(query).mappings().one()))
                conn.execute(update(table).where(condition).values(**_values(_merge(old, r))))
        return inserted

    def load(self, game: str) -> list[DrawRecord]:
        table = ORM_MODELS[game_code(game)].__table__
        with self._lock, self.engine.connect() as conn:
            return [DrawRecord.model_validate(dict(row)) for row in conn.execute(select(table).order_by(table.c.draw_id)).mappings()]

    def get(self, game: str, draw_id: int) -> DrawRecord | None:
        table = ORM_MODELS[game_code(game)].__table__
        with self._lock, self.engine.connect() as conn:
            row = conn.execute(select(table).where(table.c.draw_id == draw_id)).mappings().first()
            return DrawRecord.model_validate(dict(row)) if row else None

    def close(self) -> None:
        self.engine.dispose()


_DUCK_COLUMNS = (
    "game_type VARCHAR NOT NULL, draw_id BIGINT NOT NULL CHECK(draw_id > 0), draw_date VARCHAR NOT NULL, "
    "winning_numbers JSON NOT NULL, time_precision VARCHAR NOT NULL, bonus_number INTEGER, "
    "jackpot1_value BIGINT, jackpot2_value BIGINT, jackpot1_winners BIGINT, jackpot2_winners BIGINT, "
    "sub_prizes_json JSON, source VARCHAR NOT NULL, source_url VARCHAR, source_sha256 VARCHAR, PRIMARY KEY(game_type, draw_id)"
)
_NAMES = list(DrawRecord.model_fields)


def _duck_create(conn: Any, table: str) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({_DUCK_COLUMNS})")


def _duck_values(r: DrawRecord) -> list[Any]:
    data = r.model_dump(mode="json")
    for name in ("winning_numbers", "sub_prizes_json"):
        data[name] = json.dumps(data[name]) if data[name] is not None else None
    return [data[k] for k in _NAMES]


def _duck_insert(conn: Any, table: str, r: DrawRecord) -> None:
    conn.execute(f"INSERT INTO {table} ({','.join(_NAMES)}) VALUES ({','.join('?' for _ in _NAMES)})", _duck_values(r))


def _decode(data: dict[str, Any]) -> dict[str, Any]:
    for name in ("winning_numbers", "sub_prizes_json"):
        if isinstance(data.get(name), str):
            data[name] = json.loads(data[name])
    return data


class DuckRepository(_ParquetMixin):
    def __init__(self, path: Path | str = ":memory:") -> None:
        import duckdb
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(path))
        self._lock = threading.RLock()
        for table in TABLE_BY_GAME.values():
            _duck_create(self.conn, table)

    def upsert(self, records: Iterable[DrawRecord]) -> int:
        rows, inserted = _validated(records), 0
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                for r in rows:
                    table = TABLE_BY_GAME[r.game_type.value]
                    old = self.get(r.game_type.value, r.draw_id)
                    if old:
                        r = _merge(old, r)
                        self.conn.execute(f"DELETE FROM {table} WHERE game_type=? AND draw_id=?", [r.game_type.value, r.draw_id])
                    else:
                        inserted += 1
                    _duck_insert(self.conn, table, r)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return inserted

    def get(self, game: str, draw_id: int) -> DrawRecord | None:
        table = TABLE_BY_GAME[game_code(game)]
        with self._lock:
            cursor = self.conn.execute(f"SELECT * FROM {table} WHERE draw_id=?", [draw_id])
            names = [c[0] for c in cursor.description]
            row = cursor.fetchone()
            return DrawRecord.model_validate(_decode(dict(zip(names, row)))) if row else None

    def load(self, game: str) -> list[DrawRecord]:
        table = TABLE_BY_GAME[game_code(game)]
        with self._lock:
            cursor = self.conn.execute(f"SELECT * FROM {table} ORDER BY draw_id")
            names = [c[0] for c in cursor.description]
            return [DrawRecord.model_validate(_decode(dict(zip(names, row)))) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
