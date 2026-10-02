"""Per-draw prize data: parsers for public sources and a reconciliation step.

Sources (both public GitHub repositories, MIT-licensed, auto-updated by their owners):

* **leoodz/vn-vietlott** — ``results/power6x55.json``: every Power 6/55 draw since
  01/08/2017, scraped from minhngoc.net.vn: winning numbers, winner counts and prize
  value per tier (jackpot values *per winner* when shared).
* **Compal123/vietlot-ai** — ``data/{power645,power655,power535}.jsonl``: draws with a
  ``winners`` block (counts per tier) for Mega 6/45 (since 09/2025), Power 6/55
  (since 09/2025) and Lotto 5/35 (since launch, 06/2025).

Data-quality issues found and handled (``reconcile_power``):

1. leoodz contains records for dates with no draw (Tết, 2020/2021 COVID pauses): the
   page then shows the previous draw. Dropped by joining on the official draw list.
2. Some daily updates captured the previous draw's prize table (records shifted by one
   draw for weeks in 2025–26). Detected by fingerprint (counts + jackpot values equal
   to the previous record) and repaired by matching each draw's winner counts —
   taken from Compal123 when available — to the record that carries them.
3. Jackpot values with several winners are per-winner amounts → converted to pots.
4. Amount strings concatenated with older values (``"31,024,813,350đ30,528,…"``):
   only the first amount is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from vietlott_engine.core.games import GameCode, GameSpec
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.models import Draw
from vietlott_engine.core.prizes import PrizeRecord

log = get_logger(__name__)

COMPAL_TIER_NAMES: dict[GameCode, dict[str, str]] = {
    GameCode.MEGA_645: {"Jackpot": "jackpot1", "Giải nhất": "first", "Giải nhì": "second", "Giải ba": "third"},
    GameCode.POWER_655: {
        "Jackpot 1": "jackpot1",
        "Jackpot 2": "jackpot2",
        "Giải nhất": "first",
        "Giải nhì": "second",
        "Giải ba": "third",
    },
    GameCode.LOTTO_535: {
        "Độc đắc": "jackpot1",
        "Giải nhất": "first",
        "Giải nhì": "second",
        "Giải ba": "third",
        "Giải tư": "fourth",
        "Giải năm": "fifth",
        "Giải KK": "consolation",
    },
}
COMPAL_FILES = {GameCode.MEGA_645: "power645.jsonl", GameCode.POWER_655: "power655.jsonl", GameCode.LOTTO_535: "power535.jsonl"}
LEOODZ_TIER_NAMES = {"jackpot1": "jackpot1", "jackpot2": "jackpot2", "giai_nhat": "first", "giai_nhi": "second", "giai_ba": "third"}

_AMOUNT = re.compile(r"([\d,.]+)\s*đ")


def parse_amount(text: str | None) -> int | None:
    """First VND amount in a (possibly concatenated) string: ``"31,024,813,350đ30,5…"`` → 31024813350."""
    if not text:
        return None
    m = _AMOUNT.match(text.strip())
    return int(m.group(1).replace(",", "").replace(".", "")) if m else None


@dataclass
class RawPrize:
    draw_date: date
    numbers: tuple[int, ...]
    bonus: int | None
    winners: dict[str, int]
    amounts: dict[str, int | None]  # per-winner value for jackpots
    source: str
    stale: bool = False
    draw_id: int | None = None


@dataclass
class ReconcileReport:
    game: str
    draws: int
    with_winners: int
    with_pots: int
    dropped_non_draw_dates: int = 0
    stale_records: int = 0
    realigned: int = 0
    number_mismatches: list[int] = field(default_factory=list)
    sources: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------- parsers
def parse_leoodz_power(text: str) -> list[RawPrize]:
    out: list[RawPrize] = []
    prev_fp = None
    for row in json.loads(text):
        prizes = row.get("prizes") or {}
        winners = {LEOODZ_TIER_NAMES[k]: int(v["count"]) for k, v in prizes.items() if k in LEOODZ_TIER_NAMES and v.get("count") is not None}
        amounts = {LEOODZ_TIER_NAMES[k]: parse_amount(v.get("amount")) for k, v in prizes.items() if k in LEOODZ_TIER_NAMES}
        fp = (tuple(sorted(winners.items())), amounts.get("jackpot1"), amounts.get("jackpot2"))
        out.append(
            RawPrize(
                draw_date=datetime.strptime(row["date"], "%d-%m-%Y").date(),
                numbers=tuple(sorted(int(x) for x in row["numbers"])),
                bonus=int(row["bonus"]) if row.get("bonus") not in (None, "") else None,
                winners=winners,
                amounts=amounts,
                source="leoodz",
                stale=fp == prev_fp,
            )
        )
        prev_fp = fp
    return out


def parse_compal_winners(text: str, game: GameCode) -> list[RawPrize]:
    names = COMPAL_TIER_NAMES[game]
    out: list[RawPrize] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        w = row.get("winners")
        if not w:
            continue
        try:
            winners = {names[k]: int(v) for k, v in w.items() if k in names}
        except (TypeError, ValueError):
            continue
        res = [int(x) for x in row.get("result", [])]
        out.append(
            RawPrize(
                draw_date=date.fromisoformat(row["date"][:10]),
                numbers=tuple(sorted(res[: len(res) - 1] if game != GameCode.MEGA_645 else res)),
                bonus=res[-1] if game != GameCode.MEGA_645 and res else None,
                winners=winners,
                amounts={},
                source="compal123",
                draw_id=int(row["id"]),
            )
        )
    return out


# ------------------------------------------------------------------ reconcile
def _pots(spec: GameSpec, raw: RawPrize, winners: dict[str, int]) -> dict[str, int] | None:
    pots: dict[str, int] = {}
    for t in spec.tiers:
        if not t.is_jackpot:
            continue
        per = raw.amounts.get(t.name)
        if per is None:
            return None
        pots[t.name] = int(per * max(1, winners.get(t.name, 0)))
    return pots


def reconcile(spec: GameSpec, draws: list[Draw], compal: list[RawPrize], leoodz: list[RawPrize] | None = None) -> tuple[list[PrizeRecord], ReconcileReport]:
    """Merge sources into one validated ``PrizeRecord`` per official draw.

    Winner counts: Compal123 (keyed by draw id, verified against the winning numbers)
    first, then the leoodz record of the same date if it is not stale. Jackpot pots:
    the leoodz record (within ±3 positions of the date) whose winner counts equal the
    chosen counts — this repairs the one-draw shifts in leoodz.
    """
    leoodz = leoodz or []
    names = {t.name for t in spec.tiers}
    by_id = {d.draw_id: d for d in draws}
    comp = {r.draw_id: r for r in compal if r.draw_id in by_id}
    rep = ReconcileReport(game=spec.code.value, draws=len(draws), with_winners=0, with_pots=0)

    draw_dates = {d.draw_date for d in draws}
    leo = [r for r in leoodz if r.draw_date in draw_dates]
    rep.dropped_non_draw_dates = len(leoodz) - len(leo)
    rep.stale_records = sum(r.stale for r in leo)
    leo_pos = {r.draw_date: i for i, r in enumerate(leo)}
    by_counts: dict[tuple, list[int]] = {}
    for i, r in enumerate(leo):
        by_counts.setdefault(tuple(sorted(r.winners.items())), []).append(i)

    records: list[PrizeRecord] = []
    for d in sorted(draws, key=lambda x: x.draw_id):
        winners: dict[str, int] | None = None
        source = None
        c = comp.get(d.draw_id)
        if c is not None and set(c.winners) == names:
            if c.numbers != d.numbers:
                rep.number_mismatches.append(d.draw_id)
            else:
                winners, source = c.winners, c.source
        i = leo_pos.get(d.draw_date)
        if winners is None and i is not None and not leo[i].stale and set(leo[i].winners) == names:
            if leo[i].numbers == d.numbers:
                winners, source = leo[i].winners, leo[i].source
            else:
                rep.number_mismatches.append(d.draw_id)
        if winners is None:
            continue
        pots = None
        if i is not None:
            cands = [j for j in by_counts.get(tuple(sorted(winners.items())), []) if abs(j - i) <= 3]
            if cands:
                j = min(cands, key=lambda j: abs(j - i))
                pots = _pots(spec, leo[j], winners)
                rep.realigned += int(j != i)
        flags: list[str] = []
        if source == "compal123" and pots is not None:
            flags.append("pots_from_leoodz")
        records.append(
            PrizeRecord(game=spec.code, draw_id=d.draw_id, draw_date=d.draw_date, winners=winners, jackpot_pots=pots, source=source or "unknown", flags=tuple(flags))
        )
        rep.sources[source or "unknown"] = rep.sources.get(source or "unknown", 0) + 1
    rep.with_winners = len(records)
    rep.with_pots = sum(r.jackpot_pots is not None for r in records)
    log.info("%s prize reconciliation: %s", spec.code.value, rep)
    return records, rep


def records_to_jsonl(records: list[PrizeRecord]) -> str:
    return "\n".join(r.model_dump_json() for r in records) + "\n"


def records_from_jsonl(text: str) -> list[PrizeRecord]:
    return [PrizeRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]
