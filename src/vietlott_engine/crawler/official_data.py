"""Official vietlott.vn draw records ("canonical" schema 1.0) → engine objects.

A canonical record is what vietlott.vn's draw-detail page says about one draw: result,
the prize table (winners per tier, jackpot value), the page URL, its SHA-256 and the
official PDF. The public dataset ``pqminh-4/vietlott-data`` (MIT code) publishes these
records for Mega 6/45, Power 6/55, Lotto 5/35, Max 3D and Max 3D Pro, collected from a
machine in Vietnam (vietlott.vn rejects non-Vietnamese IPs); ``crawler.sources.
vietlott_official`` produces the same records when run from Vietnam.

Jackpot value on a detail page is per winning ticket when there are winners, so the pot
of a draw is value × max(1, winners) — the convention used everywhere in the engine.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import date

from vietlott_engine.core.games import GameCode, get_game
from vietlott_engine.core.models import Draw
from vietlott_engine.core.prizes import PrizeRecord

CANONICAL_GAMES = {"mega645": GameCode.MEGA_645, "power655": GameCode.POWER_655, "lotto535": GameCode.LOTTO_535}

TIER_BY_CODE: dict[GameCode, dict[str, str]] = {
    GameCode.MEGA_645: {"jackpot": "jackpot1", "giai-nhat": "first", "giai-nhi": "second", "giai-ba": "third"},
    GameCode.POWER_655: {"jackpot-1": "jackpot1", "jackpot-2": "jackpot2", "giai-nhat": "first", "giai-nhi": "second", "giai-ba": "third"},
    GameCode.LOTTO_535: {
        "giai-doc-dac": "jackpot1",
        "giai-nhat": "first",
        "giai-nhi": "second",
        "giai-ba": "third",
        "giai-tu": "fourth",
        "giai-nam": "fifth",
        "giai-khuyen-khich": "consolation",
    },
}


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower().replace("đ", "d"))
    return "".join(c for c in t if not unicodedata.combining(c))


def slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")


@dataclass
class CanonicalImport:
    draws: list[Draw] = field(default_factory=list)
    prizes: list[PrizeRecord] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    with_pdf: int = 0


def tier_of(game: GameCode, prize: dict) -> str | None:
    code = prize.get("code") or slug(prize.get("name", ""))
    return TIER_BY_CODE[game].get(code)


def import_canonical(rows: list[dict], game: GameCode) -> CanonicalImport:
    spec = get_game(game)
    out = CanonicalImport()
    names = {t.name for t in spec.tiers}
    for r in rows:
        try:
            res = r["result"]
            main = [int(x) for x in res["main_numbers"]]
            bonus = [int(x) for x in res.get("bonus_numbers") or []]
            did = int(str(r["draw_id"]).lstrip("#"))
            d = date.fromisoformat(r["draw_date"][:10])
            origin = r.get("data_source")
            label = "vietlott.vn" if origin in (None, "official_vietlott") else origin
            draw = Draw(game=game, draw_id=did, draw_date=d, numbers=tuple(main), bonus=bonus[0] if spec.has_bonus and bonus else None, source=label)
        except (KeyError, ValueError, TypeError) as exc:
            out.rejected.append({"draw_id": r.get("draw_id"), "error": str(exc).splitlines()[0]})
            continue
        out.draws.append(draw)
        out.with_pdf += bool(r.get("source_pdf_url"))
        winners: dict[str, int] = {}
        pots: dict[str, int] = {}
        for p in r.get("prizes") or []:
            tier = tier_of(game, p)
            if tier is None or p.get("winner_count") is None:
                continue
            winners[tier] = int(p["winner_count"])
            if p.get("jackpot_vnd") is not None:
                pots[tier] = int(p["jackpot_vnd"]) * max(1, int(p["winner_count"]))
        if set(winners) == names:
            jps = [t.name for t in spec.tiers if t.is_jackpot]
            out.prizes.append(
                PrizeRecord(
                    game=game,
                    draw_id=did,
                    draw_date=d,
                    winners=winners,
                    jackpot_pots=pots if all(j in pots for j in jps) else None,
                    source=label,
                    flags=("official_detail_page",) if label == "vietlott.vn" else ("secondary_source",),
                )
            )
    return out


def draws_to_seed_rows(draws: list[Draw]) -> list[dict]:
    """Mirror JSONL format used by ``JsonlFileSource`` (bonus appended to the numbers)."""
    rows = []
    for d in sorted(draws, key=lambda x: x.draw_id):
        res = list(d.numbers) + ([d.bonus] if d.bonus is not None else [])
        rows.append({"date": d.draw_date.isoformat(), "id": f"{d.draw_id:05d}", "result": res, "source": d.source})
    return rows
