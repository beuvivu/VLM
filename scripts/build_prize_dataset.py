#!/usr/bin/env python3
"""Rebuild the bundled seed data (draws + per-draw prize data) from public sources.

    python scripts/build_prize_dataset.py --vendor /tmp/vqe-sources

Clones (or reuses) three public repositories, merges and validates them, and writes:

* data/seed/{power645,power655,power535}.jsonl   draws (mirror format)
* data/seed/prizes_{mega645,power655,lotto535}.jsonl   reconciled prize records
* data/seed/PRIZE_DATA_REPORT.json   reconciliation report

Sources: vietvudanh/vietlott-data (draws), Compal123/vietlot-ai (draws + winners),
leoodz/vn-vietlott (Power 6/55 full prize history). All MIT-licensed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vietlott_engine.core.games import GAMES, GameCode  # noqa: E402
from vietlott_engine.core.models import Draw  # noqa: E402
from vietlott_engine.crawler.prize_sources import (  # noqa: E402
    COMPAL_FILES,
    parse_compal_winners,
    parse_leoodz_power,
    reconcile,
    records_to_jsonl,
)
from vietlott_engine.crawler.sources.mirror import MIRROR_FILES, JsonlFileSource  # noqa: E402

REPOS = {
    "vietvudanh": "https://github.com/vietvudanh/vietlott-data",
    "compal": "https://github.com/Compal123/vietlot-ai",
    "leoodz": "https://github.com/leoodz/vn-vietlott",
}


def ensure_repo(vendor: Path, key: str) -> Path:
    dest = vendor / key
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPOS[key], str(dest)], check=True)
    return dest


def merge_draws(game: GameCode, base: list[Draw], compal_text: str, leoodz_raw: list) -> tuple[list[Draw], dict]:
    spec = GAMES[game]
    by_id = {d.draw_id: d for d in base}
    added, repaired = 0, 0
    leo_by_date = {r.draw_date: r for r in leoodz_raw if not r.stale}
    for line in compal_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        res = [int(x) for x in row["result"]]
        did = int(row["id"])
        if did in by_id:
            continue
        try:
            d = Draw(
                game=game,
                draw_id=did,
                draw_date=date.fromisoformat(row["date"][:10]),
                numbers=tuple(res[: spec.pick]),
                bonus=res[spec.pick] if spec.has_bonus and len(res) > spec.pick else None,
                source="compal123",
            )
        except ValueError:
            # e.g. Power draw #944 lacks its bonus ball in one mirror: repair from leoodz
            leo = leo_by_date.get(date.fromisoformat(row["date"][:10]))
            if leo is None or tuple(sorted(res[: spec.pick])) != leo.numbers:
                continue
            d = Draw(game=game, draw_id=did, draw_date=leo.draw_date, numbers=leo.numbers, bonus=leo.bonus, source="compal123+leoodz")
            repaired += 1
        by_id[did] = d
        added += 1
    return sorted(by_id.values(), key=lambda d: d.draw_id), {"added": added, "repaired": repaired}


def to_mirror_line(d: Draw) -> str:
    result = list(d.numbers) + ([d.bonus] if d.bonus is not None else [])
    return json.dumps({"date": d.draw_date.isoformat(), "id": f"{d.draw_id:05d}", "result": result, "source": d.source})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", type=Path, required=True, help="directory holding (or receiving) the source clones")
    ap.add_argument("--out", type=Path, default=Path("data/seed"))
    a = ap.parse_args()
    a.vendor.mkdir(parents=True, exist_ok=True)
    roots = {k: ensure_repo(a.vendor, k) for k in REPOS}
    leoodz_raw = parse_leoodz_power((roots["leoodz"] / "results" / "power6x55.json").read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for game in (GameCode.MEGA_645, GameCode.POWER_655, GameCode.LOTTO_535):
        spec = GAMES[game]
        base = asyncio.run(JsonlFileSource(roots["vietvudanh"] / "data").fetch(spec))
        compal_text = (roots["compal"] / "data" / COMPAL_FILES[game]).read_text(encoding="utf-8")
        leo = leoodz_raw if game == GameCode.POWER_655 else []
        draws, merge_info = merge_draws(game, base.draws, compal_text, leo)
        # repair id gaps (e.g. Power #944: the mirror holds a corrupt record) from leoodz:
        # a non-stale record dated strictly between the two neighbouring draws
        draws.sort(key=lambda d: d.draw_id)
        have = {d.draw_id: d for d in draws}
        for did in range(draws[0].draw_id + 1, draws[-1].draw_id):
            if did in have or did - 1 not in have or did + 1 not in have:
                continue
            lo, hi = have[did - 1].draw_date, have[did + 1].draw_date
            hits = [r for r in leo if lo < r.draw_date < hi and not r.stale]
            if len(hits) == 1:
                r = hits[0]
                draws.append(Draw(game=game, draw_id=did, draw_date=r.draw_date, numbers=r.numbers, bonus=r.bonus, source="leoodz-repair"))
                merge_info["repaired"] += 1
        draws.sort(key=lambda d: d.draw_id)
        (a.out / MIRROR_FILES[game]).write_text("\n".join(to_mirror_line(d) for d in draws) + "\n", encoding="utf-8")
        records, rep = reconcile(spec, draws, parse_compal_winners(compal_text, game), leo)
        (a.out / f"prizes_{game.value}.jsonl").write_text(records_to_jsonl(records), encoding="utf-8")
        report[game.value] = {"draws": len(draws), "first_draw": draws[0].draw_date.isoformat(), "last_draw": draws[-1].draw_date.isoformat(), **merge_info, **asdict(rep)}
        print(json.dumps(report[game.value], ensure_ascii=False))
    (a.out / "PRIZE_DATA_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
