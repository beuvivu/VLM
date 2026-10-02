#!/usr/bin/env python3
"""Rebuild the bundled seed data from official vietlott.vn records.

    python scripts/build_official_dataset.py --vendor /tmp/vqe-sources

* Mega 6/45, Power 6/55, Lotto 5/35, Max 3D, Max 3D Pro: ``pqminh-4/vietlott-data``
  canonical records — vietlott.vn draw-detail pages (result, winners per tier, jackpot
  value, page SHA-256, official PDF), collected from Vietnam.
* Keno, Bingo18: ``vietvudanh/vietlott-data`` (vietlott.vn results lists).
* All 8 products (incl. Max 4D) and the not-confirmed draws: ``NhanAZ-Data/vietlott-research``
  (community archive; fills gaps — Keno from 2019, Lotto draws the canonical file lacks).
* Optional ``--v130-dir``: the snapshot of Vietlott Quant Engine 1.3.0, cross-checked and used
  as the last fill layer.

Official records always win; lower layers only add draw ids the higher ones lack, and every
disagreement is listed. Writes ``data/seed/`` (draws, prize records, product histories,
``exclusions.json``, ``keno_rules_official.json``), ``OFFICIAL_DATA_REPORT.json`` (comparison
with the seed it replaces) and ``DATA_MERGE_REPORT.json`` (per-source overlap and conflicts).
Run it on a machine in Vietnam after ``vietlott products sync`` to use your own
crawl instead (``--canonical-dir``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.core.games import GameCode  # noqa: E402
from collections import Counter  # noqa: E402

from vietlott_engine.core.products import DIGIT_PRODUCTS, EXCLUSIONS_FILE, ProductCode, SEED_FILES, drop_ids, parse_product_rows, read_jsonl, write_jsonl  # noqa: E402
from vietlott_engine.crawler.official_data import CANONICAL_GAMES, draws_to_seed_rows, import_canonical  # noqa: E402
from vietlott_engine.crawler.prize_sources import records_from_jsonl, records_to_jsonl  # noqa: E402

REPOS = {
    "pqminh-4_vietlott-data": "https://github.com/pqminh-4/vietlott-data",
    "vietlott-data": "https://github.com/vietvudanh/vietlott-data",
    "NhanAZ_vietlott-research": "https://github.com/NhanAZ-Data/vietlott-research",
}
SEED = ROOT / "data" / "seed"
DRAW_FILES = {GameCode.MEGA_645: "power645.jsonl", GameCode.POWER_655: "power655.jsonl", GameCode.LOTTO_535: "power535.jsonl"}


def ensure_repo(vendor: Path, name: str, update: bool) -> Path:
    dest = vendor / name
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPOS[name], str(dest)], check=True)
    elif update:
        subprocess.run(["git", "-C", str(dest), "pull", "-q", "--ff-only"], check=False)
    return dest


def compare_prizes(old_path: Path, new_records) -> dict:  # type: ignore[no-untyped-def]
    if not old_path.exists():
        return {"previous_records": 0}
    old = {r.draw_id: r for r in records_from_jsonl(old_path.read_text(encoding="utf-8"))}
    new = {r.draw_id: r for r in new_records}
    common = sorted(set(old) & set(new))
    winners_equal = sum(old[i].winners == new[i].winners for i in common)
    pot_rows = [i for i in common if old[i].jackpot_pots and new[i].jackpot_pots]
    pot_equal = sum(old[i].jackpot_pots == new[i].jackpot_pots for i in pot_rows)
    pot_rel = [
        abs(old[i].jackpot_pots[k] - new[i].jackpot_pots[k]) / max(new[i].jackpot_pots[k], 1)
        for i in pot_rows
        for k in new[i].jackpot_pots
    ]
    return {
        "previous_records": len(old),
        "official_records": len(new),
        "common": len(common),
        "winner_counts_identical": winners_equal,
        "winner_mismatch_draw_ids": [i for i in common if old[i].winners != new[i].winners][:30],
        "pots_compared": len(pot_rows),
        "pots_identical": pot_equal,
        "pot_max_relative_difference": max(pot_rel) if pot_rel else None,
    }


def history_map(product: ProductCode, rows: list[dict]) -> dict[int, tuple[str, list[int], str]]:
    """id → (date, values, origin) for one layer (origin from the row's ``data_source``)."""
    h, _ = parse_product_rows(product, rows, "layer")
    origin = {}
    for r in rows:
        key = r.get("id", r.get("draw_id"))
        if key is not None:
            origin[int(str(key).lstrip("#"))] = r.get("data_source")
    return {int(i): (str(d), [int(x) for x in v], origin.get(int(i))) for i, d, v in zip(h.draw_ids, h.dates, h.values)}


def merge_layers(product: ProductCode, layers: list[tuple[str, list[dict]]]) -> tuple[list[dict], dict]:
    """Union of layers by draw id; earlier layers win; conflicts are counted and listed."""
    merged: dict[int, tuple[str, list[int], str]] = {}
    stats: dict = {"layers": []}
    for label, rows in layers:
        m = history_map(product, rows)
        common = set(m) & set(merged)
        same = sum(m[i][:2] == merged[i][:2] for i in common)
        conflicts = sorted(i for i in common if m[i][:2] != merged[i][:2])
        added = sorted(set(m) - set(merged))
        stats["layers"].append(
            {
                "layer": label,
                "rows": len(m),
                "overlap_with_higher_layers": len(common),
                "identical": same,
                "conflicts": len(conflicts),
                "conflict_examples": [{"draw_id": i, "kept": merged[i][:2], "other": m[i][:2]} for i in conflicts[:5]],
                "added": len(added),
                "added_by_origin": dict(Counter(m[i][2] or "unknown" for i in added).most_common()),
            }
        )
        for i in added:
            o = m[i][2]
            merged[i] = (m[i][0], m[i][1], label if o in (None, "official_vietlott") and label != "nhanaz" else f"{label}:{o}" if o else label)
    n = DIGIT_PRODUCTS.get(product)
    out = [{"id": i, "date": merged[i][0], "result": [f"{x:0{n}d}" for x in merged[i][1]] if n else merged[i][1], "src": merged[i][2]} for i in sorted(merged)]
    stats["final_rows"] = len(out)
    stats["origin_mix"] = dict(Counter(r["src"] for r in out).most_common())
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default="/tmp/vqe-sources")
    ap.add_argument("--canonical-dir", help="directory with <game>.jsonl canonical records (default: pqminh-4 clone)")
    ap.add_argument("--update", action="store_true", help="git pull the source repositories first")
    ap.add_argument("--compare-with", help="seed directory to compare against (default: the current data/seed)")
    ap.add_argument("--nhanaz", help="NhanAZ-Data/vietlott-research folder (default: clone into --vendor)")
    ap.add_argument("--no-nhanaz", action="store_true", help="skip the community archive")
    ap.add_argument("--v130-dir", help="Vietlott Quant Engine 1.3.0 folder: cross-check and last fill layer")
    a = ap.parse_args()
    vendor = Path(a.vendor)
    vendor.mkdir(parents=True, exist_ok=True)
    canon_dir = Path(a.canonical_dir) if a.canonical_dir else ensure_repo(vendor, "pqminh-4_vietlott-data", a.update) / "data" / "canonical"
    vv = ensure_repo(vendor, "vietlott-data", a.update) / "data"
    ref = Path(a.compare_with) if a.compare_with else SEED
    report: dict = {"sources": {"canonical": "pqminh-4/vietlott-data" if not a.canonical_dir else str(canon_dir), "keno_bingo18": "vietvudanh/vietlott-data"}, "compared_with": "previous seed (v3.1: third-party prize tables)" if a.compare_with else "current seed"}
    archive = None
    if not a.no_nhanaz:
        from vietlott_engine.crawler.sources.nhanaz import NhanAZArchive

        archive = NhanAZArchive(local_dir=a.nhanaz or ensure_repo(vendor, "NhanAZ_vietlott-research", a.update))
        report["sources"]["archive"] = "NhanAZ-Data/vietlott-research"
    if a.v130_dir:
        report["sources"]["v130"] = str(a.v130_dir)
    merge: dict = {"policy": "official records first; lower layers only add missing draw ids; every disagreement listed", "products": {}}

    def archive_draws(product: ProductCode):  # type: ignore[no-untyped-def]
        return asyncio.run(archive.draws(product)) if archive else None

    def v130_draws(product: ProductCode):  # type: ignore[no-untyped-def]
        if not a.v130_dir:
            return None
        from vietlott_engine.crawler.sources.vqe130 import import_v130

        return import_v130(a.v130_dir, product)

    for name, game in CANONICAL_GAMES.items():
        rows = read_jsonl(canon_dir / f"{name}.jsonl")
        canon_ids = {int(r["draw_id"]) for r in rows if r.get("draw_id")}
        mstat: dict = {"canonical_records": len(rows)}
        for label, got in (("nhanaz", archive_draws(ProductCode(name))), ("v130", v130_draws(ProductCode(name)))):
            if got is None:
                continue
            other = import_canonical(got.rows, game)
            base = import_canonical(rows, game)
            bd = {d.draw_id: (d.numbers, d.bonus, d.draw_date) for d in base.draws}
            bp = {p.draw_id: p for p in base.prizes}
            od = {d.draw_id: (d.numbers, d.bonus, d.draw_date) for d in other.draws}
            op = {p.draw_id: p for p in other.prizes}
            origin = {int(r["draw_id"]): r.get("data_source") for r in got.rows}
            cd, cp = sorted(set(bd) & set(od)), sorted(set(bp) & set(op))
            pot_conf = [i for i in cp if bp[i].jackpot_pots != op[i].jackpot_pots]
            add = [r for r in got.rows if int(r["draw_id"]) not in canon_ids and int(r["draw_id"]) not in {int(x["draw_id"]) for x in rows}]
            mstat[label] = {
                "rows": len(got.rows),
                "draws_common": len(cd),
                "draws_identical": sum(bd[i] == od[i] for i in cd),
                "prizes_common": len(cp),
                "winners_identical": sum(bp[i].winners == op[i].winners for i in cp),
                "pots_identical": len(cp) - len(pot_conf),
                "pot_conflicts": [{"draw_id": i, "origin": origin.get(i), "kept_official": bp[i].jackpot_pots, "other": op[i].jackpot_pots} for i in pot_conf[:10]],
                "added_draws": [int(r["draw_id"]) for r in add],
                "added_by_origin": dict(Counter(r.get("data_source") for r in add)),
            }
            rows = rows + add  # fill draws the official file lacks (with their prize tables)
        merge["products"][name] = mstat
        imp = import_canonical(rows, game)
        old_draws = {}
        old_file = SEED / DRAW_FILES[game]
        ref_file = ref / DRAW_FILES[game]
        if ref_file.exists():
            for line in ref_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    old_draws[int(r["id"])] = [int(x) for x in r["result"]]
        new_rows = draws_to_seed_rows(imp.draws)
        mism = [r["id"] for r in new_rows if int(r["id"]) in old_draws and sorted(old_draws[int(r["id"])][: len(r["result"]) - (1 if game != GameCode.MEGA_645 else 0)]) != sorted(r["result"][: len(r["result"]) - (1 if game != GameCode.MEGA_645 else 0)])]
        report[name] = {
            "draws": len(imp.draws),
            "first_id": min(d.draw_id for d in imp.draws),
            "last_id": max(d.draw_id for d in imp.draws),
            "last_date": max(d.draw_date for d in imp.draws).isoformat(),
            "added_vs_previous_seed": len(set(int(r["id"]) for r in new_rows) - set(old_draws)),
            "main_number_mismatches_vs_previous_seed": mism[:30],
            "prize_records": len(imp.prizes),
            "prize_records_with_jackpot_pots": sum(p.jackpot_pots is not None for p in imp.prizes),
            "records_with_official_pdf": imp.with_pdf,
            "rejected": imp.rejected[:20],
            "prize_comparison_with_previous_seed": compare_prizes(ref / f"prizes_{game.value}.jsonl", imp.prizes),
        }
        write_jsonl(old_file, new_rows)
        (SEED / f"prizes_{game.value}.jsonl").write_text(records_to_jsonl(imp.prizes), encoding="utf-8")

    def nz_rows(product: ProductCode) -> list[dict]:
        got = archive_draws(product)
        return got.rows if got else []

    def v130_rows(product: ProductCode) -> list[dict]:
        got = v130_draws(product)
        return got.rows if got else []

    plans = {
        ProductCode.MAX3D: [("vietlott.vn/detail", read_jsonl(canon_dir / "max3d.jsonl")), ("nhanaz", nz_rows(ProductCode.MAX3D)), ("vietlott.vn/list", read_jsonl(vv / "3d.jsonl"))],
        ProductCode.MAX3D_PRO: [("vietlott.vn/detail", read_jsonl(canon_dir / "max3d_pro.jsonl")), ("nhanaz", nz_rows(ProductCode.MAX3D_PRO)), ("vietlott.vn/list", read_jsonl(vv / "3d_pro.jsonl"))],
        ProductCode.KENO: [("vietlott.vn/list", read_jsonl(vv / "keno.jsonl")), ("nhanaz", nz_rows(ProductCode.KENO))],
        ProductCode.BINGO18: [("vietlott.vn/list", read_jsonl(vv / "bingo18.jsonl")), ("nhanaz", nz_rows(ProductCode.BINGO18))],
        ProductCode.MAX4D: [("nhanaz", nz_rows(ProductCode.MAX4D))],
    }
    exclusions: dict[tuple[str, int], dict] = {}
    if archive:
        for e in asyncio.run(archive.exclusions()):
            exclusions[(e["product"], e["draw_id"])] = e | {"source": "nhanaz"}
    if a.v130_dir:
        from vietlott_engine.crawler.sources.vqe130 import v130_exclusions

        for e in v130_exclusions(a.v130_dir):
            exclusions.setdefault((e["product"], e["draw_id"]), e | {"source": "v130"})
    (SEED / EXCLUSIONS_FILE).write_text(json.dumps(sorted(exclusions.values(), key=lambda e: (e["product"], e["draw_id"])), ensure_ascii=False, indent=1), encoding="utf-8")
    for product, layers in plans.items():
        if a.v130_dir:
            layers = layers + [("v130", v130_rows(product))]
        layers = [(lbl, r) for lbl, r in layers if r]
        if not layers:
            continue
        out, stats = merge_layers(product, layers)
        write_jsonl(SEED / SEED_FILES[product], out)
        h, rejected = parse_product_rows(product, out, source="+".join(lbl for lbl, _ in layers))
        excluded = {i for (p, i) in exclusions if p == product.value}
        merge["products"][product.value] = stats | {"not_confirmed_excluded": len(excluded & set(h.draw_ids.tolist()))}
        report[product.value] = drop_ids(h, excluded).coverage() | {"rejected": len(rejected), "rejected_examples": rejected[:5]}
    if archive:
        from vietlott_engine.crawler.sources.nhanaz import keno_rules

        rules = keno_rules(asyncio.run(archive.prize_rules()))
        (SEED / "keno_rules_official.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    merge["exclusions"] = {"count": len(exclusions), "by_product": dict(Counter(p for p, _ in exclusions))}
    (SEED / "DATA_MERGE_REPORT.json").write_text(json.dumps(merge, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    (SEED / "OFFICIAL_DATA_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("rejected", "rejected_examples", "main_number_mismatches_vs_previous_seed")} if isinstance(v, dict) else v for k, v in report.items()}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
