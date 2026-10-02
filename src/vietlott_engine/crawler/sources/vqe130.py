"""Import the product snapshot of *Vietlott Quant Engine 1.3.0* (``data/products``).

1.3.0 stores every product as Parquet with one row per draw — ``draw_number, draw_date,
status, payload`` (JSON with ``result`` and provenance) — and prize cells as
``draw_number, variant, tier, payload`` (``winner_count``, ``displayed_value_vnd``). Its rows
come from a pinned revision of ``NhanAZ-Data/vietlott-research`` plus ``vietvudanh`` reference
rows, so they are converted to the archive's row format and go through the same converter
(:mod:`vietlott_engine.crawler.sources.nhanaz`). Draws with ``status = not_confirmed`` become exclusions.

Parquet is read with whichever is available: the ``duckdb`` Python package (an engine
dependency), ``pyarrow``, or the DuckDB command-line tool (``VQE_DUCKDB_CLI`` or ``duckdb`` on
``PATH``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.products import ProductCode
from vietlott_engine.crawler.sources.nhanaz import MATRIX, ArchiveDraws, attach_prizes, convert_draws

V130_PRODUCTS = {
    ProductCode.MEGA_645: "mega645",
    ProductCode.POWER_655: "power655",
    ProductCode.LOTTO_535: "lotto535",
    ProductCode.KENO: "keno",
    ProductCode.BINGO18: "bingo18",
    ProductCode.MAX3D: "max3d",
    ProductCode.MAX3D_PRO: "max3dpro",
    ProductCode.MAX4D: "max4d",
}


def read_parquet(path: Path) -> list[dict]:
    """All rows of a Parquet file as dicts (duckdb → pyarrow → DuckDB CLI)."""
    path = Path(path)
    if not path.exists():
        raise SourceError(f"missing file {path}")
    errors = []
    try:
        import duckdb  # type: ignore[import-not-found]

        con = duckdb.connect()
        cur = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — try the next reader
        errors.append(f"duckdb: {type(exc).__name__}")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        return pq.read_table(str(path)).to_pylist()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pyarrow: {type(exc).__name__}")
    cli = os.environ.get("VQE_DUCKDB_CLI") or shutil.which("duckdb")
    if cli:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.json"
            sql = f"COPY (SELECT * FROM read_parquet('{path.as_posix()}')) TO '{out.as_posix()}' (FORMAT json)"
            proc = subprocess.run([cli, "-c", sql], capture_output=True, text=True)
            if proc.returncode == 0 and out.exists():
                with out.open(encoding="utf-8") as fh:
                    return [json.loads(line) for line in fh if line.strip()]
            errors.append(f"duckdb CLI: {proc.stderr.strip()[:120]}")
    raise SourceError(f"cannot read {path.name}: install duckdb or pyarrow, or set VQE_DUCKDB_CLI ({'; '.join(errors)})")


def _draw_row(r: dict) -> dict:
    p = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
    result = {k: v for k, v in (p.get("result") or {}).items() if k != "kind"}
    status = r.get("status") or p.get("status") or "reported"
    return {
        "draw_id": str(p.get("draw_id") or r["draw_number"]),
        "draw_date": str(r.get("draw_date") or p.get("draw_date"))[:10],
        "draw_status": "confirmed" if status == "reported" else status,
        "result_json": json.dumps(result),
        "attributes_json": json.dumps(p.get("attributes") or {}),
        "source_url": p.get("source_url"),
    }


def _prize_row(r: dict) -> dict:
    p = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
    return {
        "draw_id": str(p.get("draw_id") or r["draw_number"]),
        "prize_tier": p.get("tier") or r.get("tier"),
        "winner_count": "" if p.get("winner_count") is None else str(p["winner_count"]),
        "prize_value_vnd": "" if p.get("displayed_value_vnd") is None else str(p["displayed_value_vnd"]),
    }


def products_dir(root: Path | str) -> Path:
    """Accept the package root or its ``data/products`` folder."""
    p = Path(root)
    for cand in (p, p / "data" / "products"):
        if (cand / "coverage.json").exists() or any(cand.glob("*.parquet")):
            return cand
    raise SourceError(f"{root}: no Vietlott Quant Engine 1.3.0 product snapshot (data/products/*.parquet)")


def import_v130(root: Path | str, product: ProductCode) -> ArchiveDraws:
    d = products_dir(root)
    name = V130_PRODUCTS[product]
    out = convert_draws(product, [_draw_row(r) for r in read_parquet(d / f"{name}.parquet")])
    if product in MATRIX and (d / f"{name}_prizes.parquet").exists():
        attach_prizes(out.rows, [_prize_row(r) for r in read_parquet(d / f"{name}_prizes.parquet")])
    return out


def v130_exclusions(root: Path | str) -> list[dict]:
    """``exclusions.json`` of the snapshot (issuer notices), normalised like the archive's."""
    path = products_dir(root) / "exclusions.json"
    if not path.exists():
        return []
    out = []
    for product, items in json.loads(path.read_text(encoding="utf-8")).items():
        for e in items:
            out.append(
                {
                    "product": product,
                    "draw_id": int(str(e["draw_id"]).lstrip("#")),
                    "draw_status": e.get("draw_status", "not_confirmed"),
                    "effective_date": e.get("effective_date", ""),
                    "reason": e.get("reason", ""),
                    "source_url": e.get("source_url", ""),
                }
            )
    return out
