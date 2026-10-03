"""Draw histories of Vietlott's non-matrix products: Keno, Bingo18, Max 3D, Max 3D Pro.

The k-of-n games (Mega 6/45, Power 6/55, Lotto 5/35) live in ``core.games`` /
``core.history``. The other products draw differently and are stored here as compact
NumPy histories:

* Keno       — 20 different numbers from 1–80; current published cadence ~8 minutes.
* Bingo18    — three independent numbers 1–6 ("dice") every ~6 minutes.
* Max 3D     — 20 three-digit numbers 000–999 in groups 2 / 4 / 6 / 8 (Max 3D+ uses the
  same results); Max 3D Pro — the same layout, its own draw.
* Max 4D     — archive (19/11/2016 – 31/08/2021): 6 four-digit numbers 0000–9999 (1 first,
  2 second, 3 third); the two consolation prizes are the last 3 / 2 digits of the first.

Draws that Vietlott announced as *not confirmed* (e.g. Keno/Bingo18 on 02/04/2026) are kept
in ``exclusions.json`` and left out of every history unless explicitly requested.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np


class ProductCode(str, Enum):
    MEGA_645 = "mega645"
    POWER_655 = "power655"
    LOTTO_535 = "lotto535"
    KENO = "keno"
    BINGO18 = "bingo18"
    MAX3D = "max3d"
    MAX3D_PRO = "max3dpro"
    MAX4D = "max4d"


@dataclass(frozen=True)
class ProductInfo:
    code: ProductCode
    display_name: str
    draws: str
    schedule: str
    results_path: str  # page on vietlott.vn
    matrix_game: bool


PRODUCT_INFO: dict[ProductCode, ProductInfo] = {
    ProductCode.MEGA_645: ProductInfo(ProductCode.MEGA_645, "Mega 6/45", "6 số từ 1–45", "18:00 Thứ 4, 6, CN", "/vi/trung-thuong/ket-qua-trung-thuong/645", True),
    ProductCode.POWER_655: ProductInfo(ProductCode.POWER_655, "Power 6/55", "6 số + 1 số phụ từ 1–55", "18:00 Thứ 3, 5, 7", "/vi/trung-thuong/ket-qua-trung-thuong/655", True),
    ProductCode.LOTTO_535: ProductInfo(ProductCode.LOTTO_535, "Lotto 5/35", "5 số từ 1–35 + 1 số đặc biệt 1–12", "13:00 và 21:00 hằng ngày", "/vi/trung-thuong/ket-qua-trung-thuong/535", True),
    ProductCode.KENO: ProductInfo(ProductCode.KENO, "Keno", "20 số khác nhau từ 1–80", "~8 phút/kỳ, 06:00–21:5x hằng ngày", "/vi/trung-thuong/ket-qua-trung-thuong/winning-number-keno", False),
    ProductCode.BINGO18: ProductInfo(ProductCode.BINGO18, "Bingo18", "3 số độc lập từ 1–6", "~6 phút/kỳ, 06:00–21:5x hằng ngày", "/vi/trung-thuong/ket-qua-trung-thuong/winning-number-bingo18", False),
    ProductCode.MAX3D: ProductInfo(ProductCode.MAX3D, "Max 3D / Max 3D+", "20 số 000–999 (2 + 4 + 6 + 8)", "18:00 Thứ 2, 4, 6", "/vi/trung-thuong/ket-qua-trung-thuong/max-3D", False),
    ProductCode.MAX3D_PRO: ProductInfo(ProductCode.MAX3D_PRO, "Max 3D Pro", "20 số 000–999 (2 + 4 + 6 + 8)", "18:00 Thứ 3, 5, 7", "/vi/trung-thuong/ket-qua-trung-thuong/max-3DPro", False),
    ProductCode.MAX4D: ProductInfo(ProductCode.MAX4D, "Max 4D (đã ngừng)", "6 số 0000–9999 (1 + 2 + 3)", "18:00 Thứ 3, 5, 7 (11/2016 – 08/2021)", "/vi/trung-thuong/ket-qua-trung-thuong/max-4d", False),
}

THREE_DIGIT_LAYOUT = (("special", 2), ("first", 4), ("second", 6), ("third", 8))
FOUR_DIGIT_LAYOUT = (("first", 1), ("second", 2), ("third", 3))
DIGIT_PRODUCTS = {ProductCode.MAX3D: 3, ProductCode.MAX3D_PRO: 3, ProductCode.MAX4D: 4}
WIDTH = {ProductCode.KENO: 20, ProductCode.BINGO18: 3, ProductCode.MAX3D: 20, ProductCode.MAX3D_PRO: 20, ProductCode.MAX4D: 6}


def get_product(code: str | ProductCode) -> ProductCode:
    key = code.value if isinstance(code, ProductCode) else str(code).lower().replace("_", "").replace(" ", "").replace("+", "")
    aliases = {"max3dpro": ProductCode.MAX3D_PRO, "3dpro": ProductCode.MAX3D_PRO, "3d": ProductCode.MAX3D, "max3d": ProductCode.MAX3D, "4d": ProductCode.MAX4D, "bingo": ProductCode.BINGO18, "mega": ProductCode.MEGA_645, "power": ProductCode.POWER_655, "lotto": ProductCode.LOTTO_535}
    for c in ProductCode:
        if c.value == key:
            return c
    if key in aliases:
        return aliases[key]
    raise KeyError(f"unknown product {code!r}; expected one of {[c.value for c in ProductCode]}")


@dataclass(frozen=True)
class ProductHistory:
    """Draws of one non-matrix product, oldest first.

    ``values``: Keno (D, 20) sorted ints 1–80 · Bingo18 (D, 3) ints 1–6 in drawn order ·
    Max 3D (D, 20) ints 0–999 in the official order (2 special, 4 first, 6 second, 8 third) ·
    Max 4D (D, 6) ints 0–9999 (1 first, 2 second, 3 third).
    """

    product: ProductCode
    draw_ids: np.ndarray
    dates: np.ndarray
    values: np.ndarray
    source: str = ""

    def __len__(self) -> int:
        return len(self.draw_ids)

    def id_gaps(self) -> int:
        ids = np.sort(self.draw_ids)
        return int(np.sum(np.diff(ids) - 1)) if len(ids) > 1 else 0

    def coverage(self) -> dict:
        return {
            "product": self.product.value,
            "draws": len(self),
            "first_id": int(self.draw_ids.min()) if len(self) else None,
            "last_id": int(self.draw_ids.max()) if len(self) else None,
            "first_date": str(self.dates.min()) if len(self) else None,
            "last_date": str(self.dates.max()) if len(self) else None,
            "missing_ids_inside_range": self.id_gaps(),
            "source": self.source,
        }


# ------------------------------------------------------------------ parsing
def _did(x: str | int) -> int:
    return int(str(x).strip().lstrip("#"))


def _validate(product: ProductCode, vals: list[int]) -> bool:
    if product == ProductCode.KENO:
        return len(vals) == 20 and len(set(vals)) == 20 and all(1 <= v <= 80 for v in vals)
    if product == ProductCode.BINGO18:
        return len(vals) == 3 and all(1 <= v <= 6 for v in vals)
    if product == ProductCode.MAX4D:
        return len(vals) == 6 and all(0 <= v <= 9999 for v in vals)
    return len(vals) == 20 and all(0 <= v <= 999 for v in vals)


def parse_product_rows(product: ProductCode, rows: list[dict], source: str) -> tuple[ProductHistory, list[dict]]:
    """Rows from the mirror / canonical JSONL formats → history (+ rejected rows)."""
    seen: dict[int, tuple[date, list[int]]] = {}
    rejected = []
    for r in rows:
        try:
            if product in DIGIT_PRODUCTS:
                res = r.get("result")
                if isinstance(res, dict) and isinstance(res.get("tiers"), dict):  # {"first": [...], ...}
                    layout = FOUR_DIGIT_LAYOUT if product == ProductCode.MAX4D else THREE_DIGIT_LAYOUT
                    vals = [int(x) for code, _ in layout for x in res["tiers"][code]]
                elif isinstance(res, dict) and "tiers" in res:  # canonical (official detail pages)
                    vals = [int(x) for t in res["tiers"] for x in t["numbers"]]
                elif isinstance(res, dict):  # vietvudanh: {"Giải Đặc biệt": [...], ...}
                    vals = [int(x) for k in ("Giải Đặc biệt", "Giải Nhất", "Giải Nhì", "Giải ba") for x in res[k]]
                else:  # flattened list
                    vals = [int(x) for x in res]
                did = _did(r.get("draw_id", r.get("id")))
                d = date.fromisoformat(str(r.get("draw_date", r.get("date")))[:10])
            else:
                vals = [int(x) for x in r["result"]]
                did = _did(r["id"])
                d = date.fromisoformat(str(r["date"])[:10])
            if product == ProductCode.KENO:
                vals = sorted(vals)
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({"row": str(r)[:200], "error": str(exc)})
            continue
        if not _validate(product, vals):
            rejected.append({"row": str(r)[:200], "error": "values outside the product's rules"})
            continue
        seen[did] = (d, vals)
    ids = np.array(sorted(seen), dtype=np.int64)
    dates = np.array([seen[i][0] for i in ids], dtype="datetime64[D]")
    values = np.array([seen[i][1] for i in ids], dtype=np.int16).reshape(-1, WIDTH[product])
    return ProductHistory(product, ids, dates, values, source), rejected


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Replace a complete history atomically; a failed write preserves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n"
    name = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, suffix='.tmp', delete=False) as fh:
            name = fh.name
            if str(path).endswith('.gz'):
                with gzip.GzipFile(fileobj=fh, mode='wb', compresslevel=9, mtime=0) as compressed:
                    compressed.write(text.encode('utf-8'))
            else:
                fh.write(text.encode('utf-8'))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def history_to_rows(h: ProductHistory) -> list[dict]:
    rows = []
    for i, d, v in zip(h.draw_ids, h.dates, h.values):
        vals = [int(x) for x in v]
        n = DIGIT_PRODUCTS.get(h.product)
        rows.append({"id": int(i), "date": str(d), "result": [f"{x:0{n}d}" for x in vals] if n else vals})
    return rows


SEED_FILES = {
    ProductCode.KENO: "keno.jsonl.gz",
    ProductCode.BINGO18: "bingo18.jsonl.gz",
    ProductCode.MAX3D: "max3d.jsonl",
    ProductCode.MAX3D_PRO: "max3d_pro.jsonl",
    ProductCode.MAX4D: "max4d.jsonl",
}
EXCLUSIONS_FILE = "exclusions.json"


def load_exclusions(*directories: Path | None) -> dict[str, set[int]]:
    """product → draw ids announced as not confirmed (merged over the given folders)."""
    out: dict[str, set[int]] = {}
    for d in directories:
        if d is None or not (Path(d) / EXCLUSIONS_FILE).exists():
            continue
        for e in json.loads((Path(d) / EXCLUSIONS_FILE).read_text(encoding="utf-8")):
            out.setdefault(e["product"], set()).add(int(e["draw_id"]))
    return out


def drop_ids(h: ProductHistory, ids: set[int]) -> ProductHistory:
    if not ids:
        return h
    keep = ~np.isin(h.draw_ids, list(ids))
    return ProductHistory(h.product, h.draw_ids[keep], h.dates[keep], h.values[keep], h.source)


def load_product_history(product: str | ProductCode, directory: Path, include_unconfirmed: bool = False) -> ProductHistory:
    code = get_product(product)
    if code not in SEED_FILES:
        raise KeyError(f"{code.value} is a matrix game; use the draw repository")
    path = Path(directory) / SEED_FILES[code]
    if not path.exists():
        raise FileNotFoundError(path)
    h, _ = parse_product_rows(code, read_jsonl(path), source=path.name)
    return h if include_unconfirmed else drop_ids(h, load_exclusions(directory).get(code.value, set()))
