#!/usr/bin/env python3
"""Write ``reports/verification_checklist.{md,csv}``: the vietlott.vn pages worth opening by hand.

The engine does not get past vietlott.vn's Cloudflare "verify you are human" check — a person
does, in a normal browser. This list keeps that manual work small and useful: pages where the
bundled data rests on a secondary source or on a single source, plus a reproducible random
sample of the Keno draws that only the xoso.com.vn archive has. Save the pages (Ctrl+S →
"Webpage, Single File" / .mhtml) or a DevTools HAR, then:

    vietlott products import-pages --path <folder> --dry-run   # compare only
    vietlott products import-pages --path <folder>             # compare and import

    python scripts/verification_checklist.py [--keno-sample 30] [--seed 20261002]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.core.products import read_jsonl  # noqa: E402

SEED = Path(os.environ.get("VQE_PRODUCT_SEED_DIR") or ROOT / "data" / "seed")
OUT = Path(os.environ.get("VQE_REPORTS_DIR") or ROOT / "reports")
BASE = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong"
DETAIL = {
    "mega645": BASE + "/645?id={id:05d}&nocatche=1",
    "power655": BASE + "/655?id={id:05d}&nocatche=1",
    "lotto535": BASE + "/535?id={id:05d}&nocatche=1",
    "max3d": BASE + "/max-3D?id={id:05d}&nocatche=1",
    "max3dpro": BASE + "/max-3DPro?id={id:05d}&nocatche=1",
    "max4d": BASE + "/max-4d?id={id:05d}&nocatche=1",
    "keno": BASE + "/view-detail-keno-result?id={id:07d}",
    "bingo18": BASE + "/view-detail-bingo18-result?nocatche=1&id={id:07d}",
}


def url(product: str, draw_id: int) -> str:
    return DETAIL[product].format(id=draw_id)


def build(keno_sample: int = 30, bingo_sample: int = 10, gap_sample: int = 10, seed: int = 20261002) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(group: str, product: str, draw_id: int, why: str) -> None:
        rows.append({"group": group, "product": product, "draw_id": draw_id, "url": url(product, draw_id), "why": why})

    merge = json.loads((SEED / "DATA_MERGE_REPORT.json").read_text(encoding="utf-8"))
    lotto = merge["products"].get("lotto535", {})
    for i in lotto.get("nhanaz", {}).get("added_draws", []):
        add("1 · ưu tiên", "lotto535", int(i), "kết quả và bảng giải chỉ có từ trang phụ (xosominhngoc)")
    for c in lotto.get("nhanaz", {}).get("pot_conflicts", []):
        kept, other = c["kept_official"].get("jackpot1", 0) / 1e9, c["other"].get("jackpot1", 0) / 1e9
        add("1 · ưu tiên", "lotto535", int(c["draw_id"]), f"Jackpot lệch giữa nguồn: {kept:.2f} tỷ (bản ghi chính thức) vs {other:.2f} tỷ (trang phụ)")

    keno = read_jsonl(SEED / "keno.jsonl.gz")
    src = {int(r["id"]): r.get("src", "") for r in keno}
    recent_secondary = sorted(i for i, s in src.items() if "xosominhngoc" in s)[-5:]
    if recent_secondary:
        add("1 · ưu tiên", "keno", recent_secondary[-1], "trang chi tiết Keno: đọc ô bậc 5 trùng 4 số (150.000 đ hay 0 đ?) và các cửa phụ")
        for i in recent_secondary[:-1]:
            add("1 · ưu tiên", "keno", i, "kỳ gần đây chỉ có từ trang phụ (xosominhngoc)")

    bingo = read_jsonl(SEED / "bingo18.jsonl.gz")
    bingo_secondary = sorted(int(r["id"]) for r in bingo if "xosominhngoc" in (r.get("src") or ""))
    if bingo_secondary:
        add("1 · ưu tiên", "bingo18", bingo_secondary[-1], "kỳ gần đây chỉ có từ trang phụ (xosominhngoc); trang danh sách Bingo18 lưu bằng HAR phủ được nhiều kỳ")

    xoso = np.array(sorted(i for i, s in src.items() if "xoso_com_vn_archive" in s))
    for i in sorted(rng.choice(xoso, size=min(keno_sample, len(xoso)), replace=False).tolist()) if len(xoso) else []:
        add("2 · mẫu Keno 2019–2022", "keno", int(i), "mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn")

    unknown = np.array(sorted(int(r["id"]) for r in bingo if r.get("src") == "nhanaz:unknown"))
    for i in sorted(rng.choice(unknown, size=min(bingo_sample, len(unknown)), replace=False).tolist()) if len(unknown) else []:
        add("3 · mẫu Bingo18", "bingo18", int(i), "mẫu ngẫu nhiên phần kho ghi nguồn 'unknown'")

    ids = np.array(sorted(src))
    excluded = {e["draw_id"] for e in json.loads((SEED / "exclusions.json").read_text(encoding="utf-8")) if e["product"] == "keno"}
    gaps = [int(x) for a, b in zip(ids[:-1], ids[1:]) if b - a > 1 for x in range(a + 1, b) if x not in excluded]
    for i in sorted(rng.choice(np.array(gaps), size=min(gap_sample, len(gaps)), replace=False).tolist()) if gaps else []:
        add("4 · khoảng trống Keno", "keno", int(i), "mã kỳ không có trong dữ liệu: có kỳ quay này không?")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keno-sample", type=int, default=30)
    ap.add_argument("--bingo-sample", type=int, default=10)
    ap.add_argument("--gap-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    rows = build(a.keno_sample, a.bingo_sample, a.gap_sample, a.seed)
    OUT.mkdir(exist_ok=True)
    with (OUT / "verification_checklist.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "product", "draw_id", "url", "why"])
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Danh sách trang vietlott.vn nên kiểm tra",
        "",
        "Engine không vượt bước *xác minh bạn là người* (Cloudflare) của vietlott.vn. Bạn mở các trang dưới đây bằng trình duyệt bình thường, "
        "tự qua bước xác minh, rồi lưu trang: **Ctrl+S → \"Trang web, một tệp\" (.mhtml)**. Hoặc mở DevTools (F12) ▸ Network *trước* khi tải trang, chọn bộ lọc **All**, "
        "duyệt các trang, rồi bấm nút mũi tên tải xuống **\"Export HAR (sanitized)…\"** trên thanh công cụ của tab Network (.har; Chrome/Edge bản mới — "
        "bản cũ: chuột phải ▸ \"Save all as HAR with content\"). Bỏ tất cả vào một thư mục, rồi chạy:",
        "",
        "```bash",
        "vietlott products import-pages --path <thư-mục> --dry-run   # chỉ so sánh",
        "vietlott products import-pages --path <thư-mục>             # so sánh và nhập",
        "```",
        "",
        "Nên mở chậm, như người đọc bình thường. Nhóm 1 là quan trọng nhất (vài trang); nhóm 2–4 là mẫu ngẫu nhiên (hạt giống "
        f"{a.seed}) để ước lượng tỷ lệ sai của nguồn phụ: {a.keno_sample} kỳ khớp cả ⇒ tỷ lệ sai ≤ {1 - 0.05 ** (1 / max(a.keno_sample, 1)):.1%} (độ tin cậy 95%).",
        "",
        "| Nhóm | Sản phẩm | Kỳ | Lý do | Trang |",
        "|---|---|---:|---|---|",
    ]
    md += [f"| {r['group']} | {r['product']} | {r['draw_id']} | {r['why']} | {r['url']} |" for r in rows]
    (OUT / "verification_checklist.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
