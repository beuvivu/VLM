#!/usr/bin/env python3
"""Regenerate ``reports/products.{md,json}``: coverage of all seven Vietlott products, exact
odds and return to player of Keno / Bingo18 / Max 3D, randomness tests of the Keno, Bingo18
and Max 3D histories, and the Max 3D ↔ Max 3D Pro digit cross-check.

    python scripts/products_report.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.analytics.products import digit_cell_test, digit_replication, product_randomness, split_history  # noqa: E402
from vietlott_engine.core.products import ProductHistory, read_jsonl  # noqa: E402
from vietlott_engine.core.products import PRODUCT_INFO, SEED_FILES, ProductCode, load_product_history  # noqa: E402
from vietlott_engine.game_theory import max3d  # noqa: E402
from vietlott_engine.game_theory.fastgames import bingo18_odds, keno_odds  # noqa: E402

SEED = Path(os.environ.get("VQE_PRODUCT_SEED_DIR") or ROOT / "data" / "seed")
OUT = Path(os.environ.get("VQE_REPORTS_DIR") or ROOT / "reports")


def main() -> int:
    out: dict = {}
    odr = json.loads((SEED / "OFFICIAL_DATA_REPORT.json").read_text(encoding="utf-8"))
    md = ["# Vietlott — toàn bộ sản phẩm: dữ liệu, xác suất, kiểm định ngẫu nhiên", ""]

    md += ["## 1. Phạm vi dữ liệu", "", "| Sản phẩm | Cách quay | Lịch | Kỳ trong dữ liệu | Từ – đến | Nguồn |", "|---|---|---|---:|---|---|"]
    for code, info in PRODUCT_INFO.items():
        r = odr.get(code.value, {})
        if info.matrix_game:
            span, src = f"#{r.get('first_id')} → #{r.get('last_id')} ({r.get('last_date')})", "trang chi tiết vietlott.vn (kết quả + bảng giải + Jackpot)"
        else:
            span, src = f"{r.get('first_date')} → {r.get('last_date')}", r.get("source", "")
        md.append(f"| {info.display_name} | {info.draws} | {info.schedule} | {r.get('draws', 0):,} | {span} | {src} |")
    merge = json.loads((SEED / "DATA_MERGE_REPORT.json").read_text(encoding="utf-8")) if (SEED / "DATA_MERGE_REPORT.json").exists() else {}
    excl = merge.get("exclusions", {})
    md += ["", "Keno và Bingo18: sau khi gộp các nguồn vẫn còn khoảng trống mã kỳ (Keno "
           f"{odr['keno']['missing_ids_inside_range']:,}, Bingo18 {odr['bingo18']['missing_ids_inside_range']:,} mã trong khoảng); "
           "các kiểm định giữa hai kỳ liền nhau chỉ dùng cặp có mã kỳ liên tiếp. "
           f"Vietlott thông báo {excl.get('count', 0)} kỳ *không được xác nhận* (Keno/Bingo18 ngày 02/04/2026); "
           f"{sum(v.get('not_confirmed_excluded', 0) for v in merge.get('products', {}).values())} kỳ trong số đó có trong dữ liệu và bị loại khỏi mọi phân tích.", ""]
    if merge:
        md += ["**Gộp nguồn** (bản ghi chính thức trước; lớp sau chỉ thêm mã kỳ còn thiếu):", "", "| Sản phẩm | Lớp | Số kỳ | Trùng với lớp trên | Giống hệt | Mâu thuẫn | Thêm mới |", "|---|---|---:|---:|---:|---:|---:|"]
        for prod, v in merge.get("products", {}).items():
            for L in v.get("layers", []):
                md.append(f"| {prod} | {L['layer']} | {L['rows']:,} | {L['overlap_with_higher_layers']:,} | {L['identical']:,} | {L['conflicts']} | {L['added']:,} |")
            for lab in ("nhanaz", "v130"):
                if lab in v:
                    x = v[lab]
                    md.append(f"| {prod} | {lab} (so với bản ghi chính thức) | {x['rows']:,} | {x['draws_common']:,} | {x['draws_identical']:,} kết quả, {x['winners_identical']:,} bảng giải, {x['pots_identical']:,} Jackpot | {len(x['pot_conflicts'])} | {len(x['added_draws'])} |")
        md.append("")
    out["coverage"] = {k: v for k, v in odr.items() if isinstance(v, dict)}
    out["merge"] = merge

    bacs, sides = keno_odds()
    md += ["## 2. Keno (10.000 đ/vé)", "", "| Bậc | Bảng giải (số trùng → tiền) | P(trúng) | Giải cao nhất | RTP | Bảng khác đang lưu hành |", "|---:|---|---:|---:|---:|---|"]
    for b in bacs:
        prizes = ", ".join(f"{m}→{v:,}".replace(",", ".") for m, v in sorted(b.prizes.items()))
        var = "; ".join(f"{k}: RTP {v:.3f}" for k, v in b.variants.items()) or "—"
        md.append(f"| {b.bac} | {prizes} | {b.p_any_prize:.4f} | 1/{b.one_in_top_prize:,.0f} | {b.return_to_player:.3f} | {var} |")
    md += ["", "| Cửa phụ | Điều kiện | P | Trả (đ) | RTP | Phiên bản khác | Đã xác minh |", "|---|---|---:|---|---:|---|---|"]
    for x in sides:
        pays = " / ".join(f"{v:,}".replace(",", ".") if v is not None else "?" for v in x.payouts)
        alts = "; ".join(f"{k} → {v:.3f}" for k, v in x.alternatives_rtp.items()) or "—"
        rtp = f"{x.return_to_player:.3f}" if x.return_to_player is not None else "—"
        md.append(f"| {x.name} | {x.condition} | {x.probability:.4f} | {pays} | {rtp} | {alts} | {'có' if x.verified else 'chưa'} |")
    md += ["", "Bảng giải lấy từ trang chi tiết Keno của vietlott.vn (kỳ #0284640, 13/06/2026, lưu trong kho cộng đồng NhanAZ-Data/vietlott-research; "
           "`data/seed/keno_rules_official.json`), khớp trang sản phẩm mà Vietlott Quant Engine 1.3.0 dùng. Bảng in trên trang đại lý/báo chí (VTC Pay 2020, xosovip 2025, xoso.mobi) "
           "được giữ làm phiên bản để so: VTC Pay 2020 là bảng cũ của bậc 10, xosovip chép sai bậc 5 và bậc 8. Ô duy nhất chưa thống nhất: bậc 5 trùng 4 số — trang sản phẩm ghi 150.000 đ, "
           "trang chi tiết 2026 ghi \"0 đ\" ở mọi kỳ; dùng 150.000 đ.", ""]
    out["keno"] = {"bac": [b.model_dump() for b in bacs], "side_bets": [x.model_dump() for x in sides]}

    rows = bingo18_odds()
    md += ["## 3. Bingo18 (10.000 đ/vé, 3 số 1–6 độc lập)", "", "| Cửa | Điều kiện | P(trúng) | Trả (đ) | RTP |", "|---|---|---:|---|---:|"]
    md += [f"| {r.bet} | {r.condition} | {r.probability:.4f} | {r.payout} | {r.return_to_player:.3f} |" for r in rows]
    md.append("")
    out["bingo18"] = [r.model_dump() for r in rows]

    md += ["## 4. Max 3D / 3D+ / 3D Pro", "", "| Sản phẩm | RTP (vé 2 số khác nhau) |", "|---|---:|"]
    out["max3d"] = {}
    for name, prod in max3d.PRODUCTS.items():
        s = max3d.product_summary(prod)
        md.append(f"| {prod.display_name if hasattr(prod, 'display_name') else name} | {s.rtp_distinct_numbers:.3f} |")
        out["max3d"][name] = s.model_dump()
    md.append("")

    md += ["## 5. Kiểm định ngẫu nhiên (so với luật quay công bằng, hiệu chỉnh Benjamini–Hochberg)", ""]
    out["randomness"] = {}
    hist = {code: load_product_history(code, SEED) for code in SEED_FILES}
    for code, h in hist.items():
        if not len(h):
            continue
        rep = product_randomness(h)
        out["randomness"][code.value] = rep.model_dump()
        md += [f"### {PRODUCT_INFO[code].display_name} — {rep.draws:,} kỳ ({rep.first_date} → {rep.last_date})", "", "| Kiểm định | Thống kê | p | q (BH) | Chi tiết |", "|---|---:|---:|---:|---|"]
        for t in rep.tests:
            md.append(f"| {t.name} | {t.statistic:,.2f} | {t.p_value:.4f} | {t.q_value:.3f}{' **←**' if t.q_value < 0.05 else ''} | {t.detail} |")
        md += ["", f"→ {rep.verdict}", ""]

    rep = digit_replication(hist[ProductCode.MAX3D], hist[ProductCode.MAX3D_PRO])
    rev = digit_replication(hist[ProductCode.MAX3D_PRO], hist[ProductCode.MAX3D])
    c = rep.cell
    md += ["### Max 3D: chữ số hàng đơn vị", "",
           f"Ô lệch mạnh nhất trong Max 3D là chữ số **{c.digit}** ở hàng **{c.position}** ({c.discovery_share:.4f} thay vì 0,1000). Kiểm định *chỉ ô này* trên Max 3D Pro — các kỳ quay độc lập, chưa dùng để chọn ô — "
           f"cho {c.confirm_share:.4f} ({c.confirm_hits:,}/{c.confirm_trials:,}), p một phía = {c.confirm_p_value:.1e}. Chọn ô trên Max 3D Pro rồi kiểm tra trên Max 3D cũng ra đúng ô đó "
           f"(chữ số {rev.cell.digit} hàng {rev.cell.position}, p = {rev.cell.confirm_p_value:.1e}). Ba bản sao độc lập (pqminh-4, vietvudanh, googlesky) khớp nhau ở mọi kỳ chung, nên đây không phải lỗi thu thập.", "",
           f"Tỷ lệ gộp {rep.pooled_share:.4f} (95 %: {rep.pooled_ci95[0]:.4f}–{rep.pooled_ci95[1]:.4f}). {rep.note}", ""]
    splits = []
    for code in (ProductCode.MAX3D, ProductCode.MAX3D_PRO):
        a, b = split_history(hist[code])
        r = digit_replication(a, b, f"{code.value} nửa đầu", f"{code.value} nửa sau")
        splits.append(r.model_dump())
        md.append(f"- {PRODUCT_INFO[code].display_name}, nửa đầu → nửa sau: ô chọn được là chữ số {r.cell.digit} hàng {r.cell.position} ({r.cell.discovery_share:.4f} → {r.cell.confirm_share:.4f}, p = {r.cell.confirm_p_value:.3f}).")
    md += ["", "Ý nghĩa thực tế: lệch có thật về thống kê nhưng nhỏ. Số có lợi nhất chỉ tăng xác suất ≈ "
           f"{(rep.per_number_multiplier_max - 1):.0%} (ước lượng trong mẫu, còn thiên lên); vé Max 3D cần tăng ≈ 83 % mới hòa vốn. "
           "Kết luận về cách chơi không đổi: kỳ vọng vẫn âm.", ""]
    cell4 = digit_cell_test(hist[ProductCode.MAX4D], 4, 0, c.digit, "greater") if ProductCode.MAX4D in hist else None
    if cell4 is not None:
        md += [f"**Kiểm tra lần thứ ba, đặt trước giả thuyết: Max 4D** (2016–2021, trò chơi khác, {len(hist[ProductCode.MAX4D]):,} kỳ × 6 số). "
               f"Chữ số {c.digit} ở hàng đơn vị: {cell4.share:.4f} ({cell4.hits:,}/{cell4.trials:,}; 95 %: {cell4.ci95[0]:.4f}–{cell4.ci95[1]:.4f}), p một phía = {cell4.p_value:.3f}. "
               "Cùng cỡ lệch như Max 3D và Max 3D Pro, trên một sản phẩm độc lập — bằng chứng lệch hàng đơn vị của các trò chơi chữ số mạnh thêm. "
               "Nguyên nhân (thiết bị quay, bộ cầu số) không xác định được từ dữ liệu kết quả.", ""]
    out["max3d_digit_check"] = {"max3d_to_pro": rep.model_dump(), "pro_to_max3d": rev.model_dump(), "time_splits": splits, "max4d_preregistered": cell4.model_dump() if cell4 else None}

    rows_k = read_jsonl(SEED / "keno.jsonl.gz")
    src = {r["id"]: r.get("src") for r in rows_k}
    hk = hist[ProductCode.KENO]
    md += ["### Keno theo nguồn", "", "Phần lớn các kỳ 08/2019–11/2022, cùng một phần đến 03/2025, chỉ có trong kho lưu trữ xoso.com.vn (không có bản chính thức để đối chiếu); kiểm định riêng từng phần để phát hiện lỗi chép số:", "",
           "| Phần | Kỳ | Từ – đến | q nhỏ nhất | Kết luận |", "|---|---:|---|---:|---|"]
    segs = {}
    for lab, key in (("kho xoso.com.vn", "nhanaz:xoso_com_vn_archive"), ("danh sách vietlott.vn", "vietlott.vn/list")):
        m = np.array([src.get(int(i)) == key for i in hk.draw_ids])
        if m.sum() < 1000:
            continue
        r = product_randomness(ProductHistory(hk.product, hk.draw_ids[m], hk.dates[m], hk.values[m], lab))
        segs[lab] = r.model_dump()
        md.append(f"| {lab} | {r.draws:,} | {r.first_date} → {r.last_date} | {r.min_q_value:.3f} | {r.verdict} |")
    md.append("")
    out["keno_segments"] = segs

    OUT.mkdir(exist_ok=True)
    (OUT / "products.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "products.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
