"""Regenerate ``reports/bao_vietlott.{md,json}``: every bao option Vietlott sells, exact
prize tables, bao vs bao rút gọn vs single tickets, bao EV with co-winners and taxes, and
the Max 3D family.

    python scripts/bao_report.py [--sims 100000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.core.games import LOTTO_535, MEGA_645, POWER_655, GameSpec  # noqa: E402
from vietlott_engine.game_theory import max3d  # noqa: E402
from vietlott_engine.game_theory.bao import analyse_bao, bao_catalog, bao_ev, bao_prize_table, compare_bao_strategies, fmt_vnd  # noqa: E402

OUT = Path(os.environ.get("VQE_REPORTS_DIR") or ROOT / "reports")


def numbers_for(spec: GameSpec, v: int) -> list[int]:
    step = spec.pool_size // max(v, 1)
    return sorted({1 + (i * step) % spec.pool_size for i in range(v)})[:v] if v else []


def catalog_section(spec: GameSpec) -> tuple[list[str], list[dict]]:
    md = [
        f"### {spec.display_name}",
        "",
        "| Kiểu | Số chọn | Lượt | Giá (đ) | P(trúng ≥ 1 giải) | Cùng tiền, vé lẻ | P(thưởng ≥ tiền vé) | P(trúng Jackpot/Độc đắc) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for o in bao_catalog(spec):
        nums = numbers_for(spec, o.main_numbers)
        sp = list(range(1, o.special_numbers + 1)) if spec.separate_special else None
        r = analyse_bao(spec, nums, sp)
        p_jp = sum(oc.probability for oc in r.outcomes if oc.jackpot_plays.get("jackpot1"))
        sel = f"{o.main_numbers}" + (f" + {o.special_numbers} ĐB" if spec.separate_special else "")
        md.append(f"| {o.kind} | {sel} | {o.plays:,} | {fmt_vnd(o.cost)} | {r.p_any_prize:.2%} | {r.p_any_prize_same_budget_single_tickets:.2%} | {r.p_profit:.2%} | 1/{1 / p_jp:,.0f} |")
        rows.append(o.model_dump() | {"p_any": r.p_any_prize, "p_singles": r.p_any_prize_same_budget_single_tickets, "p_profit": r.p_profit, "p_jackpot": p_jp})
    return md + [""], rows


def table_section(spec: GameSpec, level: int, specials: int = 1) -> list[str]:
    rows = bao_prize_table(spec, level, specials)
    head = f"**{spec.display_name} — Bao {level}**" + (f" (+ {specials} số đặc biệt)" if spec.separate_special else "")
    md = [head, "", "| Số trúng trong bộ chọn | Số phụ / ĐB | Xác suất | Giải nhận được |", "|---:|---|---:|---|"]
    lab = {"-": "—", "in_set": "số phụ trong bộ", "not_in_set": "số phụ ngoài bộ", "in_fixed": "số phụ trong 5 số", "not_in_fixed": "số phụ ngoài 5 số", "matched": "trùng ĐB", "missed": "không trùng ĐB"}
    for r in rows:
        md.append(f"| {r['hits_in_set']} | {lab.get(r['bonus'], r['bonus'])} | {r['probability']:.3e} | {r['vietlott_style']} |")
    return md + [""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100_000)
    a = ap.parse_args()
    out: dict = {}
    md = ["# Chơi bao Vietlott — danh mục, bảng giải, so sánh chiến thuật", ""]

    md += ["## 1. Danh mục bao và xác suất chính xác", "", "Bộ số minh họa trải đều; mọi bộ số cùng kiểu bao có cùng xác suất.", ""]
    out["catalog"] = {}
    for spec in (MEGA_645, POWER_655, LOTTO_535):
        sec, rows = catalog_section(spec)
        md += sec
        out["catalog"][spec.code.value] = rows

    md += ["## 2. Bảng giải thưởng (tính chính xác, khớp các bảng Vietlott/đại lý công bố)", ""]
    for spec, lv, s in ((MEGA_645, 5, 1), (MEGA_645, 7, 1), (MEGA_645, 8, 1), (MEGA_645, 10, 1), (POWER_655, 5, 1), (POWER_655, 7, 1), (POWER_655, 8, 1), (LOTTO_535, 4, 1), (LOTTO_535, 6, 1), (LOTTO_535, 5, 6)):
        md += table_section(spec, lv, s)

    md += ["## 3. Cùng tiền, khác hình dạng: bao vs bao rút gọn vs vé lẻ", ""]
    out["compare"] = []
    for spec, nums, sp in (
        (MEGA_645, [3, 9, 14, 22, 31, 38, 41, 44], None),
        (MEGA_645, [3, 9, 14, 22, 31, 35, 38, 41, 43, 44], None),
        (POWER_655, [4, 15, 26, 37, 48, 50, 53], None),
        (LOTTO_535, [5, 12, 19, 26, 33, 35], [7]),
    ):
        c = compare_bao_strategies(spec, nums, sp, sims=a.sims, seed=0)
        out["compare"].append(c.model_dump())
        md += [f"**{spec.display_name} — {c.bao}** {nums}", "", "| Cách chơi | Lượt | Chi phí (đ) | P(trúng ≥ 1 giải) | P(thưởng ≥ 1 triệu) | P(thưởng ≥ tiền vé) | Thưởng TB (đ) | Độ lệch chuẩn (đ) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in c.rows:
            md.append(f"| {r.strategy} | {r.plays} | {fmt_vnd(r.cost)} | {r.p_any_prize:.2%} | {r.p_payout_at_least_1m:.2%} | {r.p_profit:.2%} | {fmt_vnd(r.expected_fixed_payout)} | {fmt_vnd(r.sd_payout)} |")
        md.append("")
    md.append(f"Mô phỏng {a.sims:,} kỳ, chỉ giải cố định (Jackpot có xác suất bằng nhau giữa các cách chơi cùng số lượt khác nhau).")
    md.append("")

    md += ["## 4. Kỳ vọng sau thuế khi có người trúng chung", ""]
    out["ev"] = []
    md += ["| Vé | Jackpot | Vé bán (mô hình doanh số) | RTP không ai trúng chung | RTP có người trúng chung | Thuế theo vé vs theo lượt (kỳ vọng sau thuế) |", "|---|---:|---:|---:|---:|---:|"]

    def sold_at(spec: GameSpec, jackpot: float, fallback: int) -> int:
        """Tickets sold at this jackpot from the calibrated sales model (data/calibration)."""
        from vietlott_engine.game_theory.market import load_sales_model

        sm = load_sales_model(spec.code, Path(os.environ.get("VQE_CALIBRATION_DIR") or ROOT / "data" / "calibration"))
        return int(sm.predict(jackpot)) if sm is not None else fallback

    for spec, nums, jp, fallback in (
        (MEGA_645, [3, 9, 14, 22, 31, 38, 41], {"jackpot1": 158.8e9}, 3_600_000),
        (MEGA_645, [3, 9, 14, 22, 31, 38, 41, 44], {"jackpot1": 158.8e9}, 3_600_000),
        (POWER_655, [4, 15, 26, 37, 48, 50, 53], {"jackpot1": 60e9, "jackpot2": 5e9}, 1_500_000),
        (POWER_655, [4, 15, 26, 37, 48], {"jackpot1": 60e9, "jackpot2": 5e9}, 1_500_000),
    ):
        sold = sold_at(spec, jp["jackpot1"], fallback)
        base = analyse_bao(spec, nums, None, jp, after_tax=True)
        per_play = analyse_bao(spec, nums, None, jp, after_tax=True, tax_basis="play")
        ev = bao_ev(spec, nums, None, jp, sold)
        out["ev"].append({"game": spec.code.value, "kind": base.kind, "jackpots": jp, "tickets_sold": sold, "rtp_no_sharing": base.return_to_player, "rtp_sharing": ev.return_to_player, "net_ticket": base.expected_payout_net, "net_play": per_play.expected_payout_net})
        md.append(f"| {spec.display_name} {base.kind} | {fmt_vnd(sum(jp.values()))} | {sold:,} | {base.return_to_player:.1%} | {ev.return_to_player:.1%} | {fmt_vnd(base.expected_payout_net)} vs {fmt_vnd(per_play.expected_payout_net)} |")
    md.append("")

    md += ["## 5. Max 3D, Max 3D+, Max 3D Pro", "", "| Sản phẩm | Giải cao nhất | Xác suất | RTP (2 số khác nhau) | P(trúng ≥ 1 giải) |", "|---|---:|---:|---:|---:|"]
    out["max3d"] = {}
    for prod in max3d.PRODUCTS.values():
        s = max3d.product_summary(prod)
        out["max3d"][prod.code] = s.model_dump()
        md.append(f"| {prod.display_name} | {fmt_vnd(prod.tiers[0].value)} | 1/{s.top_prize_odds:,.0f} | {s.rtp_distinct_numbers:.1%} | {s.p_any_prize:.2%} |")
    md += ["", "| Bao | Lượt | Chi phí (đ) | P(trúng ≥ 1 giải) | Cùng tiền, lượt ngẫu nhiên | P(thưởng ≥ tiền vé) | P(giải cao nhất) |", "|---|---:|---:|---:|---:|---:|---:|"]
    out["max3d_bao"] = []
    for prod, kind, nums in (
        (max3d.MAX3D_PRO, "bao_bo_so", ["123", "456"]),
        (max3d.MAX3D_PRO, "bao_bo_so", ["112", "456"]),
        (max3d.MAX3D_PRO, "bao_nhieu_bo_so", ["123", "456", "789"]),
        (max3d.MAX3D_PRO, "bao_nhieu_bo_so", ["027", "135", "246", "381", "459", "570", "618", "702", "864", "993"]),
        (max3d.MAX3D, "dao_so", ["123"]),
        (max3d.MAX3D, "bao_vi_tri", ["1*3"]),
        (max3d.MAX3D_PLUS, "dao_so", ["123", "456"]),
    ):
        r = max3d.analyse_bao(prod, kind, nums, sims=a.sims)
        out["max3d_bao"].append(r.model_dump())
        md.append(f"| {prod.display_name} {kind} {'-'.join(nums) if len(nums) <= 3 else f'{len(nums)} số'} | {r.plays} | {fmt_vnd(r.cost)} | {r.p_any_prize:.2%} | {r.p_any_prize_same_budget_random:.2%} | {r.p_profit:.2%} | 1/{1 / r.p_top_prize:,.0f} |")

    OUT.mkdir(exist_ok=True)
    (OUT / "bao_vietlott.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "bao_vietlott.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
