"""Regenerate ``reports/vietlott_v3.md`` / ``.json`` from the bundled data:

official prize-data coverage (vietlott.vn detail pages) and how it compares with the
third-party tables used before v3.2, market calibration (payout share, sales, crowd
behaviour), EV with the calibrated crowd and sales response, bao tickets vs single tickets,
coverage portfolios, and Lotto 5/35 rolldowns read from the official jackpot series
(two-phase accrual, next-rolldown forecast and its backtest).

    python scripts/market_report.py            # uses data/calibration (builds it if missing)
    python scripts/market_report.py --rebuild  # refit the market models first (~1 min)
    python scripts/market_report.py --fast     # smaller Monte Carlo for the coverage table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.core.games import GAMES, LOTTO_535, MEGA_645, POWER_655, get_game  # noqa: E402
from vietlott_engine.crawler.pipeline import SyncPipeline  # noqa: E402
from vietlott_engine.crawler.prize_sources import records_from_jsonl  # noqa: E402
from vietlott_engine.crawler.sources.mirror import JsonlFileSource  # noqa: E402
from vietlott_engine.crawler.storage import InMemoryRepository  # noqa: E402
from vietlott_engine.game_theory.bao import analyse_bao  # noqa: E402
from vietlott_engine.game_theory.calibration import load_calibration  # noqa: E402
from vietlott_engine.game_theory.coverage import optimise_coverage  # noqa: E402
from vietlott_engine.game_theory.ev import EVCalculator, optimize_tickets  # noqa: E402
from vietlott_engine.game_theory.market import MarketReport, build_all  # noqa: E402

SEED = Path(os.environ.get("VQE_SEED_FILE_DIR") or ROOT / "data" / "seed")
CAL = Path(os.environ.get("VQE_CALIBRATION_DIR") or ROOT / "data" / "calibration")
OUT = Path(os.environ.get("VQE_REPORTS_DIR") or ROOT / "reports")


def load_repo() -> InMemoryRepository:
    repo = InMemoryRepository()
    for spec in GAMES.values():
        asyncio.run(SyncPipeline(JsonlFileSource(SEED), repo).run(spec))
        f = SEED / f"prizes_{spec.code.value}.jsonl"
        if f.exists():
            repo.upsert_prizes(records_from_jsonl(f.read_text(encoding="utf-8")))
    return repo


def market(game: str) -> MarketReport:
    return MarketReport.model_validate_json((CAL / f"market_{game}.json").read_text(encoding="utf-8"))


def ty(x: float) -> str:
    return f"{x / 1e9:,.1f} tỷ"


def calc_for(spec, repo) -> EVCalculator:  # type: ignore[no-untyped-def]
    cal = load_calibration(spec.code, CAL)
    rep = market(spec.code.value)
    pop = cal.popularity_model(repo.load_history(spec.code))
    calc = EVCalculator(spec, pop, sales=rep.sales_model)
    if rep.sales_model is None and rep.tickets_sold is not None:
        calc.DEFAULT_SOLD = int(rep.tickets_sold.median)
    return calc


def ev_section(repo) -> tuple[list[str], list[dict]]:  # type: ignore[no-untyped-def]
    rows, md = [], ["| Game · Jackpot · vé bán (mô hình) | Vé mẫu hình | Vé \"ngày sinh\" | Vé ít người chọn (tối ưu) | Jackpot hòa vốn (vé ít người chọn, có phản ứng doanh số) |", "|---|---|---|---|---|"]
    scen = [
        (MEGA_645, 30e9, None, [1, 2, 3, 4, 5, 6], [3, 7, 8, 19, 25, 31], None),
        (MEGA_645, 100e9, None, [1, 2, 3, 4, 5, 6], [3, 7, 8, 19, 25, 31], None),
        (MEGA_645, 158.8e9, None, [1, 2, 3, 4, 5, 6], [3, 7, 8, 19, 25, 31], None),
        (POWER_655, 60e9, 5e9, [1, 2, 3, 4, 5, 6], [3, 7, 8, 19, 25, 31], None),
        (POWER_655, 200e9, 10e9, [1, 2, 3, 4, 5, 6], [3, 7, 8, 19, 25, 31], None),
        (LOTTO_535, 6e9, None, [1, 2, 3, 4, 5], [3, 7, 8, 19, 25], 8),
        (LOTTO_535, 11e9, None, [1, 2, 3, 4, 5], [3, 7, 8, 19, 25], 8),
    ]
    calcs = {}
    for spec, j1, j2, pattern, bday, sp in scen:
        calc = calcs.setdefault(spec.code, calc_for(spec, repo))
        opt = optimize_tickets(calc, 1, j1, j2, None, iterations=8000, seed=0).tickets[0]
        sold, src = calc.resolve_sold(None, j1)
        a = calc.evaluate(pattern, j1, j2, None, special=sp)
        b = calc.evaluate(bday, j1, j2, None, special=sp)
        c = calc.evaluate(opt.numbers, j1, j2, None, special=opt.special)
        be = c.break_even_jackpot1_with_sales_response
        be_fixed = c.break_even_jackpot1
        rows.append({"game": spec.code.value, "jackpot1": j1, "jackpot2": j2, "tickets_sold": sold, "sold_source": src,
                     "pattern": {"ticket": pattern, "special": sp, "rtp": a.return_to_player, "ratio": a.popularity_ratio},
                     "birthday": {"ticket": bday, "special": sp, "rtp": b.return_to_player, "ratio": b.popularity_ratio},
                     "optimised": {"ticket": opt.numbers, "special": opt.special, "rtp": c.return_to_player, "ratio": c.popularity_ratio},
                     "break_even_with_sales": be, "break_even_fixed_sales": be_fixed})
        if be is not None:
            be_txt = ty(be)
        elif calc.sales is None and be_fixed is not None:
            be_txt = f"{ty(be_fixed)} ở doanh số cố định" + (" — vượt ngưỡng chia giải 12 tỷ ⇒ kỳ thường luôn âm" if spec.rolldown and be_fixed > spec.rolldown.threshold else "")
        else:
            be_txt = "không đạt (doanh số tăng nhanh hơn Jackpot)"
        md.append(
            f"| {spec.display_name} · {ty(j1)} · {sold / 1e6:.2f} tr ({src}) | RTP {a.return_to_player:.1%} (×{a.popularity_ratio:.1f}) | "
            f"RTP {b.return_to_player:.1%} (×{b.popularity_ratio:.2f}) | {opt.numbers}{' + ' + str(opt.special) if opt.special else ''}: RTP {c.return_to_player:.1%} (×{c.popularity_ratio:.2f}) | "
            f"{be_txt} |"
        )
    return md, rows


def bao_section() -> tuple[list[str], list[dict]]:
    cases = [
        (MEGA_645, [5, 12, 19, 26, 33], None),
        (MEGA_645, [5, 12, 19, 26, 33, 40, 44], None),
        (MEGA_645, [5, 12, 19, 26, 33, 40, 44, 45], None),
        (MEGA_645, [5, 12, 19, 26, 33, 40, 41, 42, 43, 44], None),
        (POWER_655, [5, 12, 19, 26, 33], None),
        (POWER_655, [5, 12, 19, 26, 33, 40, 44], None),
        (LOTTO_535, [5, 12, 19, 26], [7]),
        (LOTTO_535, [5, 12, 19, 26, 33, 35], [7]),
        (LOTTO_535, [5, 12, 19, 26, 33], [1, 2, 3, 4, 5, 6]),
    ]
    md = ["| Game | Kiểu | Số lượt | Chi phí | P(trúng ≥ 1 giải) | Cùng tiền, vé lẻ độc lập | Lượt trúng kỳ vọng |", "|---|---|---:|---:|---:|---:|---:|"]
    rows = []
    for spec, nums, sp in cases:
        r = analyse_bao(spec, nums, sp)
        exp_wins = r.plays * spec.p_any_prize
        label = r.kind + (f" + {len(sp)} số ĐB" if sp and "đặc biệt" not in r.kind else "")
        md.append(f"| {spec.display_name} | {label} | {r.plays} | {r.cost:,} | {r.p_any_prize:.2%} | {r.p_any_prize_same_budget_single_tickets:.2%} | {exp_wins:.2f} |")
        rows.append({"game": spec.code.value, "kind": label, "plays": r.plays, "cost": r.cost, "p_any": r.p_any_prize, "p_singles": r.p_any_prize_same_budget_single_tickets, "rtp_min_jackpot": r.return_to_player})
    return md, rows


def coverage_section(fast: bool) -> tuple[list[str], list[dict]]:
    ev = 60_000 if fast else 200_000
    sims = 12_000 if fast else 30_000
    cases = [(MEGA_645, 10), (MEGA_645, 28), (POWER_655, 10), (POWER_655, 28), (LOTTO_535, 6), (LOTTO_535, 12)]
    md = ["| Game | Số vé | P(≥1 giải) tối ưu [95%] | Vé ngẫu nhiên | Cận trên B·p | Số giải kỳ vọng (mọi danh mục) |", "|---|---:|---:|---:|---:|---:|"]
    rows = []
    for spec, b in cases:
        r = optimise_coverage(spec, b, sim_draws=sims, eval_draws=ev, seed=0)
        lo, hi = r.p_at_least_one_ci95
        md.append(f"| {spec.display_name} | {b} | {r.p_at_least_one:.1%} [{lo:.1%}, {hi:.1%}] | {r.p_random_tickets:.1%} | {r.upper_bound:.1%} | {r.expected_prizes_per_draw:.2f} |")
        rows.append(r.model_dump())
    return md, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    repo = load_repo()
    if a.rebuild or not (CAL / "market_power655.json").exists():
        build_all(repo, CAL, SEED / "jackpot_anchors.json")
    out: dict = {}
    md = ["# Vietlott v3.3 — dữ liệu chính thức, hành vi người chơi, EV, bao, độ phủ, chia giải", ""]

    odr = json.loads((SEED / "OFFICIAL_DATA_REPORT.json").read_text(encoding="utf-8"))
    md += ["## 1. Dữ liệu chính thức (trang chi tiết kỳ quay vietlott.vn)", "",
           "| Game | Kỳ quay | Kỳ có số người trúng từng giải | Kỳ có giá trị Jackpot | So với dữ liệu v3.1 (bên thứ ba) |", "|---|---:|---:|---:|---|"]
    for g in ("mega645", "power655", "lotto535"):
        r = odr[g]
        c = r["prize_comparison_with_previous_seed"]
        cmp_txt = f"{c.get('common', 0)} kỳ chung: số người trúng khớp {c.get('winner_counts_identical', 0)}"
        if c.get("pots_compared"):
            cmp_txt += f"; Jackpot khớp {c['pots_identical']}/{c['pots_compared']} (lệch lớn nhất ×{c['pot_max_relative_difference']:.1f})"
        if r.get("added_vs_previous_seed"):
            cmp_txt += f"; +{r['added_vs_previous_seed']} kỳ mới"
        md.append(f"| {get_game(g).display_name} | {r['draws']} (#{r['first_id']} → #{r['last_id']}, {r['last_date']}) | {r['prize_records']} | {r['prize_records_with_jackpot_pots']} | {cmp_txt} |")
    md += ["", "| Sản phẩm | Kỳ quay | Từ – đến | Mã kỳ thiếu trong khoảng | Nguồn |", "|---|---:|---|---:|---|"]
    for g, name in (("keno", "Keno"), ("bingo18", "Bingo18"), ("max3d", "Max 3D / 3D+"), ("max3dpro", "Max 3D Pro"), ("max4d", "Max 4D (đã ngừng)")):
        r = odr[g]
        md.append(f"| {name} | {r['draws']:,} | {r['first_date']} → {r['last_date']} | {r['missing_ids_inside_range']:,} | {r['source']} |")
    md += ["", "Từ v3.2 dữ liệu giải thưởng Mega/Power/Lotto là bảng giải trên trang chi tiết từng kỳ của vietlott.vn (giá trị Jackpot × số vé trúng = quỹ Jackpot kỳ đó). "
           "Dữ liệu Power 6/55 của v3.1 sai ở một số kỳ (xem cột cuối) và đã được thay. Từ v3.3, kỳ còn thiếu trong bản ghi chính thức được bổ sung từ kho cộng đồng NhanAZ-Data/vietlott-research (Lotto #920, nguồn xosominhngoc.net.vn, đánh dấu `secondary_source`); khi hai nguồn lệch nhau, bản ghi chính thức được giữ (vd. Jackpot kỳ chia giải Lotto #874: 30,22 tỷ chính thức so với 35,96 tỷ ở trang phụ) — chi tiết `data/seed/DATA_MERGE_REPORT.json`."]
    out["prize_data"] = odr

    md += ["", "## 2. Thị trường & hành vi người chơi", ""]
    out["market"] = {}
    for g in ("power655", "mega645", "lotto535"):
        rep = market(g)
        cal = rep.calibration
        lc = rep.level_check
        fs = cal.fit
        md.append(f"### {get_game(g).display_name}")
        if rep.payout_share_used is not None:
            md.append(f"- Tỷ lệ doanh thu vào quỹ Jackpot + giải cố định: s = {rep.payout_share_used:.3f} ({rep.payout_share_method}).")
        if rep.tickets_sold:
            t = rep.tickets_sold
            md.append(f"- Vé bán mỗi kỳ: trung vị {t.median:,.0f} (p10 {t.p10:,.0f} – p90 {t.p90:,.0f}); {t.method}.")
        if rep.sales_model:
            sm = rep.sales_model
            md.append(f"- Doanh số theo Jackpot: độ co giãn {sm.elasticity:.2f} ± {sm.elasticity_se:.2f} tại Jackpot trung vị, độ cong {sm.curvature:+.3f} ± {sm.curvature_se:.3f} (HAC); dự báo 50 tỷ → {sm.predict(50e9) / 1e6:.2f} tr vé, 150 tỷ → {sm.predict(150e9) / 1e6:.2f} tr vé.")
        md.append(f"- Quick pick ≈ {cal.quick_pick_share:.0%} ± {cal.quick_pick_share_se:.0%}; số được chọn nhiều nhất {cal.most_popular_numbers[:8]}, ít nhất {cal.least_popular_numbers[:8]}.")
        if fs.holdout_loglik_gain_per_draw is not None:
            span = f"toàn bộ {fs.holdout_draws} kỳ — mô hình chuyển từ Power, không ước lượng trên game này" if fs.holdout_draws >= rep.draws_with_prizes else f"{fs.holdout_draws} kỳ cuối"
            md.append(f"- Kiểm tra ngoài mẫu ({span}): {fs.holdout_loglik_gain_per_draw:+.3f} log-lik/kỳ so với đám đông chọn đều, p = {fs.holdout_lr_p_value:.2g}.")
        tc = rep.transfer_check
        if tc and tc.get("selected"):
            md.append(f"- Chọn mô hình đám đông: {'chuyển từ Power 6/55' if tc['selected'] == 'power_transfer' else 'ước lượng riêng'} "
                      f"(mô hình riêng: ngoài mẫu p = {tc['own_holdout_p_value']:.2f}, kiểm tra mức {tc['level_check_own']['correlation']:.2f}; chuyển từ Power: kiểm tra mức {tc['level_check_power_transfer']['correlation']:.2f}). "
                      "Không mô hình nào vượt đám đông chọn đều trong kiểm định likelihood, nên con số ×r của Mega chỉ mang tính tham khảo.")
        md.append(f"- Kiểm tra mức (không cần doanh số): tương quan {lc.correlation:.2f}, hệ số góc {lc.slope:.2f}, p hoán vị {lc.permutation_p_value:.4f}.")
        sp = cal.popularity_spread
        md.append("- Độ phổ biến của vé ngẫu nhiên theo mô hình: " + ", ".join(f"{k} {v:.2f}" for k, v in sp.items()) + ".")
        md += [f"- {n}" for n in rep.notes] + [""]
        out["market"][g] = rep.model_dump(mode="json", exclude={"jackpot_path", "calibration"}) | {"calibration_fit": fs.model_dump(), "quick_pick_share": cal.quick_pick_share}

    md += ["## 3. EV với mô hình đã hiệu chỉnh (sau thuế, 1 lượt 10.000 đ)", "", "(×r = độ phổ biến so với vé chọn đều; số vé bán lấy từ mô hình doanh số khi có)", ""]
    sec, out["ev"] = ev_section(repo)
    md += sec

    md += ["", "## 4. Chơi bao vs vé lẻ (chính xác, không mô phỏng)", ""]
    sec, out["bao"] = bao_section()
    md += sec + ["", "Kỳ vọng tiền thưởng trên mỗi đồng là như nhau; bao gom giải thành từng cụm nên xác suất trúng *ít nhất một* giải thấp hơn nhiều so với cùng số tiền mua vé lẻ."]

    md += ["", "## 5. Danh mục tối đa hóa P(trúng ít nhất 1 giải)", ""]
    sec, out["coverage"] = coverage_section(a.fast)
    md += sec

    rep = market("lotto535")
    off, fc, bt, rh = rep.official_rolldowns, rep.rolldown_forecast, rep.rolldown_forecast_backtest, rep.rolldowns
    if off is not None:
        acc = off.accrual
        md += ["", "## 6. Lotto 5/35 — chia giải Độc đắc (chuỗi Jackpot chính thức)", "",
               f"{off.executed} kỳ chia giải đã thực hiện, {off.pre_empted} lần công bố nhưng có người trúng Độc đắc trước. RTP thực tế trung bình của kỳ chia giải {off.mean_realised_rtp:.2f} (trước thuế); "
               f"xu hướng giảm theo thời gian (Spearman ρ = {off.rtp_trend.get('spearman_rho_vs_time', float('nan')):+.2f}, p = {off.rtp_trend.get('p_value', float('nan')):.1g}) vì càng về sau càng nhiều người dồn vào kỳ chia giải.", "",
               "**Hai pha tích lũy Jackpot** (suy ra từ giá trị công bố, không phải quy định đã công bố):", "",
               f"- Pha chậm: Jackpot tăng {acc.slow_rate:.3f} × doanh thu (cùng giải cố định: {acc.slow_total_rate:.3f}).",
               f"- Pha nhanh: {acc.fast_rate:.3f} × doanh thu (cùng giải cố định: {acc.fast_total_rate:.3f}).",
               f"- Chuyển pha ở mức Jackpot trung vị {ty(acc.switch_level_median)} ({ty(acc.switch_level_range[0])} – {ty(acc.switch_level_range[1])}).",
               f"- Phần chênh lệch trong pha chậm ≈ {ty(acc.diverted_per_restart_median)} cho mỗi lần Jackpot khởi động lại ({len(acc.diverted_per_segment)} đoạn) — khớp với việc bù lại 6 tỷ khởi điểm.", ""]
        if fc is not None:
            md += [f"**Dự báo** (sau kỳ #{fc.last_draw_id}, {fc.last_date}): Jackpot {ty(fc.pot)}, pha {fc.phase}, còn ≈ {fc.draws_to_threshold:.1f} kỳ đến ngưỡng 12 tỷ → {fc.expected_rolldown_window}."]
        if bt and bt.get("cases"):
            md += [f"Kiểm định lại dự báo trên {bt['cases']} thời điểm: sai số tuyệt đối trung vị {bt['median_abs_error_draws']:.1f} kỳ, {bt['share_within_3_draws']:.0%} trong ±3 kỳ, Spearman {bt['spearman_rho']:.2f}.", ""]
        md += ["| Kỳ | Ngày | Jackpot chia (công bố) | Vé bán (ước tính) | ×doanh số | RTP thực (trước thuế) | RTP mô hình (sau thuế) |", "|---|---|---:|---:|---:|---:|---:|"]
        for e in off.events:
            if e.executed:
                md.append(f"| #{e.draw_id} | {e.date} | {ty(e.jackpot_published or e.jackpot_estimate)} | {e.tickets_sold / 1e6:.2f} tr | {e.sales_multiple:.1f} | {e.realised_rtp_pre_tax:.2f} | {e.model_rtp_after_tax:.2f} |")
            else:
                md.append(f"| #{e.draw_id} (Độc đắc trúng trước khi chia) | {e.date} | {ty(e.jackpot_published or e.jackpot_estimate)} | {e.tickets_sold / 1e6:.2f} tr | {e.sales_multiple:.1f} | — | — |")
        rc = off.reconstruction_check
        if rc:
            md += ["", f"Kiểm tra phương pháp tái dựng của v3.1 (không có chuỗi Jackpot, nhận diện qua doanh số tăng đột biến): khớp {rc['matched']}/{rc['official_events']} sự kiện, "
                   f"bỏ sót {rc['missed']}, sai số Jackpot trung vị {rc['pot_error_median_abs']:.1%} (lớn nhất {rc['pot_error_max_abs']:.1%})."]
        out["rolldowns_official"] = off.model_dump(mode="json")
        out["rolldown_forecast"] = fc.model_dump(mode="json") if fc else None
        out["rolldown_forecast_backtest"] = bt
    elif rh is not None:
        md += ["", "## 6. Lotto 5/35 — các kỳ chia giải Độc đắc (tái dựng)", "",
               f"{rh.announced} kỳ chia giải được công bố (nhận diện bằng doanh số tăng ≥ 3×), {rh.executed} kỳ thực hiện chia. "
               f"Tốc độ tích lũy Jackpot c = {rh.accrual_rate:.3f}/đồng doanh thu, ngày chia giải c_r = {rh.rolldown_day_rate:.2f}.", ""]
        out["rolldowns"] = rh.model_dump(mode="json")

    OUT.mkdir(exist_ok=True)
    (OUT / "vietlott_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "vietlott_v3.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
