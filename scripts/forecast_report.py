#!/usr/bin/env python3
"""reports/forecast.{md,json}: the self-learning forecaster on the bundled history of all eight
products, plus a simulation check of its evidence (size under fair draws, power on a planted
bias of the size seen in Max 3D).

    python scripts/forecast_report.py [--sims 200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from math import log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine.core.products import ProductCode  # noqa: E402
from vietlott_engine.crawler.product_store import ProductStore  # noqa: E402
from vietlott_engine.forecast.data import DigitSpec, SetSpec, digit_counts, load_series  # noqa: E402
from vietlott_engine.forecast.engine import Component, Forecaster, vn, vn_int  # noqa: E402

SEED = Path(os.environ.get("VQE_PRODUCT_SEED_DIR") or ROOT / "data" / "seed")
OUT = Path(os.environ.get("VQE_REPORTS_DIR") or ROOT / "reports")


def _run(comp: Component, obs: dict) -> tuple[float, float]:
    """(max, final) cumulative log wealth (natural log) of one run."""
    st = comp.init_state()
    t = len(next(iter(obs.values())))
    incr, _ = comp.process(st, obs, np.arange(1, t + 1))
    lw = np.cumsum(incr)
    return float(lw.max()), float(lw[-1])


def _set_obs(nums: np.ndarray, n: int) -> dict:
    x = np.zeros((len(nums), n), dtype=bool)
    x[np.repeat(np.arange(len(nums)), nums.shape[1]), nums.ravel() - 1] = True
    return {"X": x, "bonus": np.zeros(len(nums), dtype=np.int16)}


def calibration(sims: int) -> dict:
    """Size under fair draws; power and draw-down under a *constant* units-6 bias of the size seen."""
    thr = log(20)
    set_rej = sum(_run(Component("main", SetSpec(45, 6), 20, 1 / 1560), _set_obs(np.argpartition(np.random.default_rng(10_000 + s).random((500, 45)), 6, axis=1)[:, :6] + 1, 45))[0] >= thr for s in range(sims))
    dspec = DigitSpec(10, 3, 20, ("trăm", "chục", "đơn vị"))
    dig_rej = sum(_run(Component("digits", dspec, 20, 1 / 1560), {"C": digit_counts(np.random.default_rng(20_000 + s).integers(0, 10, (500, 20, 3)), 10)})[0] >= thr for s in range(sims))
    probs = np.full(10, 0.89 / 9)
    probs[6] = 0.11
    runs = max(sims // 2, 20)
    out = {"sims": sims, "size_set_6_45_T500": set_rej / sims, "size_digit_max3d_T500": dig_rej / sims, "power_runs": runs}
    for length, key in ((1139, "max3d"), (786, "max3dpro")):
        found, drops = 0, []
        for s in range(runs):
            rng = np.random.default_rng(30_000 + s + length)
            d = rng.integers(0, 10, (length, 20, 3))
            d[:, :, 2] = rng.choice(10, size=(length, 20), p=probs)
            mx, fin = _run(Component("digits", dspec, 20, 1 / 1560), {"C": digit_counts(d, 10)})
            found += mx >= thr
            drops.append((mx - fin) / log(10))
        out[f"power_{key}_T{length}"] = found / runs
        out[f"drops_{key}"] = drops
    return out


def _timeline(f: Forecaster, series) -> dict:  # type: ignore[no-untyped-def]
    """Exact (per-draw) e-value path: first crossing of 20×, share of later draws above it, start of
    the current run above it, peak and current value."""
    if not f.trace:
        return {}
    ids = np.concatenate([a for a, _ in f.trace])
    lw = np.concatenate([b for _, b in f.trace])
    thr = np.log10(20)

    def day(i: int) -> str:
        return str(series.dates[int(np.searchsorted(series.draw_ids, int(i)))])

    above = lw >= thr
    peak = int(lw.argmax())
    out = {"peak_id": int(ids[peak]), "peak_date": day(ids[peak]), "peak_log10": float(lw[peak]), "now_log10": float(lw[-1]), "draws": int(lw.size)}
    if above.any():
        first = int(np.flatnonzero(above)[0])
        below_after = np.flatnonzero(~above[first:])
        out.update({
            "first_cross_id": int(ids[first]), "first_cross_date": day(ids[first]),
            "share_above_after_first": float(above[first:].mean()), "draws_after_first": int(lw.size - first),
        })
        if above[-1]:
            start = first + (int(below_after[-1]) + 1 if below_after.size else 0)
            out.update({"run_start_id": int(ids[start]), "run_start_date": day(ids[start])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    a = ap.parse_args()
    reports, rows, timeline, keno_fit = {}, [], {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        store = ProductStore(Path(tmp), SEED)  # bundled data only: reproducible
        for code in ProductCode:
            t0 = time.time()
            f = Forecaster(code, keep_trace=True)
            series = load_series(code, store=store, seed_dir=SEED)
            f.update(series)
            rep = f.forecast()
            reports[code.value] = rep.model_dump()
            timeline[code.value] = _timeline(f, series)
            if code == ProductCode.KENO:
                comp, names = f.state["components"]["main"], f.components["main"].names
                keno_fit = {n: float(comp["cum_ll"][names.index(n)] / log(10)) for n in ("hot_ewma10", "hot_all_a50", "cold_all_a50", "overdue", "repeat", "logistic")}
            c = rep.components[0]
            non_uniform = next(e for e in c.experts if e.name != "uniform")
            rows.append({
                "product": code.value, "name": rep.display_name, "draws": rep.draws, "seconds": round(time.time() - t0, 1),
                "max": rep.evidence.max_log10_wealth, "now": rep.evidence.log10_wealth, "found": rep.evidence.found,
                "uniform": c.uniform_weight, "top": f"{non_uniform.name} ({vn(100 * non_uniform.weight, 1)}%)",
                "ratio": c.pick_score.ratio, "z": c.pick_score.z,
                "lift": (c.set.numbers[0].lift if c.set else max(s.lift for p in c.digit.positions for s in p.symbols)),  # type: ignore[union-attr]
                "verdict": rep.verdict,
            })
    cal = calibration(a.sims)
    OUT.mkdir(exist_ok=True)
    (OUT / "forecast.json").write_text(json.dumps({"products": reports, "summary": rows, "timeline": timeline, "keno_strategy_fit_log10": keno_fit, "calibration": cal}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    md = [
        "# Dự báo tự học cho mọi sản phẩm (v3.5) — kết quả trên dữ liệu thật",
        "",
        "`vietlott forecast next` · `python scripts/forecast_report.py`. Mỗi *chuyên gia* (ngẫu nhiên đều, số nóng/lạnh toàn bộ lịch sử và gần đây, "
        "số lâu chưa về / vừa về, lặp kỳ trước, hồi quy logistic học trực tuyến, Dirichlet cho trò chữ số, và một rổ giả thuyết \"một số / một chữ số lệch ±10–25%\") "
        "đưa ra một phân phối xác suất đầy đủ cho kỳ tới, chỉ từ các kỳ trước. Hỗn hợp Bayes *fixed-share* tự chỉnh trọng số sau mỗi kỳ theo mức mỗi chuyên gia đoán đúng. "
        "Tỷ số hợp lý của hỗn hợp so với máy quay công bằng là một **e-value hợp lệ ở mọi thời điểm**: vượt 20 (10^1,30) mới là bằng chứng (α = 5%).",
        "",
        "| Sản phẩm | Kỳ đã học | e-value cao nhất / hiện tại | Bằng chứng? | Trọng số \"ngẫu nhiên\" | Chuyên gia khác nặng nhất | Chấm lùi: trùng × ngẫu nhiên (z) | Hệ số mạnh nhất mô hình tự cho |",
        "|---|---:|---|---|---:|---|---|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {vn_int(r['draws'])} | 10^{vn(r['max'])} / 10^{vn(r['now'])} | {'**có**' if r['found'] else 'không'} | {vn(100 * r['uniform'], 1)}% | {r['top']} | ×{vn(r['ratio'], 3)} ({vn(r['z'])}) | ×{vn(r['lift'], 3)} |"
        )
    md += [
        "",
        f"**Kiểm tra trên dữ liệu mô phỏng** ({cal['sims']} lịch sử mỗi loại, 500 kỳ): máy quay công bằng 6/45 → e-value vượt 20 ở {vn(100 * cal['size_set_6_45_T500'], 1)}% số lịch sử; "
        f"kiểu Max 3D → {vn(100 * cal['size_digit_max3d_T500'], 1)}% (giới hạn lý thuyết 5%). Khi chữ số 6 hàng đơn vị ra 11% thay vì 10% (cỡ lệch thấy ở Max 3D), "
        f"mô hình tìm ra ở {vn(100 * cal['power_max3d_T1139'], 0)}% số lịch sử dài 1.139 kỳ và {vn(100 * cal['power_max3dpro_T786'], 0)}% số lịch sử dài 786 kỳ ({cal['power_runs']} lịch sử mỗi loại).",
        "",
        "## Kỳ tới theo từng sản phẩm",
    ]
    for code, rep in reports.items():
        md += ["", f"### {rep['display_name']} — kỳ #{rep['target_id']}", "", rep["verdict"], ""]
        tl = timeline.get(code, {})
        if rep["evidence"]["found"] and tl.get("first_cross_id"):
            drops = np.asarray(cal.get(f"drops_{code}", []))
            drop = tl["peak_log10"] - tl["now_log10"]
            run = f"liên tục trên ngưỡng từ #{tl['run_start_id']} ({tl['run_start_date']})" if tl.get("run_start_id") else "hiện dưới ngưỡng"
            line = (
                f"E-value vượt 20 lần đầu ở kỳ #{tl['first_cross_id']} ({tl['first_cross_date']}); từ đó trên ngưỡng ở {vn(100 * tl['share_above_after_first'], 0)}% "
                f"trong {vn_int(tl['draws_after_first'])} kỳ, {run}. Cao nhất 10^{vn(tl['peak_log10'])} ở #{tl['peak_id']} ({tl['peak_date']}), nay 10^{vn(tl['now_log10'])}."
            )
            if drops.size:
                share = float((drops >= drop).mean())
                if share >= 0.10:
                    line += (f" Mức giảm từ đỉnh ({vn(drop)}) không chứng tỏ độ lệch yếu đi: với một độ lệch *không đổi* cùng cỡ, "
                             f"{vn(100 * share, 0)}% lịch sử mô phỏng giảm ít nhất chừng ấy.")
                else:
                    line += (f" Mức giảm từ đỉnh ({vn(drop)}) khá hiếm nếu độ lệch không đổi (chỉ {vn(100 * share, 0)}% lịch sử mô phỏng giảm chừng ấy): "
                             "gợi ý độ lệch yếu hơn sau đỉnh — chỉ là gợi ý, vì mốc đỉnh được chọn sau khi xem dữ liệu.")
            md += [line, ""]
        for c in rep["components"]:
            if c["set"]:
                sf = c["set"]
                md.append("- Xác suất cao nhất: " + ", ".join(f"{x['symbol']} ({vn(100 * x['p'], 3)}%, ×{vn(x['lift'], 4)})" for x in sf["numbers"][:6]) + f" — ngẫu nhiên {vn(100 * sf['numbers'][0]['p'] / sf['numbers'][0]['lift'], 2)}%.")
                if len(sf["ticket"]) <= 6:
                    md.append(f"- Bộ đề xuất {' '.join(f'{x:02d}' for x in sf['ticket'])}: P(trúng cả bộ) 1/{vn_int(1 / sf['p_ticket_model'])} theo mô hình, 1/{vn_int(1 / sf['p_ticket_fair'])} ngẫu nhiên.")
                if sf["keno"]:
                    md.append("- Keno, vé tốt nhất mỗi bậc theo mô hình (RTP mô hình / ngẫu nhiên): " + "; ".join(f"bậc {t['bac']}: {vn(t['rtp_model'], 4)} / {vn(t['rtp_fair'], 4)}" for t in sf["keno"]) + ".")
            if c["digit"]:
                df = c["digit"]
                for pos in df["positions"]:
                    md.append(f"- {pos['label']}: " + ", ".join(f"{x['symbol']} ({vn(100 * x['p'], 2)}%, ×{vn(x['lift'], 3)})" for x in pos["symbols"][:3]))
                md.append("- Số đề xuất: " + ", ".join(f"{x['symbol']} (×{vn(x['lift'], 3)})" for x in df["top_numbers"]))
                for b in df["bets"][:3]:
                    md.append(f"- {b['bet']}: RTP {vn(b['rtp_model'], 4)} theo mô hình, {vn(b['rtp_fair'], 4)} ngẫu nhiên.")
    md += [
        "",
        "## Đọc kết quả thế nào",
        "",
        "- **Không có bằng chứng** (Mega, Power, Lotto, Keno, Bingo18, Max 4D): trọng số dồn về chuyên gia \"ngẫu nhiên\" và các giả thuyết lệch nhẹ "
        "(Keno, Bingo18 — nhiều dữ liệu nhất — 98–99% cho \"ngẫu nhiên\"); các hệ số × mô hình tự cho (thường ×1,00x–×1,02) là dao động của dữ liệu, không phải lợi thế. "
        "Chấm lùi trên toàn bộ lịch sử: lựa chọn của mô hình trùng trong phạm vi ngẫu nhiên (|z| < 2).",
        "- **Có bằng chứng** (Max 3D, Max 3D Pro): mô hình chỉ ra chữ số 6 hàng đơn vị mà không được lập trình để tìm nó (giả thuyết lệch được đặt đều cho mọi chữ số ở mọi vị trí). "
        "Nhưng nó chạy trên cùng dữ liệu đã cho phát hiện ở mục 1.6, nên đây là cùng bằng chứng nhìn theo cách khác, không phải xác nhận độc lập; trên Max 4D nó không thấy độ lệch này. "
        "Lợi thế ≈ ×1,075 cho chữ số 6 (×1,08 cho số đề xuất, phần nhỉnh thêm từ hai chữ số kia chưa có bằng chứng), đưa RTP Max 3D từ 0,545 lên ≈ 0,59: **vẫn mất trung bình ~41% tiền vé**.",
        "- Thành tích thật chỉ được tính từ các dự báo ghi *trước* kỳ quay (`forecast next` → `forecast update` → `forecast scoreboard`). "
        "Sổ là tệp cục bộ để tự theo dõi, không phải bằng chứng chống sửa; với Keno/Bingo18, hãy đồng bộ ngay trước khi dự báo.",
    ]
    if keno_fit:
        md += [
            "",
            "## Các cách chọn số phổ biến, xem như mô hình xác suất (Keno, 297 nghìn kỳ)",
            "",
            "log10 tỷ số hợp lý so với máy quay công bằng: mô hình gán xác suất cho kết quả thật kém máy công bằng bao nhiêu bậc 10. "
            "Đây là độ khớp của *mô hình*; một vé chọn theo số nóng vẫn có xác suất trúng như mọi vé khác nếu máy công bằng. Độ lớn chủ yếu phản ánh mô hình nghiêng mạnh đến đâu.",
            "",
            "| Mô hình | log10 tỷ số hợp lý |",
            "|---|---:|",
        ] + [f"| {name} | {vn(val, 1)} |" for name, val in sorted(keno_fit.items(), key=lambda x: x[1])]
    (OUT / "forecast.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
