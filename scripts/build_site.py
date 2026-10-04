#!/usr/bin/env python3
"""Build the static site published to GitHub Pages: ``site/index.html`` (results) and ``forecast.html`` (analysis) plus JSON.

    python scripts/build_site.py [--out site] [--dir data/forecast]

Reads the saved forecaster state of each product (``vietlott forecast next`` / ``update`` write
it) and the ledger; nothing is learnt here. Products without a saved state are skipped.

JSON (for other dashboards, e.g. VLA): ``data/summary.json`` (one row per product),
``data/forecast/<product>.json`` (full forecast + e-value path), ``data/scoreboard.json``,
``data/ledger.jsonl`` (every recorded forecast, scored or not).
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vietlott_engine import __version__  # noqa: E402
from vlm.web.dashboard import build_dashboard  # noqa: E402
from vietlott_engine.core.products import PRODUCT_INFO, ProductCode  # noqa: E402
from vietlott_engine.forecast.engine import LEDGER, Forecaster, scoreboard, state_path, vn, vn_int  # noqa: E402

THRESHOLD = 1.30103  # log10 20


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def factor(log10_value: float) -> str:
    """e-value as a multiplier a reader can picture: ×0,35 · ×1,58 · ×372 · ×10^5,2."""
    v = 10 ** log10_value
    if v >= 10_000:
        return f"×10^{vn(log10_value, 1)}"
    return f"×{vn(v, 2 if v < 10 else 0)}"


def sparkline(path: list[list[float]], width: int = 320, height: int = 72) -> str:
    """Inline SVG of log10 e-value by draw, with the ×20 evidence line."""
    if len(path) < 2:
        return '<p class="muted">Chưa đủ kỳ để vẽ.</p>'
    ys = [p[1] for p in path]
    lo, hi = min(min(ys), -1.0), max(max(ys), THRESHOLD + 0.3)
    pad = 4

    def y(v: float) -> float:
        return pad + (hi - v) / (hi - lo) * (height - 2 * pad)

    n = len(path)
    pts = " ".join(f"{pad + i / (n - 1) * (width - 2 * pad):.1f},{y(v):.1f}" for i, v in enumerate(ys))
    above = max(ys) >= THRESHOLD
    label = (f"Đường e-value theo kỳ, từ kỳ #{int(path[0][0])} đến #{int(path[-1][0])}: cao nhất {factor(max(ys))}, hiện {factor(ys[-1])}; "
             f"{'đã' if above else 'chưa'} vượt ngưỡng ×20.")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
        f'<line x1="{pad}" x2="{width - pad}" y1="{y(THRESHOLD):.1f}" y2="{y(THRESHOLD):.1f}" class="thr"/>'
        f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>'
        f'<polyline points="{pts}" class="{"line hit" if above else "line"}"/>'
        f'<text x="{width - pad}" y="{y(THRESHOLD) - 3:.1f}" class="thr-label">×20</text></svg>'
    )


def product_section(rep: dict, path: list[list[float]], board: dict | None) -> str:
    ev = rep["evidence"]
    status = "có bằng chứng" if ev["found"] else ("không dùng làm bằng chứng" if not ev.get("valid", True) else "chưa có bằng chứng")
    head = f"Dự báo kỳ #{rep['target_id']}" if rep.get("target_id") else "Đã ngừng phát hành: chỉ phân tích lịch sử"
    parts = [
        f'<section class="card" id="{esc(rep["product"])}" aria-labelledby="h-{esc(rep["product"])}">',
        f'<header><h2 id="h-{esc(rep["product"])}">{esc(rep["display_name"])}</h2>'
        f'<span class="badge {"ok" if ev["found"] else "no"}">{esc(status)}</span></header>',
        f'<p class="meta">{esc(head)} · học từ {vn_int(rep["draws"])} kỳ, dữ liệu tới #{esc(rep["last_id"])} ngày {esc(rep["last_date"])}</p>',
        f'<p>{esc(rep["verdict"])}</p>',
        '<div class="split"><div>',
        "<h3>E-value theo kỳ</h3>", sparkline(path),
        f'<p class="muted">Cao nhất {factor(ev["max_log10_wealth"])}, hiện {factor(ev["log10_wealth"])}. Vượt ×20 mới là bằng chứng (α = 5%).</p>',
        "</div><div>",
    ]
    if board and board.get("scored"):
        parts.append(f'<h3>Sổ dự báo</h3><p>{board["recorded"]} dự báo ghi trước kỳ quay, {board["scored"]} đã chấm: trùng {vn(board["hits"], 0)} so với kỳ vọng ngẫu nhiên {vn(board["expected"], 1)} (×{vn(board["ratio"], 3)}).</p>')
    elif board:
        parts.append(f'<h3>Sổ dự báo</h3><p>{board["recorded"]} dự báo đã ghi, chưa kỳ nào được chấm.</p>')
    else:
        parts.append('<h3>Sổ dự báo</h3><p class="muted">Chưa có dự báo nào được ghi trước kỳ quay.</p>')
    parts.append("</div></div>")
    for c in rep["components"]:
        label = "" if len(rep["components"]) == 1 else f" — {esc(c['name'])}"
        if c.get("set"):
            sf = c["set"]
            p0 = sf["numbers"][0]["p"] / sf["numbers"][0]["lift"]
            rows = "".join(f'<tr><th scope="row">{esc(x["symbol"])}</th><td>{vn(100 * x["p"], 3)}%</td><td>×{vn(x["lift"], 4)}</td></tr>' for x in sf["numbers"][:10])
            parts.append(f'<h3>Xác suất có mặt kỳ tới{label}</h3><table><caption>10 số cao nhất theo mô hình; ngẫu nhiên {vn(100 * p0, 2)}% mỗi số</caption>'
                         f'<thead><tr><th scope="col">Số</th><th scope="col">Xác suất</th><th scope="col">So với ngẫu nhiên</th></tr></thead><tbody>{rows}</tbody></table>')
            if len(sf["ticket"]) <= 6:
                parts.append(f'<p>Bộ đề xuất <strong>{" ".join(f"{x:02d}" for x in sf["ticket"])}</strong>: P(trúng cả bộ) theo mô hình 1/{vn_int(1 / sf["p_ticket_model"])}, ngẫu nhiên 1/{vn_int(1 / sf["p_ticket_fair"])}.</p>')
            if sf.get("keno"):
                krows = "".join(f'<tr><th scope="row">{t["bac"]}</th><td>{" ".join(f"{x:02d}" for x in t["numbers"])}</td><td>{vn(t["rtp_model"], 4)}</td><td>{vn(t["rtp_fair"], 4)}</td></tr>' for t in sf["keno"])
                parts.append('<table><caption>Keno: vé tốt nhất mỗi bậc theo mô hình (RTP = tiền thưởng kỳ vọng / tiền vé)</caption><thead><tr><th scope="col">Bậc</th><th scope="col">Số</th>'
                             f'<th scope="col">RTP mô hình</th><th scope="col">RTP ngẫu nhiên</th></tr></thead><tbody>{krows}</tbody></table>')
        if c.get("digit"):
            df = c["digit"]
            prow = "".join(f'<tr><th scope="row">{esc(p["label"])}</th>' + "".join(f'<td>{esc(s["symbol"])} <span class="muted">×{vn(s["lift"], 3)}</span></td>' for s in p["symbols"][:3]) + "</tr>" for p in df["positions"])
            parts.append(f'<h3>Ký hiệu mạnh nhất mỗi vị trí{label}</h3><table><caption>Ba ký hiệu cao nhất mỗi vị trí, so với ngẫu nhiên</caption>'
                         f'<thead><tr><th scope="col">Vị trí</th><th scope="col">1</th><th scope="col">2</th><th scope="col">3</th></tr></thead><tbody>{prow}</tbody></table>')
            tops = ", ".join(f"<strong>{esc(x['symbol'])}</strong> ×{vn(x['lift'], 3)}" for x in df["top_numbers"][:5])
            parts.append(f"<p>Số đề xuất: {tops}.</p>")
            if df.get("bets"):
                brows = "".join(f'<tr><th scope="row">{esc(b["bet"])}</th><td>{vn(b["rtp_model"], 4)}</td><td>{vn(b["rtp_fair"], 4)}</td></tr>' for b in df["bets"][:4])
                parts.append(f'<table><caption>Cửa có RTP cao nhất theo mô hình</caption><thead><tr><th scope="col">Cửa</th><th scope="col">RTP mô hình</th><th scope="col">RTP ngẫu nhiên</th></tr></thead><tbody>{brows}</tbody></table>')
    parts.append("</section>")
    return "\n".join(parts)



def build(out: Path, directory: Path, *, data_dir: Path | None = None, seed_dir: Path | None = None, journal: Path | None = None) -> dict:
    from vietlott_engine.core.config import get_settings
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    seed_dir = seed_dir or settings.product_seed_dir
    journal = journal or (settings.auto_update_dir / 'results.jsonl' if settings.auto_update_dir else data_dir / 'results' / 'results.jsonl')
    dashboard = build_dashboard(data_dir, seed_dir, directory, journal)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'scripts' / 'site_assets', out / 'assets', dirs_exist_ok=True)
    (out / 'assets' / 'index.html').unlink(missing_ok=True)
    (out / "data" / "forecast").mkdir(parents=True, exist_ok=True)
    boards = {r["product"]: r for r in scoreboard(directory)}
    summary, sections = [], []
    for code in ProductCode:
        p = state_path(directory, code)
        if not p.exists():
            continue
        f = Forecaster.load(p)
        rep = f.forecast().model_dump(mode="json")
        main_comp = next(iter(f.state["components"].values()))
        path = [[float(a), float(b)] for a, b in main_comp["path"]]
        (out / "data" / "forecast" / f"{code.value}.json").write_text(json.dumps({**rep, "evidence_path": path}, ensure_ascii=False), encoding="utf-8")
        c0 = rep["components"][0]
        summary.append({
            "product": code.value, "name": PRODUCT_INFO[code].display_name, "draws": rep["draws"], "last_id": rep["last_id"], "last_date": rep["last_date"],
            "target_id": rep["target_id"], "evidence_found": rep["evidence"]["found"], "log10_wealth": rep["evidence"]["log10_wealth"],
            "max_log10_wealth": rep["evidence"]["max_log10_wealth"], "uniform_weight": c0["uniform_weight"], "verdict": rep["verdict"],
            "scoreboard": boards.get(code.value),
        })
        sections.append(product_section(rep, path, boards.get(code.value)))
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"generated_at": generated, "version": __version__, "products": summary, "dashboard": dashboard["stats"]}
    (out / "data" / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "data" / "scoreboard.json").write_text(json.dumps(list(boards.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    ledger = directory / LEDGER
    if ledger.exists():
        shutil.copyfile(ledger, out / "data" / "ledger.jsonl")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    rows = "".join(
        f'<tr><th scope="row"><a href="#{esc(r["product"])}">{esc(r["name"])}</a></th><td>{vn_int(r["draws"])}</td><td>{esc(r["last_date"])}</td>'
        f'<td>{factor(r["max_log10_wealth"])} / {factor(r["log10_wealth"])}</td><td>{"<strong>có</strong>" if r["evidence_found"] else "không"}</td>'
        f'<td>{vn(100 * r["uniform_weight"], 1)}%</td>'
        f'<td>{(str(r["scoreboard"]["scored"]) + " đã chấm, ×" + vn(r["scoreboard"]["ratio"], 3)) if r["scoreboard"] and r["scoreboard"].get("scored") else "—"}</td></tr>'
        for r in summary
    )
    page = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM · Phân tích & tính toán dự báo</title>
<meta name="description" content="Dự báo tự học có kiểm chứng cho 8 sản phẩm Vietlott: xác suất kỳ tới, chỉ số bằng chứng e-value và sổ dự báo ghi trước kỳ quay.">
<script src="assets/theme.js"></script>
<link rel="stylesheet" href="assets/styles.css">
</head>
<body>
<header class="site-header"><div class="nav-inner"><a class="brand" href="index.html"><span class="brand-mark">V</span><span>VLM<small>VIETLOTT MASTER</small></span></a><nav aria-label="Điều hướng"><a class="nav-link" href="index.html">Kết quả & đối chiếu</a><a class="nav-link active" href="forecast.html">Phân tích</a></nav><button class="icon-button theme-toggle" type="button" aria-label="Chuyển giao diện tối">◐</button></div></header>
<main class="analysis-main">
<a class="text-link" href="index.html">← Trang kết quả, dự báo & đối chiếu</a>
<h1>Phân tích & tính toán dự báo</h1>
<p class="lede">Dự báo tự học cho 8 sản phẩm Vietlott, kèm phép đo cho biết mô hình có thật sự hơn ngẫu nhiên không.</p>
<p class="note">Nếu máy quay công bằng, không ai dự đoán được kết quả với xác suất cao. Mô hình ở đây học sau mỗi kỳ và tự đo bằng
<strong>e-value</strong> (hợp lệ ở mọi thời điểm): chỉ khi vượt 20 mới là bằng chứng. Ngay cả khi có bằng chứng, mọi cửa vẫn có kỳ vọng âm.
Đây là công cụ phân tích, không phải lời khuyên đặt cược.</p>
<div class="table-wrap"><table>
<caption>Tóm tắt — cập nhật {esc(generated)} (UTC), phiên bản {esc(__version__)}</caption>
<thead><tr><th scope="col">Sản phẩm</th><th scope="col">Kỳ đã học</th><th scope="col">Dữ liệu tới</th><th scope="col">E-value cao nhất / hiện tại</th>
<th scope="col">Bằng chứng</th><th scope="col">Trọng số "ngẫu nhiên"</th><th scope="col">Sổ dự báo</th></tr></thead>
<tbody>{rows}</tbody></table></div>
{''.join(sections)}
<footer>
<p>Dữ liệu máy đọc: <a href="data/summary.json">summary.json</a> · <a href="data/scoreboard.json">scoreboard.json</a> · <a href="data/ledger.jsonl">ledger.jsonl</a> · data/forecast/&lt;sản phẩm&gt;.json.</p>
<p>Nguồn dữ liệu: kho cộng đồng NhanAZ-Data/vietlott-research, pqminh-4/vietlott-data, vietvudanh/vietlott-data (MIT) và trang chính thức vietlott.vn khi truy cập được. Mã nguồn MIT.</p>
</footer>
</main>
</body>
</html>
"""
    (out / "forecast.html").write_text(page, encoding="utf-8")
    serialized = json.dumps(dashboard, ensure_ascii=False, allow_nan=False)
    (out / 'data' / 'dashboard.json').write_text(serialized, encoding='utf-8')
    homepage = (ROOT / 'scripts' / 'site_assets' / 'index.html').read_text(encoding='utf-8')
    (out / 'index.html').write_text(homepage.replace('__DASHBOARD_JSON__', serialized.replace('<', '\\u003c')), encoding='utf-8')
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--dir", default=None, help="forecaster state + ledger (default: the settings, <data_dir>/forecast)")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Vietnamese output through a Windows pipe (cp1252)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if a.dir is None:
        from vietlott_engine.core.config import get_settings

        s = get_settings()
        a.dir = str(s.forecast_dir or s.data_dir / "forecast")
    meta = build(Path(a.out), Path(a.dir))
    print(f"{meta['dashboard']['results']} sản phẩm có kết quả → {Path(a.out) / 'index.html'}; phân tích → forecast.html")
    return 0 if meta["dashboard"]["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
