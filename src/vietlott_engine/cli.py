"""Command-line interface.

Examples
    vietlott sync --game all                       # GitHub mirror → DuckDB + Parquet
    vietlott sync --game mega645 --source vietlott # official site
    vietlott sync --game all --source file --path ./seed
    vietlott analyze --game power655
    vietlott ev --game mega645 --ticket 3 17 22 35 41 44 --jackpot1 45e9
    vietlott optimize --game power655 --n 5 --jackpot1 120e9
    vietlott wheel --game mega645 --pool 3 7 11 15 19 23 27 31 35 39 --t 3 --m 4
    vietlott backtest --game mega645 --tickets 5 --out reports/
    vietlott serve --port 8000

  Vietlott products / market (v3)
    vietlott prizes --game all                     # winner counts + jackpot pots → store
    vietlott market --out data/calibration         # payout share, sales, crowd calibration, rolldowns
    vietlott odds --game lotto535
    vietlott ev --game lotto535 --ticket 3 14 22 29 33 --special 7 --jackpot1 9e9
    vietlott bao --game mega645 --numbers 5 12 19 26 33 40 44 45
    vietlott coverage --game power655 --budget 20
    vietlott rolldown --jackpot 20e9 --tickets-sold 2000000
    vietlott rolldown --history                   # official series + next-rolldown forecast
    vietlott bao --game power655 --catalog
    vietlott bao --game power655 --table 7
    vietlott bao --game mega645 --numbers 3 9 14 22 31 38 41 44 --compare
    vietlott max3d --product max3dpro --kind bao_bo_so --numbers 123 456

  All products / official data (v3.2)
    vietlott products list                         # coverage of Keno, Bingo18, Max 3D, Max 3D Pro
    vietlott products sync                         # auto: vietlott.vn → community archive → mirrors
    vietlott products sync --source vietlott        # vietlott.vn directly (Vietnamese IP only)
    vietlott products sync --source nhanaz --path ~/vietlott-research   # local archive clone
    vietlott products sync --source v130 --path ~/Vietlott-Quant-Engine-1.3.0
    vietlott sync --game all --source auto         # Mega/Power/Lotto with the same fallback
    vietlott products analyze --product all        # randomness tests + Max 3D digit cross-check
    vietlott products odds --product keno
    vietlott products import-pages --path ~/Downloads/vietlott --dry-run   # pages / HAR you saved
    vietlott prizes --game all --source canonical  # official prize tables (detail pages)
    vietlott prizes --game mega645 --source vietlott --last 10

  Self-learning forecaster (v3.5)
    vietlott forecast next --product all          # learn new draws, forecast, record in the ledger
    vietlott forecast update                      # after a draw: learn it and score recorded forecasts
    vietlott forecast scoreboard                  # live track record of recorded forecasts
    vietlott forecast fit --product keno          # relearn from scratch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from vietlott_engine.core.config import get_settings
from vietlott_engine.core.games import GAMES, get_game
from vietlott_engine.core.logging import configure_logging, get_logger

log = get_logger("vqe.cli")

STRATEGY_NAMES = ["random", "hot", "cold", "overdue", "bayes", "markov", "gcn", "anti_popularity", "wheel"]


def _repo():
    from vietlott_engine.api.main import build_repository

    return build_repository(get_settings())


def _games(arg: str) -> list:
    return list(GAMES.values()) if arg == "all" else [get_game(arg)]


def _print(obj: object, as_json: bool = True) -> None:
    if hasattr(obj, "model_dump_json"):
        print(obj.model_dump_json(indent=2))  # type: ignore[attr-defined]
    elif as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        print(obj)


def cmd_sync(a: argparse.Namespace) -> int:
    from vietlott_engine.crawler.pipeline import SyncPipeline, build_http_client, build_source

    s = get_settings()
    repo = _repo()

    async def run() -> list[dict]:
        from vietlott_engine.crawler.prize_sources import records_from_jsonl

        out = []
        async with build_http_client(s) as client:
            src = build_source(s, client, a.source, Path(a.path) if a.path else None)
            for spec in _games(a.game):
                rep = (await SyncPipeline(src, repo).run(spec, full_refresh=a.full)).to_dict()
                rep.update(_archive_extras(src, spec, repo))
                prize_file = Path(a.path) / f"prizes_{spec.code.value}.jsonl" if a.source == "file" and a.path else None
                if prize_file is not None and prize_file.exists():  # offline snapshot also carries prize data
                    rep["prize_records"] = repo.upsert_prizes(records_from_jsonl(prize_file.read_text(encoding="utf-8")))
                out.append(rep)
        return out

    _print(asyncio.run(run()))
    return 0


def _archive_extras(src, spec, repo) -> dict:  # type: ignore[no-untyped-def]
    """Fallback attempts, and prize tables that came with archive draws (nhanaz / v130)."""
    from vietlott_engine.crawler.official_data import import_canonical
    from vietlott_engine.crawler.sources.fallback import ArchiveDrawSource, FallbackDrawSource

    out: dict = {}
    used = src
    if isinstance(src, FallbackDrawSource):
        out["attempts"] = src.log.to_list()
        used = next((x for x in src.sources if x.name == src.log.used), None)
    if isinstance(used, ArchiveDrawSource) and used.last_records:
        prizes = import_canonical(used.last_records, spec.code).prizes
        out["prize_records"] = repo.upsert_prizes(prizes) if prizes else 0
    return out


def cmd_analyze(a: argparse.Namespace) -> int:
    from vietlott_engine.analytics.cooccurrence import cooccurrence_report
    from vietlott_engine.analytics.gaps import gap_report
    from vietlott_engine.analytics.randomness import randomness_report
    from vietlott_engine.probability.bayesian import bayesian_report
    from vietlott_engine.probability.markov import markov_report

    repo = _repo()
    for spec in _games(a.game):
        h = repo.load_history(spec.code)
        rr = randomness_report(h, mc_sims=a.sims, seed=a.seed)
        print(f"\n=== {spec.display_name}: {len(h)} draws ({rr.first_date} → {rr.last_date}) ===")
        for t in rr.tests:
            print(f"  {t.name:<24} stat={t.statistic:>10.3f}  p={t.p_value:.4f}  q={t.q_value:.3f}")
        print("  →", rr.verdict)
        g = gap_report(h)
        print(f"  gaps: geometric fit p={g.geometric_fit.p_value:.3f}; hazard by gap = " + ", ".join(f"{r.gap}:{r.hazard:.3f}" for r in g.hazard[:6]) + " … (constant ⇒ no 'overdue' effect)")
        c = cooccurrence_report(h, mc_sims=a.sims // 3, seed=a.seed)
        print(f"  pairs: {c.significant_pairs} significant at FDR 5%; over-dispersion p={c.dispersion_test.p_value:.3f}")
        b = bayesian_report(h)
        print(f"  Dirichlet: log BF(heterogeneous vs uniform)={b.log_bayes_factor_vs_uniform:.1f}; empirical-Bayes α₀={b.empirical_bayes_alpha0:.0f}")
        if b.predictive:
            print(f"  predictive gain vs k/n: {b.predictive.mean_log_score_gain:+.4f} nats/draw (t={b.predictive.t_statistic:+.2f})")
        m = markov_report(h, sims=a.sims // 3, seed=a.seed)
        print(f"  Markov: transition dependence p={m.number_dependence.p_value:.3f}; " + ", ".join(f"{ch.name} MI={ch.mutual_information_bits:.4f} bits (p={ch.p_value:.2f})" for ch in m.feature_chains))
    return 0


def _calculator(spec, quick_pick: float | None = None, use_history: bool = True):  # type: ignore[no-untyped-def]
    """EV calculator with the calibrated crowd + sales model when ``calibration_dir`` has them."""
    from vietlott_engine.game_theory.calibration import load_calibration
    from vietlott_engine.game_theory.ev import EVCalculator
    from vietlott_engine.game_theory.market import load_sales_model
    from vietlott_engine.game_theory.popularity import PopularityModel, PopularityParams

    s = get_settings()
    history = None
    if use_history:
        try:
            history = _repo().load_history(spec.code)
        except Exception:  # no data yet: no "copy last result" effect
            history = None
    cal = load_calibration(spec.code, s.calibration_dir)
    if cal is not None:
        pop = cal.popularity_model(history)
        if quick_pick is not None:
            pop = pop.with_params(quick_pick_share=quick_pick)
    else:
        last = tuple(int(x) for x in history.numbers[-1]) if history is not None and len(history) else None
        pop = PopularityModel(spec, PopularityParams(quick_pick_share=quick_pick or s.quick_pick_share), last_draw=last)
    calc = EVCalculator(spec, pop, sales=load_sales_model(spec.code, s.calibration_dir))
    if calc.sales is None:
        mk = s.calibration_dir / f"market_{spec.code.value}.json"
        if mk.exists():
            sold = json.loads(mk.read_text(encoding="utf-8")).get("tickets_sold") or {}
            calc.DEFAULT_SOLD = int(sold.get("median") or s.default_tickets_sold)
        else:
            calc.DEFAULT_SOLD = s.default_tickets_sold
    return calc


def cmd_ev(a: argparse.Namespace) -> int:
    spec = get_game(a.game)
    calc = _calculator(spec, a.quick_pick)
    _print(calc.evaluate(a.ticket, a.jackpot1, a.jackpot2, a.tickets_sold, special=a.special))
    return 0


def cmd_optimize(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.ev import optimize_tickets

    spec = get_game(a.game)
    calc = _calculator(spec, a.quick_pick)
    _print(optimize_tickets(calc, a.n, a.jackpot1, a.jackpot2, a.tickets_sold, a.max_overlap, tuple(a.exclude or ()), a.iterations, a.seed))
    return 0


def cmd_wheel(a: argparse.Namespace) -> int:
    from vietlott_engine.wheeling.cover import WheelRequest, build_wheel

    spec = get_game(a.game)
    _print(build_wheel(WheelRequest(pool=a.pool, ticket_size=spec.pick, guarantee=a.t, condition=a.m), spec, seed=a.seed))
    return 0


def cmd_backtest(a: argparse.Namespace) -> int:
    from vietlott_engine.backtest.engine import BacktestConfig, WalkForwardBacktester
    from vietlott_engine.backtest.strategies import STRATEGY_REGISTRY, default_strategies
    from vietlott_engine.game_theory.popularity import PopularityModel

    repo = _repo()
    out_dir = Path(a.out) if a.out else None
    for spec in _games(a.game):
        h = repo.load_history(spec.code)
        strategies = [STRATEGY_REGISTRY[s]() for s in a.strategies] if a.strategies else default_strategies(include_ml=not a.no_ml)
        report = WalkForwardBacktester(h, BacktestConfig(start=a.start, end=a.end, tickets_per_draw=a.tickets, seed=a.seed), PopularityModel(spec)).run(strategies)
        md = report.to_markdown()
        print(md)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"backtest_{spec.code.value}.md").write_text(md, encoding="utf-8")
            (out_dir / f"backtest_{spec.code.value}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            log.info("wrote reports to %s", out_dir)
    return 0


def cmd_ml(a: argparse.Namespace) -> int:
    from vietlott_engine.ml_models.features import build_snapshots
    from vietlott_engine.ml_models.gcn import GCNConfig, walk_forward_skill

    repo = _repo()
    out = []
    for spec in _games(a.game):
        h = repo.load_history(spec.code)
        snaps = build_snapshots(h)
        for model in ("gcn", "logistic"):
            rep = walk_forward_skill(snaps, h.k, a.first_test, a.refit_every, GCNConfig(epochs=a.epochs), model=model)
            out.append({"game": spec.code.value, **rep.model_dump()})
    _print(out)
    return 0


def _print_forecast(rep) -> None:  # type: ignore[no-untyped-def]
    """Human-readable forecast (Vietnamese number format)."""
    from vietlott_engine.forecast.engine import vn, vn_int

    head = f"dự báo kỳ #{rep.target_id:05d}" if rep.target_id is not None else "đã ngừng phát hành: chỉ phân tích lịch sử"
    print(f"\n=== {rep.display_name} · {head} (học từ {vn_int(rep.draws)} kỳ, tới #{rep.last_id} ngày {rep.last_date}) ===")
    ev = rep.evidence
    status = "CÓ" if ev.found else ("KHÔNG DÙNG (cửa sổ --last tự chọn)" if not ev.valid else "chưa có")
    print(f"  Bằng chứng (e-value, hợp lệ ở mọi thời điểm): {status} — cao nhất 10^{vn(ev.max_log10_wealth)}, hiện 10^{vn(ev.log10_wealth)}, ngưỡng 10^{vn(ev.threshold_log10)} (×20)")
    for c in rep.components:
        label = "" if len(rep.components) == 1 else f"[{c.name}] "
        print(f"  {label}Trọng số đã học: " + " · ".join(f"{e.name} {vn(100 * e.weight, 1)}%" for e in c.experts[:5]))
        ps = c.pick_score
        print(f"  {label}Chấm điểm lùi {vn_int(ps.draws)} kỳ: lựa chọn của mô hình trùng {vn(ps.hits_per_draw, 3)}/kỳ, ngẫu nhiên {vn(ps.expected_per_draw, 3)} (×{vn(ps.ratio, 3)}, z = {vn(ps.z)}); {ps.recent_draws} kỳ gần nhất {vn(ps.recent_hits_per_draw, 3)}/kỳ")
        if c.set is not None:
            sf = c.set
            p0 = 100 * float(sf.numbers[0].p) / float(sf.numbers[0].lift)
            print(f"  {label}Xác suất từng số (ngẫu nhiên {vn(p0)}%): " + " · ".join(f"{x.symbol}: {vn(100 * x.p, 3)}% (×{vn(x.lift, 4)})" for x in sf.numbers[:8]))
            if len(sf.ticket) <= 6:
                print(f"  {label}Bộ đề xuất: {' '.join(f'{x:02d}' for x in sf.ticket)} — P(trúng cả bộ) mô hình 1/{vn_int(1 / sf.p_ticket_model)} (×{vn(sf.ticket_lift, 4)}), ngẫu nhiên 1/{vn_int(1 / sf.p_ticket_fair)}")
                pm, pf = sum(sf.match_dist_model[3:]), sum(sf.match_dist_fair[3:])
                print(f"  {label}P(trùng ≥ 3 số): mô hình {vn(100 * pm, 3)}% · ngẫu nhiên {vn(100 * pf, 3)}%")
            for t in sf.keno:
                print(f"    Keno bậc {t.bac:>2}: {' '.join(f'{x:02d}' for x in t.numbers):<30} RTP mô hình {vn(t.rtp_model, 4)} · ngẫu nhiên {vn(t.rtp_fair, 4)} · P(có giải) {vn(100 * t.p_any_prize_model)}% vs {vn(100 * t.p_any_prize_fair)}%")
        if c.digit is not None:
            df = c.digit
            for pos in df.positions:
                print(f"  {label}{pos.label:<12}: " + " · ".join(f"{x.symbol} {vn(100 * x.p)}% (×{vn(x.lift, 3)})" for x in pos.symbols[:4]))
            print(f"  {label}Số đề xuất: " + " · ".join(f"{x.symbol} (×{vn(x.lift, 3)})" for x in df.top_numbers) + f" — P(số đầu xuất hiện trong kỳ) {vn(100 * df.p_top_appears_model, 3)}% vs {vn(100 * df.p_top_appears_fair, 3)}%")
            for b in df.bets[:4]:
                prob = f"P {vn(100 * b.p_model, 3)}% vs {vn(100 * b.p_fair, 3)}% · " if b.p_fair >= 1e-4 else ""
                print(f"    {b.bet:<34} {prob}RTP {vn(b.rtp_model, 4)} vs {vn(b.rtp_fair, 4)}")
    print("  → " + rep.verdict)


def cmd_forecast(a: argparse.Namespace) -> int:
    """Self-learning forecaster: fit / update / next / scoreboard (v3.5)."""
    from vietlott_engine.core.products import ProductCode, get_product
    from vietlott_engine.forecast.data import load_series
    from vietlott_engine.forecast.engine import record, refresh, scoreboard, vn, vn_int
    from vietlott_engine.forecast.schedule import record_window

    s = get_settings()
    directory = Path(a.dir) if a.dir else (s.forecast_dir or s.data_dir / "forecast")
    products = list(ProductCode) if a.product == "all" else [get_product(a.product)]
    if a.action == "scoreboard":
        rows = scoreboard(directory, None if a.product == "all" else products[0])
        if a.json:
            _print(rows)
            return 0
        if not rows:
            print("Chưa có dự báo nào được ghi. Chạy `forecast next` trước kỳ quay, rồi `forecast update` sau kỳ quay.")
        for r in rows:
            if not r["scored"]:
                print(f"{r['product']:<10} ghi {r['recorded']:>4} dự báo, chưa kỳ nào được chấm (chờ kỳ quay, rồi `forecast update`)")
                continue
            print(f"{r['product']:<10} ghi {r['recorded']:>4} dự báo, đã chấm {r['scored']:>4}: trùng {vn(r['hits'], 0)} / kỳ vọng ngẫu nhiên {vn(r['expected'], 1)} (×{vn(r['ratio'], 3)})")
        return 0
    repo, store = _repo(), _product_store()
    out = []
    for code in products:
        series = load_series(code, repository=repo, store=store, seed_dir=s.product_seed_dir)
        f, learned, scored = refresh(code, series, directory, refit=a.action == "fit", last=a.last)
        for e in scored:
            comps = e["score"]["components"]
            print(f"  ✓ chấm dự báo {e['product']} kỳ #{e['target_id']}: " + ", ".join(f"{n} trùng {c['hits']} (ngẫu nhiên {vn(c['expected'], 2)})" for n, c in comps.items()))
        if a.action in ("fit", "update"):
            st = f.state
            row = {"product": code.value, "learned_draws": learned, "draws": st["draws"], "last_id": st["last_id"], "log10_wealth": round(st["log_wealth"] / 2.302585093, 3), "max_log10_wealth": round(st["max_log_wealth"] / 2.302585093, 3), "scored_forecasts": len(scored)}
            out.append(row)
            row["evidence_valid"] = st.get("window") is None
            if not a.json:
                note = "" if row["evidence_valid"] else " — cửa sổ --last tự chọn: không dùng làm bằng chứng"
                print(f"{code.value:<10} học thêm {vn_int(learned):>7} kỳ (tổng {vn_int(st['draws'])}, tới #{st['last_id']}); e-value hiện 10^{vn(row['log10_wealth'])}, cao nhất 10^{vn(row['max_log10_wealth'])}{note}")
            continue
        rep = f.forecast()
        if not a.no_record:
            ok, note, target = record_window(code, series.dates)
            if ok or (a.force_record and rep.target_id is not None):
                record(directory, rep, pre_draw=ok, target_time=target.isoformat() if target else None)
            rep.recorded, rep.record_note = ok or (a.force_record and rep.target_id is not None), note if ok else f"{'ghi cưỡng bức (không tính vào thành tích): ' if a.force_record else 'không ghi vào sổ: '}{note}"
            if not a.json:
                print(f"  {'✓' if ok else '!'} Sổ dự báo {code.value}: {rep.record_note}")
        if a.json:
            out.append(rep.model_dump())
        else:
            _print_forecast(rep)
    if a.json:
        _print(out)
    elif a.action == "next" and not a.no_record:
        print(f"\nSổ dự báo: {directory / 'ledger.jsonl'}. Sau kỳ quay chạy `vietlott forecast update` để chấm điểm, `vietlott forecast scoreboard` để xem thành tích thật.")
    return 0


def cmd_inference(a: argparse.Namespace) -> int:
    from vietlott_engine.inference.report import inference_report

    repo = _repo()
    out_dir = Path(a.out) if a.out else None
    for spec in _games(a.game):
        rep = inference_report(repo.load_history(spec.code), sims=a.sims, alpha=a.alpha, seed=a.seed)
        print(f"\n=== {spec.display_name}: {rep.draws} draws ({rep.first_date} → {rep.last_date}) ===")
        for f in rep.findings:
            print("  •", f)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"inference_{spec.code.value}.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    return 0


def cmd_decide(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.calibration import load_calibration
    from vietlott_engine.game_theory.decision import ev_uncertainty, kelly, payout_distribution, payout_profile

    spec = get_game(a.game)
    calc = _calculator(spec)
    sold = a.tickets_sold
    values, probs = payout_distribution(calc, a.ticket, a.jackpot1, a.jackpot2, sold, special=a.special)
    out: dict = {
        "ev": calc.evaluate(a.ticket, a.jackpot1, a.jackpot2, sold, special=a.special).model_dump(),
        "payout": payout_profile(values, probs, spec.ticket_price).model_dump(),
        "kelly": kelly(values, probs, spec.ticket_price, a.bankroll).model_dump(),
    }
    if a.sold_range:
        cal = load_calibration(spec.code, get_settings().calibration_dir)
        out["uncertainty"] = ev_uncertainty(
            spec, a.ticket, a.jackpot1 or spec.min_jackpots["jackpot1"], tuple(a.sold_range), a.jackpot2, sims=a.sims, special=a.special, calibration=cal
        ).model_dump()
    _print(out)
    return 0


def cmd_prizes(a: argparse.Namespace) -> int:
    from vietlott_engine.crawler.pipeline import PrizeSyncPipeline, build_http_client
    from vietlott_engine.crawler.product_store import sync_canonical_prizes

    s = get_settings()
    repo = _repo()

    async def run() -> list[dict]:
        out = []
        async with build_http_client(s) as client:
            for spec in _games(a.game):
                if a.source == "compal":
                    pipe = PrizeSyncPipeline(client, repo, s.prize_winners_base_url, s.prize_power_history_url)
                    out.append((await pipe.run(spec)).to_dict())
                else:
                    rep = await sync_canonical_prizes(
                        client, repo, spec, source=a.source, canonical_base_url=s.canonical_base_url,
                        vietlott_base_url=s.vietlott_base_url, last=a.last, bootstrap_cookie=s.vietlott_cookie_bootstrap,
                        nhanaz_base_url=s.nhanaz_base_url, nhanaz_dir=Path(a.path) if a.path and a.source == "nhanaz" else s.nhanaz_dir,
                        v130_dir=Path(a.path) if a.path and a.source == "v130" else s.v130_dir,
                    )
                    out.append(rep.to_dict())
        return out

    _print(asyncio.run(run()))
    return 0


def _product_store():
    from vietlott_engine.crawler.product_store import ProductStore

    s = get_settings()
    return ProductStore(s.data_dir, s.product_seed_dir)


def _row_id(r: dict) -> int:
    try:
        return int(str(r.get("id", r.get("draw_id"))).strip().lstrip("#"))
    except (TypeError, ValueError):
        return -1


def _import_pages(a: argparse.Namespace, store) -> int:  # type: ignore[no-untyped-def]
    """Pages saved by a person from vietlott.vn (HTML / MHTML / HAR) → verify, then import."""
    from vietlott_engine.core.games import GameCode
    from vietlott_engine.core.products import get_product
    from vietlott_engine.crawler.official_data import import_canonical
    from vietlott_engine.crawler.sources.saved_pages import compare_keno_cells, compare_matrix, compare_products, import_pages

    if not a.path:
        print("--path is required: a saved page, a .har file or a folder of them", file=sys.stderr)
        return 2
    repo = _repo()
    imp = import_pages(a.path, None if a.product == "all" else get_product(a.product))
    report = {
        "summary": imp.summary(),
        "products": compare_products(imp, store),
        "matrix_games": compare_matrix(imp, repo),
        "keno_prize_cells": compare_keno_cells(imp) if imp.keno_cells else None,
        "imported": {},
    }
    if not a.dry_run:
        # A draw whose saved page disagrees with the data is only overwritten on request: the parsers
        # read markup that can change, so a disagreement is shown first and resolved by a person.
        def keep(c: dict):  # type: ignore[no-untyped-def]
            held = set() if a.replace_mismatches else set(c.get("mismatch_ids", []))
            return lambda i: int(i) not in held

        for name, rows in imp.product_rows.items():
            ok = keep(report["products"].get(name, {}))
            chosen = [r for r in rows if ok(_row_id(r))]
            new, rej = store.upsert(get_product(name), chosen, "vietlott.vn/saved")
            report["imported"][name] = {"new_draws": new, "rejected": rej, "held_back_mismatches": len(rows) - len(chosen)}
        for name, recs in imp.canonical.items():
            ok = keep(report["matrix_games"].get(name, {}))
            got = import_canonical(recs, GameCode(name))
            draws = [d for d in got.draws if ok(d.draw_id)]
            prizes = [p for p in got.prizes if ok(p.draw_id)]
            report["imported"][name] = {
                "draws": repo.upsert(draws) if draws else 0,
                "prize_tables": repo.upsert_prizes(prizes) if prizes else 0,
                "held_back_mismatches": len(got.draws) - len(draws),
            }
    if a.json:
        _print(report)
        return 0
    s = report["summary"]
    print(f"{s['documents']} trang/mục đọc được, {s['parsed']} có dữ liệu; {s['by_product_kind']}")
    for u in s["unparsed"]:
        print(f"  ! {u['document']} ({u['product'] or '?'}, {u['kind']}): {u['problem']}")
    for name, c in report["products"].items():
        ub = c["mismatch_rate_upper95"]
        print(f"  {name}: {c['pages_rows']} kỳ trên trang, so được {c['compared']}, giống hệt {c['identical']}" + (f" (tỷ lệ sai ≤ {ub:.1%} với độ tin cậy 95%)" if ub is not None else "") + f", mới {len(c['new_draws'])}")
        for m in c["mismatches"]:
            print(f"     ≠ kỳ {m['draw_id']}: trang {m['page']} | dữ liệu {m['data']}")
    for name, c in report["matrix_games"].items():
        print(f"  {name}: {c['draws']} kỳ, {c['prize_tables']} bảng giải; so được {c['draws_compared']} kỳ / {c['prize_tables_compared']} bảng giải; lệch {len(c['draw_mismatches'])} kỳ, {len(c['prize_mismatches'])} bảng giải; mới {c['new_draws'][:10]}")
        for m in c["prize_mismatches"]:
            print(f"     ≠ kỳ {m['draw_id']}: trang {m['page']} | dữ liệu {m['data']}")
    if report["keno_prize_cells"]:
        k = report["keno_prize_cells"]
        print(f"  Keno: {k['cells_compared']} ô giải so với bảng của engine, khác {len(k['differences'])}: {k['differences'][:5]}")
    print("  " + ("(chạy thử: chưa ghi gì)" if a.dry_run else f"đã nhập: {report['imported']}"))
    held = sum(v.get("held_back_mismatches", 0) for v in report["imported"].values())
    if held:
        print(f"  {held} kỳ lệch với dữ liệu chưa được ghi đè — xem từng chỗ ở trên; nếu trang đúng, chạy lại với --replace-mismatches")
    return 0


def cmd_products(a: argparse.Namespace) -> int:
    from vietlott_engine.core.products import PRODUCT_INFO, ProductCode, get_product
    from vietlott_engine.crawler.product_store import PRODUCTS

    s = get_settings()
    store = _product_store()
    targets = list(PRODUCTS) if a.product == "all" else [get_product(a.product)]
    if a.action == "list":
        cov = {c["product"]: c for c in store.coverage()}
        print(f"{'sản phẩm':<18} {'quay':<34} {'lịch':<36} {'kỳ':>8}  {'từ – đến':<25} thiếu")
        for code in ProductCode:
            info = PRODUCT_INFO[code]
            if info.matrix_game:
                n = len(_repo().load(code.value)) if a.with_matrix else None
                print(f"{info.display_name:<18} {info.draws:<34} {info.schedule:<36} {n if n is not None else '(sync)':>8}")
                continue
            c = cov.get(code.value, {})
            span = f"{c.get('first_date')} – {c.get('last_date')}" if c.get("draws") else "—"
            print(f"{info.display_name:<18} {info.draws:<34} {info.schedule:<36} {c.get('draws', 0):>8,}  {span:<25} {c.get('missing_ids_inside_range', '—'):,}")
        return 0
    if a.action == "sync":
        from vietlott_engine.crawler.pipeline import build_http_client
        from vietlott_engine.crawler.product_store import ProductSyncPipeline

        async def run() -> list[dict]:
            out = []
            async with build_http_client(s) as client:
                pipe = ProductSyncPipeline(
                    client, store, vietlott_base_url=s.vietlott_base_url, mirror_base_url=s.product_mirror_base_url,
                    canonical_base_url=s.canonical_base_url, bootstrap_cookie=s.vietlott_cookie_bootstrap, pages_per_batch=s.max_concurrency,
                    nhanaz_base_url=s.nhanaz_base_url, nhanaz_dir=Path(a.path) if a.path and a.source == "nhanaz" else s.nhanaz_dir,
                    v130_dir=Path(a.path) if a.path and a.source == "v130" else s.v130_dir, fallback_order=s.fallback_order,
                )
                for code in targets:
                    out.append((await pipe.run(code, a.source, a.max_pages or s.product_sync_max_pages, a.full)).to_dict())
            return out

        _print(asyncio.run(run()))
        return 0
    if a.action == "analyze":
        from vietlott_engine.analytics.products import digit_replication, product_randomness

        out = []
        for code in targets:
            rep = product_randomness(store.load(code, include_unconfirmed=a.include_unconfirmed))
            if a.json:
                out.append(rep.model_dump())
                continue
            print(f"\n=== {PRODUCT_INFO[code].display_name}: {rep.draws:,} kỳ ({rep.first_date} → {rep.last_date}), {rep.consecutive_pairs:,} cặp kỳ liền nhau ===")
            for t in rep.tests:
                flag = "  ← q < 0.05" if t.q_value < 0.05 else ""
                print(f"  {t.name:<44} stat {t.statistic:10.2f}  p {t.p_value:.4f}  q {t.q_value:.3f}{flag}")
                if t.detail:
                    print(f"      {t.detail}")
            print(f"  → {rep.verdict}")
        if {ProductCode.MAX3D, ProductCode.MAX3D_PRO} <= set(targets):
            rep = digit_replication(store.load(ProductCode.MAX3D), store.load(ProductCode.MAX3D_PRO))
            if a.json:
                out.append(rep.model_dump())
            else:
                c = rep.cell
                print(f"\n=== Kiểm tra chéo Max 3D → Max 3D Pro: chữ số {c.digit} hàng {c.position} ===")
                print(f"  phát hiện ({rep.discovery}) {c.discovery_share:.4f} · xác nhận ({rep.confirmation}) {c.confirm_share:.4f} ({c.confirm_hits:,}/{c.confirm_trials:,}), p một phía {c.confirm_p_value:.2e}")
                print(f"  gộp {rep.pooled_share:.4f} [{rep.pooled_ci95[0]:.4f}, {rep.pooled_ci95[1]:.4f}]; số lợi nhất {rep.best_number} ×{rep.per_number_multiplier_max:.3f}")
                print(f"  {rep.note}")
            if ProductCode.MAX4D in targets:
                from vietlott_engine.analytics.products import digit_cell_test

                cell = digit_cell_test(store.load(ProductCode.MAX4D), 4, 0, rep.cell.digit, "greater" if rep.cell.confirm_z > 0 else "less")
                if a.json:
                    out.append(cell.model_dump())
                else:
                    print(f"\n=== Max 4D (đặt trước giả thuyết): chữ số {cell.digit} hàng {cell.position} = {cell.share:.4f} ({cell.hits:,}/{cell.trials:,}), p một phía {cell.p_value:.3f}")
        if a.json:
            _print(out)
        return 0
    if a.action == "import-pages":
        return _import_pages(a, store)
    if a.action == "odds":
        from vietlott_engine.game_theory.fastgames import bingo18_odds, keno_odds

        for code in targets:
            if code == ProductCode.KENO:
                bacs, sides = keno_odds()
                if a.json:
                    _print({"bac": [b.model_dump() for b in bacs], "side_bets": [x.model_dump() for x in sides]})
                    continue
                print("\n=== Keno (10.000 đ/vé) ===")
                for b in bacs:
                    var = "  " + ", ".join(f"[{k}: RTP {v:.3f}]" for k, v in b.variants.items()) if b.variants else ""
                    print(f"  bậc {b.bac:>2}: P(trúng) {b.p_any_prize:.4f}  giải cao nhất 1/{b.one_in_top_prize:,.0f}  RTP {b.return_to_player:.3f}{var}")
                for x in sides:
                    rtp = f"{x.return_to_player:.3f}" if x.return_to_player is not None else "—"
                    alts = ", ".join(f"{k} → {v:.3f}" for k, v in x.alternatives_rtp.items())
                    print(f"  {x.name:<22} P {x.probability:.4f}  RTP {rtp}{'  (khác: ' + alts + ')' if alts else ''}{'' if x.verified else '  [chưa xác minh]'}")
            elif code == ProductCode.BINGO18:
                rows = bingo18_odds()
                if a.json:
                    _print([r.model_dump() for r in rows])
                    continue
                print("\n=== Bingo18 (10.000 đ/vé) ===")
                for r in rows:
                    print(f"  {r.bet:<26} {r.condition:<34} P {r.probability:.4f}  trả {r.payout:<26} RTP {r.return_to_player:.3f}")
            elif code == ProductCode.MAX4D:
                print("\n=== Max 4D: đã ngừng phát hành (kỳ cuối 31/08/2021) — chỉ có lịch sử kết quả (`products analyze --product max4d`), không có bảng giá hiện hành.")
            else:
                from vietlott_engine.game_theory.max3d import get_product as m3d, product_summary

                for name in (["max3d", "max3dplus"] if code == ProductCode.MAX3D else ["max3dpro"]):
                    _print(product_summary(m3d(name)))
        return 0
    return 2


def cmd_market(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.market import build_all

    s = get_settings()
    out = Path(a.out) if a.out else s.calibration_dir
    reports = build_all(_repo(), out, s.anchors_file)
    for game, rep in reports.items():
        cal = rep.calibration
        print(f"\n=== {game}: {rep.draws_with_prizes}/{rep.draws} draws with prize data ===")
        if rep.payout_share_used is not None:
            print(f"  payout share s = {rep.payout_share_used:.3f} ({rep.payout_share_method})")
        if rep.tickets_sold:
            t = rep.tickets_sold
            print(f"  tickets sold: median {t.median:,.0f} (p10 {t.p10:,.0f} – p90 {t.p90:,.0f}) — {t.method}")
        print(f"  crowd: quick-pick share {cal.quick_pick_share:.2f} ± {cal.quick_pick_share_se:.2f}; most popular {cal.most_popular_numbers[:6]}, least {cal.least_popular_numbers[:6]}")
        lc = rep.level_check
        print(f"  level check (winners vs predicted popularity): corr {lc.correlation:.2f}, slope {lc.slope:.2f}, permutation p {lc.permutation_p_value:.4f}")
        if rep.sales_model:
            sm = rep.sales_model
            print(f"  sales elasticity to jackpot: {sm.elasticity:.2f} ± {sm.elasticity_se:.2f} (curvature {sm.curvature:+.3f} ± {sm.curvature_se:.3f})")
        for note in rep.notes:
            print("  •", note)
    print(f"\nwritten to {out}")
    return 0


def cmd_odds(a: argparse.Namespace) -> int:
    from vietlott_engine.api.routers.products import odds

    for spec in _games(a.game):
        t = odds(spec)
        print(f"\n=== {spec.display_name}: {t.combinations:,} combinations, {spec.ticket_price:,} đ/play ===")
        for r in t.tiers:
            print(f"  {r.tier:<12} {r.rule:<34} 1 / {r.one_in:>14,.0f}   prize {r.prize:>16,.0f}")
        print(f"  any prize: {t.p_any_prize:.4%} (1 / {t.one_in_any_prize:.1f})")
    return 0


def cmd_bao(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.bao import analyse_bao, bao_catalog, bao_ev, bao_prize_table, compare_bao_strategies, fmt_vnd

    spec = get_game(a.game)
    if a.catalog:
        rows = bao_catalog(spec)
        if a.json:
            _print([r.model_dump() for r in rows])
            return 0
        print(f"{spec.display_name}: các kiểu bao Vietlott bán (10.000 đ/lượt)")
        for r in rows:
            print(f"  {r.kind:<22} {r.main_numbers:>2} số chính{f' + {r.special_numbers} ĐB' if spec.separate_special else '':<8} {r.plays:>6} lượt  {fmt_vnd(r.cost):>13} đ")
        return 0
    if a.table is not None:
        rows = bao_prize_table(spec, a.table, len(a.specials or [1]))
        if a.json:
            _print(rows)
            return 0
        print(f"{spec.display_name} — bảng giải thưởng Bao {a.table} (j = số trúng nằm trong các số đã chọn)")
        for r in rows:
            print(f"  j={r['hits_in_set']} {r['bonus']:<13} p={r['probability']:.3e}  {r['vietlott_style']:<44} {r['plays_per_tier']}")
        return 0
    if not a.numbers:
        print("--numbers is required (or use --catalog / --table)", file=sys.stderr)
        return 2
    jp = {k: v for k, v in (("jackpot1", a.jackpot1), ("jackpot2", a.jackpot2)) if v is not None}
    if a.compare:
        _print(compare_bao_strategies(spec, a.numbers, a.specials, sims=a.sims, seed=a.seed))
        return 0
    res = analyse_bao(spec, a.numbers, a.specials, jp, after_tax=a.after_tax, tax_basis=a.tax_basis, strict=a.strict)
    if a.json:
        _print(res)
        return 0
    print(f"{res.kind} {res.numbers}{' ĐB ' + str(res.specials) if res.specials else ''}: {res.plays} lượt, {fmt_vnd(res.cost)} đ{'' if res.documented else '  [không có trong danh mục Vietlott]'}")
    for note in res.notes:
        print(f"  ! {note}")
    print(f"  P(trúng ≥ 1 giải) = {res.p_any_prize:.4%}  (cùng tiền mua {res.plays} vé lẻ độc lập: {res.p_any_prize_same_budget_single_tickets:.4%})")
    print(f"  P(tiền thưởng ≥ tiền vé) = {res.p_profit:.4%}")
    print(f"  tiền thưởng kỳ vọng {fmt_vnd(res.expected_payout)} đ trước thuế, {fmt_vnd(res.expected_payout_net)} đ sau thuế (tính thuế theo {res.tax_basis}); RTP {res.return_to_player:.3f}")
    for row in res.prize_table:
        print(f"  j={row['hits_in_set']} {row['bonus']:<13} p={row['probability']:.3e}  {row['vietlott_style']:<44} {row['plays_per_tier']}")
    if a.tickets_sold:
        ev = bao_ev(spec, res.numbers, res.specials, jp or {t.name: spec.min_jackpots[t.name] for t in spec.tiers if t.is_jackpot}, a.tickets_sold, tax_basis=a.tax_basis)
        print(f"  có người trúng chung (N = {a.tickets_sold:,} vé): kỳ vọng sau thuế {fmt_vnd(ev.expected_payout_net)} đ, RTP {ev.return_to_player:.3f}")
    return 0


def cmd_max3d(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.max3d import analyse_bao as max3d_bao
    from vietlott_engine.game_theory.max3d import get_product, product_summary

    prod = get_product(a.product)
    if a.summary or not a.numbers:
        _print(product_summary(prod))
        return 0
    _print(max3d_bao(prod, a.kind, a.numbers, sims=a.sims, seed=a.seed, stake_multiple=a.stake))
    return 0


def cmd_coverage(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.coverage import optimise_coverage

    spec = get_game(a.game)
    res = optimise_coverage(spec, a.budget, a.min_tier, a.sim_draws, a.eval_draws, seed=a.seed)
    if a.json:
        _print(res)
        return 0
    lo, hi = res.p_at_least_one_ci95
    print(f"{spec.display_name}: {a.budget} tickets ({res.cost:,} đ), prize ≥ {res.min_tier}")
    print(f"  P(≥1 prize) optimised {res.p_at_least_one:.2%} [{lo:.2%}, {hi:.2%}]  vs random {res.p_random_tickets:.2%}  (bound {res.upper_bound:.2%})")
    for i, t in enumerate(res.tickets):
        sp = f" + {res.specials[i]}" if res.specials else ""
        print(f"  {i + 1:>3}: {t}{sp}")
    return 0


def cmd_rolldown(a: argparse.Namespace) -> int:
    from vietlott_engine.game_theory.rolldown import rolldown_ev

    spec = get_game("lotto535")
    if a.history:
        path = get_settings().calibration_dir / "market_lotto535.json"
        if not path.exists():
            print("no market model; run `vietlott market` first", file=sys.stderr)
            return 1
        data = json.loads(path.read_text(encoding="utf-8"))
        off, fc = data.get("official_rolldowns"), data.get("rolldown_forecast")
        if off:
            if a.json:
                _print({"official": off, "forecast": fc, "forecast_backtest": data.get("rolldown_forecast_backtest")})
                return 0
            acc = off["accrual"]
            print(f"official jackpot series: {off['executed']} rolldowns executed, {off['pre_empted']} pre-empted (jackpot won first)")
            print(f"  accrual: slow {acc['slow_rate']:.3f} / fast {acc['fast_rate']:.3f} per đồng; switch at ≈ {acc['switch_level_median'] / 1e9:.1f} tỷ; ≈ {acc['diverted_per_restart_median'] / 1e9:.2f} tỷ refunded per restart")
            for e in off["events"]:
                if e["executed"]:
                    print(f"  #{e['draw_id']:<5} {e['date']} chia giải pot {e['jackpot_estimate'] / 1e9:5.1f} tỷ  sold≈{e['tickets_sold'] / 1e6:4.2f}M (x{e['sales_multiple']:4.1f})  RTP {e['realised_rtp_pre_tax']:.2f} pre-tax, {e['model_rtp_after_tax']:.2f} after tax")
                else:
                    print(f"  #{e['draw_id']:<5} {e['date']} announced, jackpot won first ({e['jackpot_estimate'] / 1e9:.1f} tỷ)")
            print(f"  trend: {off['rtp_trend']}")
            if fc:
                print(f"next: after #{fc['last_draw_id']} ({fc['last_date']}) pot {fc['pot'] / 1e9:.2f} tỷ, {fc['phase']} phase, ≈ {fc['draws_to_threshold']:.1f} draws to 12 tỷ → {fc['expected_rolldown_window']}")
            return 0
        rh = data.get("rolldowns")
        if a.json or not rh:
            _print(rh)
            return 0
        print(f"announced {rh['announced']}, executed {rh['executed']}; accrual c = {rh['accrual_rate']:.3f}, rolldown day {rh['rolldown_day_rate']:.2f}")
        for e in rh["events"]:
            tag = "chia giải" if e["executed"] else "JP won  "
            print(f"  #{e['draw_id']:<5} {e['date']} {tag} pot {e['jackpot_estimate'] / 1e9:5.1f} tỷ  sold {e['tickets_sold'] / 1e6:4.2f}M (x{e['sales_multiple']:4.1f})  RTP {e['realised_rtp_pre_tax']:.2f} [{e['realised_rtp_conservative']:.2f}]")
        print(f"  trend: {rh['rtp_trend']}")
        return 0
    if a.jackpot is None or a.tickets_sold is None:
        print("--jackpot and --tickets-sold are required (or use --history)", file=sys.stderr)
        return 2
    _print(rolldown_ev(spec, a.jackpot, a.tickets_sold))
    return 0


def _load_bundled(repo, seed_dir: Path) -> list[dict]:  # type: ignore[no-untyped-def]
    """Bundled JSONL snapshot (draws + prize tables) → the draw store. Idempotent."""
    from vietlott_engine.crawler.pipeline import SyncPipeline
    from vietlott_engine.crawler.prize_sources import records_from_jsonl
    from vietlott_engine.crawler.sources.mirror import JsonlFileSource

    async def run() -> list[dict]:
        out = []
        for spec in GAMES.values():
            rep = (await SyncPipeline(JsonlFileSource(seed_dir), repo).run(spec)).to_dict()
            prize_file = seed_dir / f"prizes_{spec.code.value}.jsonl"
            if prize_file.exists():
                rep["prize_records"] = repo.upsert_prizes(records_from_jsonl(prize_file.read_text(encoding="utf-8")))
            out.append(rep)
        return out

    return asyncio.run(run())


def cmd_init(a: argparse.Namespace) -> int:
    """First run after installing: load the bundled data, optionally learn every forecaster, check."""
    from vietlott_engine.core.products import ProductCode
    from vietlott_engine.forecast.data import load_series
    from vietlott_engine.forecast.engine import refresh, vn, vn_int

    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    print(f"1/3 Nạp dữ liệu đi kèm từ {s.product_seed_dir} vào kho {s.duckdb_path} …")
    repo = _repo()
    for r in _load_bundled(repo, Path(s.product_seed_dir)):
        print(f"    {r.get('game', '?'):<9} {vn_int(r.get('total_after', 0)):>6} kỳ, {vn_int(r.get('prize_records', 0)):>5} bảng giải")
    if a.skip_forecast:
        print("2/3 Bỏ qua bước học dự báo (--skip-forecast). Chạy sau bằng `vietlott forecast fit`.")
    else:
        print("2/3 Học bộ dự báo cho 8 sản phẩm (Keno ~300 nghìn kỳ, khoảng 1 phút) …")
        store = _product_store()
        directory = s.forecast_dir or s.data_dir / "forecast"
        for code in ProductCode:
            f, learned, _ = refresh(code, load_series(code, repository=repo, store=store, seed_dir=s.product_seed_dir), directory, refit=True)
            ev = f.forecast().evidence
            print(f"    {code.value:<9} {vn_int(learned):>7} kỳ · e-value cao nhất 10^{vn(ev.max_log10_wealth)} · {'có bằng chứng' if ev.found else 'không có bằng chứng'}")
    print("3/3 Kiểm tra cài đặt …")
    return cmd_doctor(argparse.Namespace(offline=a.offline, json=False))


def _check_url(url: str, timeout: float) -> tuple[bool, str]:
    import httpx

    from vietlott_engine.crawler.http import is_cloudflare_challenge

    try:
        method = "HEAD" if "githubusercontent" in url else "GET"  # the data files are large: only ask whether they are there
        r = httpx.request(method, url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "vietlott-quant-engine doctor"})
    except httpx.ProxyError as exc:
        return False, f"proxy từ chối ({exc.__class__.__name__})"
    except httpx.HTTPError as exc:
        return False, f"không kết nối được ({exc.__class__.__name__})"
    if is_cloudflare_challenge(r):
        return False, f"Cloudflare yêu cầu xác minh người dùng (HTTP {r.status_code}) — xem README mục 1.8"
    if r.status_code == 403:
        return False, "HTTP 403 — vietlott.vn chỉ phục vụ IP Việt Nam, hoặc mạng/proxy chặn"
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}"
    return True, f"HTTP {r.status_code}"


def cmd_doctor(a: argparse.Namespace) -> int:
    """Check Python, packages, bundled data, storage, forecaster state and (unless --offline) the sources."""
    import importlib
    import platform

    from vietlott_engine import __version__
    from vietlott_engine.core.products import SEED_FILES, ProductCode

    s = get_settings()
    rows: list[tuple[str, bool, str, bool]] = []  # (item, ok, detail, required)

    def add(item: str, ok: bool, detail: str, required: bool = True) -> None:
        rows.append((item, ok, detail, required))

    add("Python ≥ 3.11", sys.version_info >= (3, 11), f"{platform.python_version()} ({platform.system()} {platform.machine()})")
    add("vietlott-quant-engine", True, f"{__version__} tại {Path.cwd()}")
    for mod, name in (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("pydantic", "pydantic"), ("pydantic_settings", "pydantic-settings"),
                      ("httpx", "httpx"), ("bs4", "beautifulsoup4"), ("duckdb", "duckdb"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn")):
        try:
            m = importlib.import_module(mod)
            add(f"gói {name}", True, str(getattr(m, "__version__", "đã cài")), required=mod != "duckdb")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            add(f"gói {name}", False, f"chưa cài được: {exc.__class__.__name__}", required=mod != "duckdb")
    seed = Path(s.product_seed_dir)
    needed = ["power645.jsonl", "power655.jsonl", "power535.jsonl", *SEED_FILES.values(), "exclusions.json"]
    missing = [n for n in needed if not (seed / n).exists()]
    add("dữ liệu đi kèm", not missing, f"{seed}: đủ {len(needed)} tệp" if not missing else f"thiếu {', '.join(missing)} trong {seed}")
    try:
        repo = _repo()
        counts = {spec.code.value: repo.count(spec.code) for spec in GAMES.values()}
        empty = [g for g, c in counts.items() if c == 0]
        add("kho kết quả (DuckDB)", not empty, ", ".join(f"{g} {c}" for g, c in counts.items()) + ("" if not empty else " — chạy `vietlott init`"), required=False)
    except Exception as exc:  # noqa: BLE001
        add("kho kết quả (DuckDB)", False, f"{exc.__class__.__name__}: {exc}", required=False)
    fdir = s.forecast_dir or s.data_dir / "forecast"
    states = [c.value for c in ProductCode if (Path(fdir) / f"{c.value}.json").exists()]
    add("bộ dự báo đã học", len(states) == len(ProductCode), f"{len(states)}/{len(ProductCode)} sản phẩm" + ("" if len(states) == len(ProductCode) else " — chạy `vietlott forecast fit`"), required=False)
    try:
        Path(s.data_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(s.data_dir) / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("ghi được vào thư mục dữ liệu", True, str(Path(s.data_dir).resolve()))
    except OSError as exc:
        add("ghi được vào thư mục dữ liệu", False, str(exc))
    if not a.offline:
        for name, url in (("kho cộng đồng (GitHub)", s.nhanaz_base_url.rstrip("/") + "/exclusions.csv"), ("bản sao GitHub", s.canonical_base_url.rstrip("/") + "/mega645.jsonl"), ("vietlott.vn", s.vietlott_base_url)):
            ok, detail = _check_url(url, timeout=10)
            add(f"mạng: {name}", ok, detail, required=False)
    if getattr(a, "json", False):
        _print([{"item": i, "ok": ok, "detail": d, "required": r} for i, ok, d, r in rows])
    else:
        for item, ok, detail, required in rows:
            mark = "✓" if ok else ("✗" if required else "!")
            print(f"  {mark} {item:<30} {detail}")
        bad = [i for i, ok, _, r in rows if not ok and r]
        print("  → Cài đặt dùng được." if not bad else f"  → Cần sửa: {', '.join(bad)}.")
        if any(not ok and not r and i.startswith("mạng: vietlott") for i, ok, _, r in rows):
            print("    (Không vào được vietlott.vn là bình thường ở ngoài Việt Nam; `--source auto` sẽ dùng nguồn dự phòng.)")
    return 0 if all(ok or not r for _, ok, _, r in rows) else 1


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("vietlott_engine.api.main:app", host=a.host, port=a.port, workers=a.workers, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from vietlott_engine import __version__

    p = argparse.ArgumentParser(prog="vietlott", description="Vietlott Quant Engine — dữ liệu, xác suất, kiểm định và dự báo tự học có kiểm chứng")
    p.add_argument("--version", action="version", version=f"vietlott-quant-engine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="lần chạy đầu sau khi cài: nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra")
    s.add_argument("--skip-forecast", action="store_true", help="không học bộ dự báo lúc này")
    s.add_argument("--offline", action="store_true", help="không kiểm tra mạng")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("doctor", help="kiểm tra Python, gói, dữ liệu, kho, bộ dự báo và mạng")
    s.add_argument("--offline", action="store_true", help="không kiểm tra mạng")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    game_choices = ["all", "mega645", "power655", "lotto535"]

    s = sub.add_parser("sync", help="download draws into DuckDB/Parquet")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--source", choices=["github_mirror", "vietlott", "file", "nhanaz", "v130", "auto"])
    s.add_argument("--path", help="directory or .jsonl file for --source file; archive / 1.3.0 folder for nhanaz / v130")
    s.add_argument("--full", action="store_true", help="re-download everything")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("analyze", help="randomness, gaps, co-occurrence, Bayesian, Markov")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--sims", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_analyze)

    for name, fn in (("ev", cmd_ev), ("optimize", cmd_optimize)):
        s = sub.add_parser(name, help="expected value of a ticket" if name == "ev" else "anti-popularity ticket optimiser")
        s.add_argument("--game", required=True, choices=game_choices[1:])
        s.add_argument("--jackpot1", type=float)
        s.add_argument("--jackpot2", type=float)
        s.add_argument("--tickets-sold", type=int)
        s.add_argument("--quick-pick", type=float, help="override the calibrated quick-pick share")
        s.add_argument("--special", type=int, help="Lotto 5/35 special number 1..12")
        if name == "ev":
            s.add_argument("--ticket", type=int, nargs="+", required=True)
        else:
            s.add_argument("--n", type=int, default=5)
            s.add_argument("--max-overlap", type=int, default=2)
            s.add_argument("--exclude", type=int, nargs="*")
            s.add_argument("--iterations", type=int, default=20_000)
            s.add_argument("--seed", type=int)
        s.set_defaults(func=fn)

    s = sub.add_parser("wheel", help="covering-design wheel")
    s.add_argument("--game", required=True, choices=game_choices[1:])
    s.add_argument("--pool", type=int, nargs="+", required=True)
    s.add_argument("--t", type=int, default=3, help="guaranteed matches on one ticket")
    s.add_argument("--m", type=int, default=4, help="...if at least m drawn numbers are in the pool")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_wheel)

    s = sub.add_parser("backtest", help="walk-forward backtest")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--strategies", nargs="*", choices=STRATEGY_NAMES)
    s.add_argument("--start", type=int, default=300)
    s.add_argument("--end", type=int)
    s.add_argument("--tickets", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-ml", action="store_true")
    s.add_argument("--out", help="directory for .md/.json reports")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("ml", help="walk-forward skill of GCN vs logistic baseline")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--first-test", type=int, default=900)
    s.add_argument("--refit-every", type=int, default=150)
    s.add_argument("--epochs", type=int, default=40)
    s.set_defaults(func=cmd_ml)

    s = sub.add_parser("inference", help="fairness certificate: FWER per number, power/TOST, hierarchical Bayes, e-processes, change points")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--sims", type=int, default=1000)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="directory for JSON reports")
    s.set_defaults(func=cmd_inference)

    s = sub.add_parser("decide", help="payout distribution, Kelly stake and EV uncertainty for one ticket")
    s.add_argument("--game", required=True, choices=game_choices[1:])
    s.add_argument("--ticket", type=int, nargs="+", required=True)
    s.add_argument("--special", type=int)
    s.add_argument("--jackpot1", type=float)
    s.add_argument("--jackpot2", type=float)
    s.add_argument("--tickets-sold", type=int)
    s.add_argument("--bankroll", type=float, default=1e9)
    s.add_argument("--sold-range", type=int, nargs=2, metavar=("LOW", "HIGH"), help="run the prior-uncertainty analysis")
    s.add_argument("--sims", type=int, default=300)
    s.set_defaults(func=cmd_decide)

    s = sub.add_parser("prizes", help="download winner counts / jackpot pots and reconcile with stored draws")
    s.add_argument("--game", default="all", choices=game_choices)
    s.add_argument("--source", default="auto", choices=["auto", "canonical", "vietlott", "nhanaz", "v130", "compal"], help="auto = first that answers of vietlott → canonical → nhanaz → v130; canonical = official detail-page records (GitHub); vietlott = vietlott.vn (Vietnam only); nhanaz = community archive; v130 = Vietlott Quant Engine 1.3.0 snapshot; compal = older third-party tables")
    s.add_argument("--last", type=int, default=20, help="--source vietlott: number of most recent stored draws to fetch")
    s.add_argument("--path", help="--source nhanaz/v130: local archive folder or 1.3.0 package folder")
    s.set_defaults(func=cmd_prizes)

    s = sub.add_parser("products", help="Keno, Bingo18, Max 3D, Max 3D Pro: list / sync / analyze / odds")
    s.add_argument("action", choices=["list", "sync", "analyze", "odds", "import-pages"], help="import-pages: pages you saved from vietlott.vn after passing its human check (--path file/folder: .html, .mhtml, .har)")
    s.add_argument("--product", default="all", help="all | keno | bingo18 | max3d | max3dpro | max4d")
    s.add_argument("--source", default="auto", choices=["auto", "vietlott", "nhanaz", "mirror", "canonical", "v130"], help="sync: auto (vietlott → nhanaz → mirror/canonical → v130), vietlott (from Vietnam), nhanaz (community archive), mirror, canonical (Max 3D), v130 (1.3.0 snapshot)")
    s.add_argument("--path", help="sync --source nhanaz/v130: local archive folder or 1.3.0 package folder")
    s.add_argument("--include-unconfirmed", action="store_true", help="analyze: keep draws Vietlott announced as not confirmed")
    s.add_argument("--dry-run", action="store_true", help="import-pages: compare with the data only, write nothing")
    s.add_argument("--replace-mismatches", action="store_true", help="import-pages: also overwrite draws whose saved page disagrees with the data (default: hold them back)")
    s.add_argument("--max-pages", type=int, help="sync --source vietlott: results-list pages per product")
    s.add_argument("--full", action="store_true", help="sync --source vietlott: ignore the stored last draw id")
    s.add_argument("--with-matrix", action="store_true", help="list: also count Mega/Power/Lotto draws in the store")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_products)

    s = sub.add_parser("forecast", help="self-learning next-draw forecaster for all products (fit / update / next / scoreboard)")
    s.add_argument("action", choices=["next", "update", "fit", "scoreboard"], nargs="?", default="next")
    s.add_argument("--product", default="all", help="all | mega645 | power655 | lotto535 | keno | bingo18 | max3d | max3dpro | max4d")
    s.add_argument("--last", type=int, default=None, help="fit: learn only from the last N draws")
    s.add_argument("--dir", default=None, help="state + ledger folder (default <data_dir>/forecast)")
    s.add_argument("--no-record", action="store_true", help="next: do not write the forecast to the ledger")
    s.add_argument("--force-record", action="store_true", help="next: record even when the draw may have started (kept out of the scoreboard)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("market", help="payout share, tickets sold, crowd calibration, sales model, rolldowns")
    s.add_argument("--out", help="output directory (default: VQE_CALIBRATION_DIR)")
    s.set_defaults(func=cmd_market)

    s = sub.add_parser("odds", help="exact probability of every prize tier")
    s.add_argument("--game", default="all", choices=game_choices)
    s.set_defaults(func=cmd_odds)

    s = sub.add_parser("bao", help="chơi bao: catalogue, Vietlott-style prize table, exact distribution, comparison")
    s.add_argument("--game", required=True, choices=game_choices[1:])
    s.add_argument("--numbers", type=int, nargs="+", help="k−1 numbers (Bao 5/4) or ≥ k numbers (Bao v)")
    s.add_argument("--specials", type=int, nargs="*", help="Lotto 5/35 special numbers (2–12 = bao số đặc biệt)")
    s.add_argument("--catalog", action="store_true", help="list every bao option Vietlott sells")
    s.add_argument("--table", type=int, metavar="V", help="prize lookup table of Bao V")
    s.add_argument("--compare", action="store_true", help="bao vs bao rút gọn vs spread / random singles (same money)")
    s.add_argument("--jackpot1", type=float)
    s.add_argument("--jackpot2", type=float)
    s.add_argument("--tickets-sold", type=int, help="add the EV with other players sharing the jackpot")
    s.add_argument("--after-tax", action="store_true")
    s.add_argument("--tax-basis", choices=["ticket", "play"], default="ticket")
    s.add_argument("--strict", action="store_true", help="reject options Vietlott does not sell")
    s.add_argument("--sims", type=int, default=100_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_bao)

    s = sub.add_parser("max3d", help="Max 3D / 3D+ / 3D Pro: exact odds and bao plays")
    s.add_argument("--product", default="max3dpro", help="max3d | max3dplus | max3dpro")
    s.add_argument("--kind", default="co_ban", choices=["co_ban", "bao_bo_so", "bao_nhieu_bo_so", "dao_so", "bao_vi_tri"])
    s.add_argument("--numbers", nargs="+", help="three-digit numbers (bao vị trí: patterns like 1*3)")
    s.add_argument("--stake", type=int, default=1, help="stake multiple (1 = 10.000 đ per play)")
    s.add_argument("--sims", type=int, default=200_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--summary", action="store_true")
    s.set_defaults(func=cmd_max3d)

    s = sub.add_parser("coverage", help="tickets maximising P(at least one prize) for a budget")
    s.add_argument("--game", required=True, choices=game_choices[1:])
    s.add_argument("--budget", type=int, default=10)
    s.add_argument("--min-tier", help="count only prizes at or above this tier")
    s.add_argument("--sim-draws", type=int, default=30_000)
    s.add_argument("--eval-draws", type=int, default=200_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_coverage)

    s = sub.add_parser("rolldown", help="Lotto 5/35 rolldown (chia giải) EV or history")
    s.add_argument("--jackpot", type=float)
    s.add_argument("--tickets-sold", type=int)
    s.add_argument("--history", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_rolldown)

    s = sub.add_parser("serve", help="run the FastAPI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(func=cmd_serve)
    return p


def _utf8_console() -> None:
    """Vietnamese output on any console: Windows code pages and pipes otherwise raise UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    from vietlott_engine.paths import enter_project

    _utf8_console()
    enter_project()
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - one readable line for the user; VQE_DEBUG=1 shows the traceback
        if os.environ.get("VQE_DEBUG"):
            raise
        hint = ""
        if "lock" in str(exc).lower() or exc.__class__.__name__ == "IOException":
            hint = " — kho DuckDB có thể đang được tiến trình khác dùng (vd. `vietlott serve`)"
        elif isinstance(exc, FileNotFoundError):
            hint = " — chạy lệnh trong thư mục dự án, hoặc đặt VQE_HOME tới thư mục đó"
        print(f"Lỗi ({exc.__class__.__name__}): {exc}{hint}", file=sys.stderr)
        print("Chạy `vietlott doctor` để kiểm tra cài đặt và mạng; đặt VQE_DEBUG=1 để xem chi tiết.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
