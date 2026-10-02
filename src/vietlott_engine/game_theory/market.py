"""End-to-end market model per game: payout share → tickets sold → crowd calibration →
validation → jackpot path → sales response. Produces the artefacts the EV engine uses.

Power 6/55  : jackpot pots are published every draw → tickets sold from the accounting
              identity; crowd calibrated with known sales (strongest identification).
Lotto 5/35  : winners for every draw since launch, 7 tiers incl. the special number →
              calibrated with per-draw sales scales.
Mega 6/45   : prize data only since 09/2025 and no jackpot series → the Power crowd is
              transferred (same players) and validated on Mega's winner counts; sales
              N̂_t estimated from winners; the jackpot path is rebuilt from N̂_t and closed
              on a published jackpot value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vietlott_engine.core.games import GameCode, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.prizes import PrizeHistory
from vietlott_engine.game_theory.calibration import (
    BehaviourCalibration,
    LevelCheck,
    calibrate,
    fitted_tickets_sold,
    level_check,
    transfer_calibration,
)
from vietlott_engine.game_theory.rolldown import OfficialRolldowns, RolldownForecast, RolldownHistory, backtest_rolldown_forecast, forecast_next_rolldown, rolldown_history, rolldown_history_official
from vietlott_engine.game_theory.sales import (
    JackpotPath,
    PayoutShare,
    SalesModel,
    advertised_jackpot,
    estimate_payout_share,
    fit_sales_model,
    jackpot_hit_check,
    reconstruct_jackpot,
    tickets_sold_from_accounting,
)

log = get_logger(__name__)


class Anchor(BaseModel):
    game: str
    draw_id: int
    value: float
    kind: str = "exact"  # exact | lower_bound | approx
    source: str = ""


class SoldSummary(BaseModel):
    draws: int
    median: float
    p10: float
    p90: float
    latest: float | None
    method: str


class MarketReport(BaseModel):
    game: str
    draws: int
    draws_with_prizes: int
    payout_share: PayoutShare | None = None
    payout_share_used: float | None = None
    payout_share_method: str | None = None
    tickets_sold: SoldSummary | None = None
    sales_model: SalesModel | None = None
    calibration: BehaviourCalibration
    level_check: LevelCheck
    jackpot_hits: dict | None = None
    jackpot_path: JackpotPath | None = None
    anchor_checks: list[dict] = []
    rolldowns: RolldownHistory | None = None  # reconstruction from sales surges (no jackpot series needed)
    official_rolldowns: OfficialRolldowns | None = None  # from the official jackpot series
    rolldown_forecast: RolldownForecast | None = None
    rolldown_forecast_backtest: dict | None = None
    transfer_check: dict | None = None
    notes: list[str] = []


def _sold_summary(sold: np.ndarray, method: str) -> SoldSummary:
    ok = np.isfinite(sold)
    return SoldSummary(
        draws=int(ok.sum()),
        median=float(np.median(sold[ok])),
        p10=float(np.quantile(sold[ok], 0.1)),
        p90=float(np.quantile(sold[ok], 0.9)),
        latest=float(sold[ok][-1]) if ok.any() else None,
        method=method,
    )


def build_power(h: DrawHistory, ph: PrizeHistory) -> tuple[MarketReport, np.ndarray]:
    share = estimate_payout_share(ph, h.dates)
    sold = tickets_sold_from_accounting(ph, share.overall)
    cal = calibrate(h, ph, sold)
    pop = cal.popularity_model(None, pattern_priors=False)
    hits = jackpot_hit_check(pop.ticket_probability, sold, ph.jackpot_won()[:, 0], h.spec, h.numbers.astype(int))
    sm = fit_sales_model(h, sold, advertised_jackpot(ph))
    rep = MarketReport(
        game=h.spec.code.value,
        draws=len(h),
        draws_with_prizes=int(ph.has_winners.sum()),
        payout_share=share,
        payout_share_used=share.overall,
        payout_share_method="method of moments on tiers 2–3 (identity E[y_m] = N·H_m)",
        tickets_sold=_sold_summary(sold, "jackpot accounting identity ΔJ + F = s·R"),
        sales_model=sm,
        calibration=cal,
        level_check=level_check(cal, h, ph),
        jackpot_hits=hits,
        notes=[
            f"Tier estimates of s agree ({', '.join(f'{k}: {v:.3f}' for k, v in share.by_tier.items())}), validating the identity.",
            f"Sales elasticity to the advertised jackpot: {sm.elasticity:.2f} ± {sm.elasticity_se:.2f} at the median jackpot, "
            f"curvature {sm.curvature:+.3f} ± {sm.curvature_se:.3f} (HAC).",
        ],
    )
    return rep, sold


def build_lotto(h: DrawHistory, ph: PrizeHistory, anchors: list[Anchor] | None = None) -> tuple[MarketReport, np.ndarray]:
    cal = calibrate(h, ph, None)
    sold = fitted_tickets_sold(cal, h, ph)
    lotto_anchors = [(a.draw_id, a.value) for a in (anchors or []) if a.game == h.spec.code.value]
    rh = rolldown_history(h.spec, h, ph, sold, lotto_anchors)
    official = forecast = backtest = None
    if ph.has_pots.mean() > 0.9:
        official = rolldown_history_official(h.spec, h, ph, sold, reconstruction=rh)
        forecast = forecast_next_rolldown(h.spec, h, ph, sold, official)
        backtest = backtest_rolldown_forecast(h.spec, h, ph, sold, official)
    rep = MarketReport(
        game=h.spec.code.value,
        draws=len(h),
        draws_with_prizes=int(ph.has_winners.sum()),
        tickets_sold=_sold_summary(sold, "NB profile from winner counts (E[y_g] = N·P_g(W_t))"),
        calibration=cal,
        level_check=level_check(cal, h, ph),
        rolldowns=rh,
        official_rolldowns=official,
        rolldown_forecast=forecast,
        rolldown_forecast_backtest=backtest,
        notes=[
            "Sales are estimated from winner counts (no sales figures are published per draw)."
            + (
                f" Official jackpot series: {official.executed} rolldowns executed, {official.pre_empted} pre-empted; accrual "
                f"{official.accrual.slow_rate:.3f} (slow) / {official.accrual.fast_rate:.3f} (fast) per đồng; slow phase refunds "
                f"≈{official.accrual.diverted_per_restart_median / 1e9:.2f} tỷ per jackpot restart."
                if official
                else " No jackpot series available: rolldowns reconstructed from sales surges."
            ),
            ("Reconstruction without the jackpot series (v3 method, kept as a check): " if official else "")
            + f"{rh.announced} rolldown draws announced, {rh.executed} executed; jackpot accrual c = {rh.accrual_rate:.3f} "
            f"per đồng of sales (rolldown day {rh.rolldown_day_rate:.2f}); mean realised RTP of rolldown draws "
            f"{(rh.mean_realised_rtp or float('nan')):.2f} (pre-tax), falling as rolldown sales grow."
            + (f" Official series: mean realised RTP {official.mean_realised_rtp:.2f}." if official and official.mean_realised_rtp else ""),
        ],
    )
    return rep, sold


def build_mega(h: DrawHistory, ph: PrizeHistory, power_cal: BehaviourCalibration, anchors: list[Anchor]) -> tuple[MarketReport, np.ndarray]:
    if ph.has_pots.mean() > 0.8:
        # official jackpot series: same accounting route as Power; the Power crowd transfer
        # is kept as an out-of-sample check of both calibrations
        rep, sold = build_power(h, ph)
        tr = transfer_calibration(power_cal, h, ph)
        own, other = np.asarray(rep.calibration.beta), np.asarray(tr.beta)
        tr_lc = level_check(tr, h, ph)
        own_p = rep.calibration.fit.holdout_lr_p_value
        # selection rule (stated before looking at EV): keep Mega's own crowd fit unless it
        # fails its out-of-sample test AND the transferred Power crowd explains the level of
        # third-prize winners better
        use_transfer = own_p is not None and own_p > 0.05 and tr_lc.correlation > rep.level_check.correlation
        rep.transfer_check = {
            "beta_correlation_own_vs_power_transfer": float(np.corrcoef(own, other)[0, 1]),
            "quick_pick_own": rep.calibration.quick_pick_share,
            "quick_pick_power": tr.quick_pick_share,
            "own_holdout_p_value": own_p,
            "own_holdout_gain_per_draw": rep.calibration.fit.holdout_loglik_gain_per_draw,
            "level_check_own": rep.level_check.model_dump(),
            "level_check_power_transfer": tr_lc.model_dump(),
            "selected": "power_transfer" if use_transfer else "own",
        }
        if use_transfer:
            rep.notes.append(
                f"Crowd model: Mega's own fit does not beat a uniform crowd out of sample (holdout p = {own_p:.2f}); the Power 6/55 "
                f"crowd transferred to Mega explains third-prize winner levels better (corr {tr_lc.correlation:.2f} vs "
                f"{rep.level_check.correlation:.2f}), so the transferred model is used for popularity and EV."
            )
            rep.calibration, rep.level_check = tr, tr_lc
        return rep, sold
    cal = transfer_calibration(power_cal, h, ph)
    sold = fitted_tickets_sold(cal, h, ph)
    pop = cal.popularity_model(None, pattern_priors=False)
    hits = jackpot_hit_check(pop.ticket_probability, sold, ph.jackpot_won()[:, 0], h.spec, h.numbers.astype(int))
    rep = MarketReport(
        game=h.spec.code.value,
        draws=len(h),
        draws_with_prizes=int(ph.has_winners.sum()),
        tickets_sold=_sold_summary(sold, "NB profile from winner counts with the transferred crowd model"),
        calibration=cal,
        level_check=level_check(cal, h, ph),
        jackpot_hits=hits,
    )
    ids = [int(x) for x in h.draw_ids]
    exact = [a for a in anchors if a.game == h.spec.code.value and a.kind == "exact" and a.draw_id in ids]
    if exact:
        a = max(exact, key=lambda x: x.draw_id)
        idx = ids.index(a.draw_id)
        won = ph.jackpot_won()[:, 0]
        prev_wins = [i for i in range(idx) if won[i]]
        if prev_wins:
            seg = list(range(prev_wins[-1] + 1, idx + 1))
            fixed = ph.fixed_prizes_paid()
            jmin = float(h.spec.min_jackpots["jackpot1"])
            s = float((a.value - jmin + np.nansum(fixed[seg])) / np.nansum(sold[seg] * h.spec.ticket_price))
            rep.payout_share_used = s
            rep.payout_share_method = f"closure on published jackpot {a.value:,.0f} at draw {a.draw_id} (segment of {len(seg)} draws since the last win)"
            path = reconstruct_jackpot(h.spec, ph, sold, s, anchor_index=idx, anchor_value=a.value)
            rep.jackpot_path = path
            jvals = np.array([np.nan if v is None else v for v in path.jackpot])
            for other in anchors:
                if other.game == h.spec.code.value and other is not a and other.draw_id in ids:
                    v = jvals[ids.index(other.draw_id)]
                    rep.anchor_checks.append({"draw_id": other.draw_id, "published": other.value, "kind": other.kind, "reconstructed": None if np.isnan(v) else float(v), "source": other.source})
            before = np.full(len(h), np.nan)
            mins = h.spec.min_jackpots["jackpot1"]
            for t in range(1, len(h)):
                if won[t - 1]:
                    before[t] = mins
                elif np.isfinite(jvals[t - 1]):
                    before[t] = jvals[t - 1]
            ok = np.isfinite(before) & np.isfinite(sold)
            if ok.sum() >= 30:
                sm = fit_sales_model(h, sold, before)
                rep.sales_model = sm
                rep.notes.append(
                    f"Sales vs rebuilt jackpot: elasticity {sm.elasticity:.2f} ± {sm.elasticity_se:.2f} at the median jackpot, curvature "
                    f"{sm.curvature:+.3f} ± {sm.curvature_se:.3f} ('jackpot fever': sales accelerate at large jackpots); the jackpot "
                    "regressor is itself rebuilt from estimated sales, so treat the elasticity as indicative."
                )
            rep.notes.append(
                f"Effective jackpot share s = {s:.3f} from the closure. Published intermediate values (anchor_checks) "
                "test the shape of the rebuilt path; disagreement there means the sales estimates or s vary within the segment."
            )
    return rep, sold


def load_anchors(path: Path) -> list[Anchor]:
    if not path.exists():
        return []
    return [Anchor(**a) for a in json.loads(path.read_text(encoding="utf-8"))]


def build_all(repo, out_dir: Path, anchors_path: Path | None = None) -> dict[str, MarketReport]:
    """Run the three game pipelines and persist calibrations + reports + tickets sold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(anchors_path) if anchors_path else []
    reports: dict[str, MarketReport] = {}
    sold_all: dict[str, tuple[DrawHistory, np.ndarray]] = {}
    h, ph = repo.load_prize_history(GameCode.POWER_655)
    reports["power655"], sold = build_power(h, ph)
    sold_all["power655"] = (h, sold)
    h, ph = repo.load_prize_history(GameCode.LOTTO_535)
    if ph.has_winners.sum() >= 50:
        reports["lotto535"], sold = build_lotto(h, ph, anchors)
        sold_all["lotto535"] = (h, sold)
    h, ph = repo.load_prize_history(GameCode.MEGA_645)
    if ph.has_winners.sum() >= 30:
        reports["mega645"], sold = build_mega(h, ph, reports["power655"].calibration, anchors)
        sold_all["mega645"] = (h, sold)
    for game, rep in reports.items():
        rep.calibration.save(out_dir / f"behaviour_{game}.json")
        (out_dir / f"market_{game}.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
        hh, ss = sold_all[game]
        rows = [{"draw_id": int(i), "date": str(d), "tickets_sold": round(float(v))} for i, d, v in zip(hh.draw_ids, hh.dates, ss) if np.isfinite(v)]
        (out_dir / f"tickets_sold_{game}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    log.info("market models written to %s", out_dir)
    return reports


def load_sales_model(game: str | GameCode, directory: Path) -> SalesModel | None:
    path = Path(directory) / f"market_{get_game(game).code.value}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SalesModel(**data["sales_model"]) if data.get("sales_model") else None
