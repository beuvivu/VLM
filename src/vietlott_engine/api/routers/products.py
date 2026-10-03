"""Vietlott product features: exact odds, prize data, market calibration, bao tickets,
coverage portfolios and Lotto 5/35 rolldowns."""

from __future__ import annotations

import asyncio
from math import comb

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from vietlott_engine.api.deps import AppState, get_state, resolve_game
from vietlott_engine.api.schemas import BaoCompareRequest, BaoRequest, CoverageRequest, PrizeSyncRequest, RolldownRequest
from vietlott_engine.core.exceptions import DataValidationError, InsufficientDataError
from vietlott_engine.core.games import DEFAULT_TAX, GameSpec
from vietlott_engine.crawler.pipeline import PrizeSyncPipeline, build_http_client
from vietlott_engine.game_theory.bao import BaoAnalysis, BaoComparison, BaoEV, BaoOption, analyse_bao, bao_catalog, bao_ev, bao_prize_table, compare_bao_strategies
from vietlott_engine.game_theory.coverage import CoverageResult, optimise_coverage
from vietlott_engine.game_theory.rolldown import RolldownEV, RolldownHistory, rolldown_ev

router = APIRouter(prefix="/games/{game}")


class TierOdds(BaseModel):
    tier: str
    rule: str
    prize: float | None
    probability: float
    one_in: float
    ev_contribution_pre_tax: float


class OddsTable(BaseModel):
    game: str
    combinations: int
    ticket_price: int
    tiers: list[TierOdds]
    p_any_prize: float
    one_in_any_prize: float
    expected_prizes_per_100_tickets: float
    bao_plays: dict[str, int]
    note: str


def _rule(spec: GameSpec, t) -> str:  # type: ignore[no-untyped-def]
    r = t.main_range()
    main = f"{r.start}" if len(r) == 1 else f"{r.start}–{r.stop - 1}"
    extra = "số đặc biệt" if spec.separate_special else "số phụ"
    if t.bonus_required:
        return f"trùng {main} số + {extra}"
    shadowed = any(o is not t and o.bonus_required and set(o.main_range()) & set(r) for o in spec.tiers)
    return f"trùng {main} số, không trùng {extra}" if shadowed else f"trùng {main} số"


@router.get("/odds", response_model=OddsTable, tags=["vietlott"])
def odds(spec: GameSpec = Depends(resolve_game)) -> OddsTable:
    """Exact hypergeometric probability of every prize tier and of winning anything."""
    probs = spec.tier_probabilities
    tiers = []
    for t in spec.tiers:
        prize = float(t.fixed_amount) if t.fixed_amount is not None else float(spec.min_jackpots.get(t.name, 0))
        tiers.append(TierOdds(tier=t.name, rule=_rule(spec, t), prize=prize, probability=probs[t.name], one_in=1 / probs[t.name], ev_contribution_pre_tax=probs[t.name] * prize))
    k, n = spec.pick, spec.pool_size
    bao = {f"Bao {k - 1}": n - k + 1} | {f"Bao {v}": comb(v, k) for v in spec.bao_levels if v > k}
    return OddsTable(
        game=spec.code.value,
        combinations=spec.total_combinations,
        ticket_price=spec.ticket_price,
        tiers=tiers,
        p_any_prize=spec.p_any_prize,
        one_in_any_prize=1 / spec.p_any_prize,
        expected_prizes_per_100_tickets=100 * spec.p_any_prize,
        bao_plays=bao,
        note="Jackpot rows use the minimum jackpot. Every combination has exactly these probabilities.",
    )


@router.get("/prizes", tags=["vietlott"])
def prizes(spec: GameSpec = Depends(resolve_game), limit: int = Query(20, ge=1, le=5000), state: AppState = Depends(get_state)) -> dict:
    """Winners per tier (and jackpot pots when published) for the latest draws."""
    recs = state.repository.load_prizes(spec.code)
    if not recs:
        raise InsufficientDataError(f"no prize data for {spec.code.value}; call POST /games/{spec.code.value}/prizes/sync first")
    return {"game": spec.code.value, "total": len(recs), "records": [r.model_dump(mode="json") for r in recs[-limit:][::-1]]}


@router.post("/prizes/sync", tags=["vietlott"])
async def prizes_sync(body: PrizeSyncRequest | None = None, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> dict:
    """Download winner counts / jackpot pots and reconcile them with the stored draws.

    ``canonical`` (default) imports official detail-page records; ``vietlott`` reads the
    detail pages of the most recent stored draws directly (Vietnamese IP only)."""
    from vietlott_engine.crawler.product_store import sync_canonical_prizes

    body = body or PrizeSyncRequest()
    s = state.settings
    async with state.sync_lock(spec.code.value):
        async with build_http_client(s) as client:
            if body.source == "compal":
                pipe = PrizeSyncPipeline(client, state.repository, body.winners_base_url or s.prize_winners_base_url, body.power_history_url or s.prize_power_history_url)
                report = (await pipe.run(spec)).to_dict()
            else:
                report = (
                    await sync_canonical_prizes(
                        client, state.repository, spec, source=body.source, canonical_base_url=s.canonical_base_url,
                        vietlott_base_url=s.vietlott_base_url, last=body.last, bootstrap_cookie=s.vietlott_cookie_bootstrap,
                        nhanaz_base_url=s.nhanaz_base_url, nhanaz_dir=s.nhanaz_dir, v130_dir=s.v130_dir,
                    )
                ).to_dict()
    state.invalidate(spec)
    return report


@router.get("/market", tags=["vietlott"])
def market(spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> dict:
    """Payout share, tickets sold, sales model, crowd calibration and (Lotto) rolldowns."""
    rep = state.market(spec)
    if rep is None:
        raise InsufficientDataError("no market model; run `vietlott market` (writes data/calibration)")
    return rep.model_dump(mode="json", exclude={"jackpot_path"})


class BaoResponse(BaseModel):
    analysis: BaoAnalysis
    ev_with_co_winners: BaoEV | None = None


@router.get("/bao/catalog", response_model=list[BaoOption], tags=["bao"])
def bao_options(spec: GameSpec = Depends(resolve_game)) -> list[BaoOption]:
    """Every bao option Vietlott sells for the game: numbers, plays, cost."""
    return bao_catalog(spec)


@router.get("/bao/table", tags=["bao"])
def bao_table(spec: GameSpec = Depends(resolve_game), level: int = Query(..., ge=4, le=20, description="Bao level (number of main numbers)"), specials: int = Query(1, ge=1, le=12)) -> dict:
    """Vietlott-style prize lookup table of a bao level (what it wins when j drawn numbers are in the set)."""
    try:
        return {"game": spec.code.value, "level": level, "specials": specials, "rows": bao_prize_table(spec, level, specials)}
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc


@router.post("/bao", response_model=BaoResponse, response_model_exclude_none=True, tags=["bao"])
def bao(body: BaoRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> BaoResponse:
    """Exact payout distribution of a bao (system) ticket vs the same money in single tickets."""
    jp = {k: v for k, v in (("jackpot1", body.jackpot1), ("jackpot2", body.jackpot2)) if v is not None}
    try:
        res = analyse_bao(spec, body.numbers, body.specials, jp, after_tax=body.after_tax, tax_basis=body.tax_basis, strict=body.strict)
        ev = None
        if body.tickets_sold:
            cal = state.calibration(spec)
            pop = cal.popularity_model(None) if cal is not None else None
            full_jp = {t.name: jp.get(t.name, spec.min_jackpots[t.name]) for t in spec.tiers if t.is_jackpot}
            ev = bao_ev(spec, body.numbers, body.specials, full_jp, body.tickets_sold, pop, tax_basis=body.tax_basis)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    return BaoResponse(analysis=res if body.include_outcomes else res.model_copy(update={"outcomes": []}), ev_with_co_winners=ev)


@router.post("/bao/compare", response_model=BaoComparison, tags=["bao"])
async def bao_compare(body: BaoCompareRequest, spec: GameSpec = Depends(resolve_game)) -> BaoComparison:
    """Same money, different shapes: bao vs bao rút gọn vs spread-out singles vs quick picks."""
    try:
        return await asyncio.to_thread(compare_bao_strategies, spec, body.numbers, body.specials, body.sims, body.seed, (body.guarantee, body.condition))
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc


@router.post("/coverage", response_model=CoverageResult, tags=["vietlott"])
async def coverage(body: CoverageRequest, spec: GameSpec = Depends(resolve_game)) -> CoverageResult:
    """Choose B tickets that maximise P(at least one prize) in a draw (spread, not luck)."""
    try:
        return await asyncio.to_thread(
            optimise_coverage, spec, body.budget, body.min_tier, body.sim_draws, body.eval_draws, body.candidates, body.local_rounds, body.seed
        )
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc


def _require_rolldown(spec: GameSpec) -> None:
    if spec.rolldown is None:
        raise DataValidationError(f"{spec.display_name} has no rolldown (chia giải) rule; only Lotto 5/35 does")


@router.post("/rolldown/ev", response_model=RolldownEV, tags=["vietlott"])
def rolldown_expected_value(body: RolldownRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> RolldownEV:
    """EV of one ticket in a Lotto 5/35 rolldown draw, with the sales level at which it breaks even."""
    _require_rolldown(spec)
    sold = body.tickets_sold
    if sold is None:
        rep = state.market(spec)
        hist = (rep.official_rolldowns or rep.rolldowns) if rep else None
        recent = [e.tickets_sold for e in (hist.events if hist else []) if e.executed][-5:]
        sold = int(sum(recent) / len(recent)) if recent else 2_000_000
    return rolldown_ev(spec, body.jackpot, sold, DEFAULT_TAX)


@router.get("/rolldown/official", tags=["vietlott"])
def rolldown_official_endpoint(spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> dict:
    """Rolldowns read from the official jackpot series, the two-phase accrual, the
    next-rolldown forecast and its backtest (v3.2)."""
    _require_rolldown(spec)
    rep = state.market(spec)
    if rep is None or rep.official_rolldowns is None:
        raise InsufficientDataError("no official jackpot series; run `vietlott prizes --source canonical` then `vietlott market`")
    return {
        "official": rep.official_rolldowns.model_dump(mode="json"),
        "forecast": rep.rolldown_forecast.model_dump(mode="json") if rep.rolldown_forecast else None,
        "forecast_backtest": rep.rolldown_forecast_backtest,
    }


@router.get("/rolldown/history", response_model=RolldownHistory, tags=["vietlott"])
def rolldown_history_endpoint(spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> RolldownHistory:
    """v3 reconstruction (sales surges, no jackpot series); see /rolldown/official for the official series."""
    _require_rolldown(spec)
    rep = state.market(spec)
    if rep is None or rep.rolldowns is None:
        raise InsufficientDataError("no rolldown history; run `vietlott market`")
    return rep.rolldowns
