"""Expected value, anti-popularity optimisation, wheels and backtests."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vietlott_engine.api.deps import AppState, get_state, resolve_game
from pydantic import BaseModel

from vietlott_engine.api.schemas import BacktestRequest, DecisionRequest, EVRequest, OptimizeRequest, UncertaintyRequest, WheelAPIRequest
from vietlott_engine.backtest.engine import BacktestConfig, BacktestReport, WalkForwardBacktester
from vietlott_engine.backtest.strategies import STRATEGY_REGISTRY
from vietlott_engine.core.exceptions import DataValidationError, InsufficientDataError
from vietlott_engine.core.games import GameSpec
from vietlott_engine.game_theory.decision import EVUncertainty, KellyResult, PayoutProfile, ev_uncertainty, kelly, payout_distribution, payout_profile
from vietlott_engine.game_theory.ev import EVBreakdown, EVCalculator, OptimizationResult, optimize_tickets
from vietlott_engine.game_theory.popularity import PopularityModel, PopularityParams
from vietlott_engine.wheeling.cover import WheelRequest, WheelResult, build_wheel

router = APIRouter(prefix="/games/{game}")


def _popularity(spec: GameSpec, state: AppState, quick_pick_share: float | None, use_last_draw: bool) -> PopularityModel:
    """Calibrated crowd model when ``data/calibration`` has one, literature prior otherwise."""
    history = None
    if use_last_draw:
        try:
            history = state.history(spec)
        except InsufficientDataError:
            history = None
    cal = state.calibration(spec)
    if cal is not None:
        model = cal.popularity_model(history, pattern_priors=True)
        return model.with_params(quick_pick_share=quick_pick_share) if quick_pick_share is not None else model
    last = tuple(int(x) for x in history.numbers[-1]) if history is not None else None
    qp = state.settings.quick_pick_share if quick_pick_share is None else quick_pick_share
    return PopularityModel(spec, PopularityParams(quick_pick_share=qp), last_draw=last)


def _calculator(spec: GameSpec, state: AppState, quick_pick_share: float | None = None, use_last_draw: bool = True) -> EVCalculator:
    calc = EVCalculator(spec, _popularity(spec, state, quick_pick_share, use_last_draw), sales=state.sales_model(spec))
    if calc.sales is None:
        calc.DEFAULT_SOLD = state.typical_tickets_sold(spec)
    return calc


@router.post("/ev", response_model=EVBreakdown, tags=["game-theory"])
def expected_value(body: EVRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> EVBreakdown:
    calc = _calculator(spec, state, body.quick_pick_share, body.use_last_draw)
    return calc.evaluate(body.ticket, body.jackpot1, body.jackpot2, body.tickets_sold, special=body.special)


@router.post("/ev/optimize", response_model=OptimizationResult, tags=["game-theory"])
def optimize(body: OptimizeRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> OptimizationResult:
    calc = _calculator(spec, state)
    return optimize_tickets(
        calc,
        n_tickets=body.n_tickets,
        jackpot1=body.jackpot1,
        jackpot2=body.jackpot2,
        tickets_sold=body.tickets_sold,
        max_overlap=body.max_overlap,
        exclude=tuple(body.exclude),
        iterations=body.iterations,
        seed=body.seed,
    )


class DecisionResponse(BaseModel):
    ev: EVBreakdown
    payout: PayoutProfile
    kelly: KellyResult


@router.post("/ev/decision", response_model=DecisionResponse, tags=["game-theory"])
def decision(body: DecisionRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> DecisionResponse:
    """Full payout distribution and growth-optimal (Kelly) stake for one ticket."""
    calc = _calculator(spec, state)
    sold = body.tickets_sold
    values, probs = payout_distribution(calc, body.ticket, body.jackpot1, body.jackpot2, sold, special=body.special)
    return DecisionResponse(
        ev=calc.evaluate(body.ticket, body.jackpot1, body.jackpot2, sold, special=body.special),
        payout=payout_profile(values, probs, spec.ticket_price),
        kelly=kelly(values, probs, spec.ticket_price, body.bankroll),
    )


@router.post("/ev/uncertainty", response_model=EVUncertainty, tags=["game-theory"])
def uncertainty(body: UncertaintyRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> EVUncertainty:
    """EV as a distribution over crowd-behaviour and sales priors (with sensitivity ranking)."""
    if body.tickets_sold_low > body.tickets_sold_high:
        raise DataValidationError("tickets_sold_low must not exceed tickets_sold_high")
    try:
        last = tuple(int(x) for x in state.history(spec).numbers[-1])
    except InsufficientDataError:
        last = None
    cal = state.calibration(spec) if body.use_calibration else None
    return ev_uncertainty(
        spec, body.ticket, body.jackpot1, (body.tickets_sold_low, body.tickets_sold_high), body.jackpot2, last, body.sims, body.seed, special=body.special, calibration=cal
    )


@router.post("/wheel", response_model=WheelResult, tags=["wheeling"])
def wheel(body: WheelAPIRequest, spec: GameSpec = Depends(resolve_game)) -> WheelResult:
    if body.guarantee > body.condition:
        raise DataValidationError("guarantee (t) cannot exceed condition (m)")
    try:
        return build_wheel(
            WheelRequest(pool=body.pool, ticket_size=spec.pick, guarantee=body.guarantee, condition=body.condition),
            spec,
            seed=body.seed,
            simulate=body.simulate,
            exact=body.exact,
            time_limit=body.time_limit,
        )
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc


@router.post("/backtest", response_model=BacktestReport, tags=["backtest"])
def backtest(body: BacktestRequest, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> BacktestReport:
    if len(body.strategies) > state.settings.max_backtest_strategies:
        raise DataValidationError(f"at most {state.settings.max_backtest_strategies} strategies per request")
    if body.tickets_per_draw > state.settings.max_backtest_tickets:
        raise DataValidationError(f"at most {state.settings.max_backtest_tickets} tickets per draw")
    names = list(dict.fromkeys(body.strategies))
    strategies = [STRATEGY_REGISTRY[s]() for s in names]
    h = state.history(spec)
    cfg = BacktestConfig(start=body.start, end=body.end, tickets_per_draw=body.tickets_per_draw, seed=body.seed)
    return WalkForwardBacktester(h, cfg, popularity=PopularityModel(spec)).run(strategies)
