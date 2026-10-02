"""Inferential statistics endpoints (v2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from vietlott_engine.api.deps import AppState, get_state, resolve_game
from vietlott_engine.core.games import GameSpec
from vietlott_engine.inference.changepoint import ChangepointReport, changepoint_report
from vietlott_engine.inference.hierarchical import HierarchicalReport, hierarchical_report
from vietlott_engine.inference.multiple_testing import PerNumberReport, per_number_deviations
from vietlott_engine.inference.power import PowerEquivalenceReport, power_equivalence_report
from vietlott_engine.inference.report import InferenceReport, inference_report
from vietlott_engine.inference.sequential import SequentialReport, sequential_report

router = APIRouter(prefix="/games/{game}/inference", tags=["inference"])


@router.get("/report", response_model=InferenceReport)
def full_report(spec: GameSpec = Depends(resolve_game), sims: int = Query(1000, ge=200, le=20_000), alpha: float = Query(0.05, gt=0, lt=0.5), seed: int | None = 0, state: AppState = Depends(get_state)) -> InferenceReport:
    """Fairness certificate: per-number FWER, power/equivalence, hierarchical Bayes, e-processes, change points."""
    return inference_report(state.history(spec), sims=sims, alpha=alpha, seed=seed)


@router.get("/per-number", response_model=PerNumberReport)
def per_number(spec: GameSpec = Depends(resolve_game), sims: int = Query(2000, ge=200, le=50_000), seed: int | None = 0, state: AppState = Depends(get_state)) -> PerNumberReport:
    return per_number_deviations(state.history(spec), sims=sims, seed=seed)


@router.get("/power", response_model=PowerEquivalenceReport)
def power(spec: GameSpec = Depends(resolve_game), alpha: float = Query(0.05, gt=0, lt=0.5), state: AppState = Depends(get_state)) -> PowerEquivalenceReport:
    return power_equivalence_report(state.history(spec), alpha)


@router.get("/hierarchical", response_model=HierarchicalReport)
def hierarchical(spec: GameSpec = Depends(resolve_game), material: float = Query(0.05, gt=0, lt=1), state: AppState = Depends(get_state)) -> HierarchicalReport:
    return hierarchical_report(state.history(spec), material=material)


@router.get("/sequential", response_model=SequentialReport)
def sequential(spec: GameSpec = Depends(resolve_game), alpha: float = Query(0.05, gt=0, lt=0.5), state: AppState = Depends(get_state)) -> SequentialReport:
    return sequential_report(state.history(spec), alpha)


@router.get("/changepoints", response_model=ChangepointReport)
def changepoints(spec: GameSpec = Depends(resolve_game), sims: int = Query(500, ge=100, le=5000), seed: int | None = 0, state: AppState = Depends(get_state)) -> ChangepointReport:
    return changepoint_report(state.history(spec), sims=sims, seed=seed)
