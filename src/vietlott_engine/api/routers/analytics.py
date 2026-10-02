"""Statistical analytics, probabilistic models and ML diagnostics."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from vietlott_engine.analytics.cooccurrence import CooccurrenceReport, cooccurrence_report
from vietlott_engine.analytics.gaps import GapReport, gap_report
from vietlott_engine.analytics.randomness import RandomnessReport, randomness_report
from vietlott_engine.api.deps import AppState, get_state, resolve_game
from vietlott_engine.core.games import GameSpec
from vietlott_engine.ml_models.features import build_snapshots
from vietlott_engine.ml_models.gcn import GCNConfig, GCNPredictor, SkillReport, walk_forward_skill
from vietlott_engine.probability.bayesian import BayesianReport, PredictiveScore, bayesian_report, tune_decay
from vietlott_engine.probability.markov import MarkovReport, markov_report

router = APIRouter(prefix="/games/{game}")


@router.get("/analytics/randomness", response_model=RandomnessReport, tags=["analytics"])
def randomness(spec: GameSpec = Depends(resolve_game), mc_sims: int = Query(1000, ge=100, le=20_000), seed: int | None = 0, state: AppState = Depends(get_state)) -> RandomnessReport:
    return randomness_report(state.history(spec), mc_sims=mc_sims, seed=seed)


@router.get("/analytics/gaps", response_model=GapReport, tags=["analytics"])
def gaps(spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> GapReport:
    return gap_report(state.history(spec))


@router.get("/analytics/cooccurrence", response_model=CooccurrenceReport, tags=["analytics"])
def cooccurrence(spec: GameSpec = Depends(resolve_game), top: int = Query(15, ge=1, le=200), mc_sims: int = Query(300, ge=50, le=5000), seed: int | None = 0, state: AppState = Depends(get_state)) -> CooccurrenceReport:
    return cooccurrence_report(state.history(spec), top=top, mc_sims=mc_sims, seed=seed)


@router.get("/probability/bayesian", response_model=BayesianReport, tags=["probability"])
def bayesian(spec: GameSpec = Depends(resolve_game), alpha0: float = Query(1.0, gt=0), decay: float = Query(1.0, gt=0, le=1), state: AppState = Depends(get_state)) -> BayesianReport:
    return bayesian_report(state.history(spec), alpha0=alpha0, decay=decay)


@router.get("/probability/bayesian/decay-scan", response_model=list[PredictiveScore], tags=["probability"])
def decay_scan(spec: GameSpec = Depends(resolve_game), alpha0: float = Query(1.0, gt=0), state: AppState = Depends(get_state)) -> list[PredictiveScore]:
    return tune_decay(state.history(spec), alpha0=alpha0)


@router.get("/probability/markov", response_model=MarkovReport, tags=["probability"])
def markov(spec: GameSpec = Depends(resolve_game), sims: int = Query(300, ge=50, le=5000), seed: int | None = 0, state: AppState = Depends(get_state)) -> MarkovReport:
    return markov_report(state.history(spec), sims=sims, seed=seed)


@router.get("/ml/gcn/skill", response_model=SkillReport, tags=["ml"])
def gcn_skill(
    spec: GameSpec = Depends(resolve_game),
    first_test: int = Query(900, ge=100),
    refit_every: int = Query(150, ge=20),
    model: str = Query("gcn", pattern="^(gcn|logistic)$"),
    state: AppState = Depends(get_state),
) -> SkillReport:
    h = state.history(spec)
    return walk_forward_skill(build_snapshots(h), h.k, min(first_test, len(h) - 20), refit_every, GCNConfig(epochs=40), model=model)


@router.get("/ml/gcn/next", tags=["ml"])
def gcn_next(spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> dict:
    h = state.history(spec)
    snaps = build_snapshots(h)
    model = GCNPredictor(snaps.x.shape[2], GCNConfig(epochs=40), h.k / h.n).fit(snaps, 30, len(h))
    p = model.predict_proba(snaps, len(h))[0]
    base = h.k / h.n
    order = np.argsort(-p)
    return {
        "game": spec.code.value,
        "trained_on_draws": len(h),
        "baseline_probability": base,
        "numbers": [{"number": int(i + 1), "probability": float(p[i]), "lift": float(p[i] / base)} for i in order],
        "note": "Check /ml/gcn/skill first: without positive out-of-sample skill these probabilities are noise around k/n.",
    }
