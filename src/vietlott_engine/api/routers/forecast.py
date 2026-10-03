"""Self-learning forecaster (v3.5): next-draw probabilities with anytime-valid evidence.

The model learns every new draw before answering; the first call for Keno learns ~300k
draws (about a minute) — run ``vietlott forecast fit`` once to prepare the state.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from vietlott_engine.api.deps import AppState, get_state
from vietlott_engine.core.products import ProductCode, get_product
from vietlott_engine.forecast.data import load_series
from vietlott_engine.forecast.engine import ForecastReport, record, refresh, scoreboard
from vietlott_engine.forecast.schedule import record_window
from vlm.forecast.service import forecast_guard

router = APIRouter(prefix="/forecast", tags=["forecast"])


class FitSummary(BaseModel):
    product: str
    learned_draws: int
    draws: int
    last_id: int | None
    log10_wealth: float
    max_log10_wealth: float
    scored_forecasts: int
    evidence_valid: bool = True


class EvidencePath(BaseModel):
    product: str
    component: str
    path: list[tuple[int, float]]
    threshold_log10: float


def _code(product: str) -> ProductCode:
    try:
        return get_product(product)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _dir(state: AppState) -> Path:
    s = state.settings
    return Path(s.forecast_dir or s.data_dir / "forecast")


def _series(state: AppState, code: ProductCode):  # type: ignore[no-untyped-def]
    return load_series(code, repository=state.repository, store=state.product_store(), seed_dir=state.settings.product_seed_dir)


def _refresh(state: AppState, code: ProductCode, refit: bool = False, last: int | None = None, series=None):  # type: ignore[no-untyped-def]
    series = series if series is not None else _series(state, code)
    with forecast_guard(_dir(state)):
        return refresh(code, series, _dir(state), refit=refit, last=last)


@router.get("/{product}", response_model=ForecastReport)
def next_draw(product: str, record_forecast: bool = Query(False, alias="record", description="write it to the ledger, to be scored after the draw"), state: AppState = Depends(get_state)) -> ForecastReport:
    """Learn any new draws, then forecast the next one (probabilities, picks, evidence, verdict).
    With ``record=true`` the forecast goes to the ledger only if its draw has not started yet."""
    code = _code(product)
    with forecast_guard(_dir(state)):
        series = _series(state, code)
        f, _, _ = _refresh(state, code, series=series)
        rep = f.forecast()
        if record_forecast:
            ok, note, target = record_window(code, series.dates)
            if ok:
                record(_dir(state), rep, pre_draw=True, target_time=target.isoformat() if target else None)
            rep.recorded, rep.record_note = ok, note if ok else f"không ghi vào sổ: {note}"
        return rep


@router.post("/{product}/fit", response_model=FitSummary)
def fit(product: str, last: int | None = Query(None, ge=50, description="learn only from the last N draws"), state: AppState = Depends(get_state)) -> FitSummary:
    """Forget the saved state and learn again from the stored history."""
    code = _code(product)
    f, learned, scored = _refresh(state, code, refit=True, last=last)
    st = f.state
    return FitSummary(product=code.value, learned_draws=learned, draws=st["draws"], last_id=st["last_id"], log10_wealth=round(st["log_wealth"] / 2.302585093, 4), max_log10_wealth=round(st["max_log_wealth"] / 2.302585093, 4), scored_forecasts=len(scored), evidence_valid=st.get("window") is None)


@router.get("/{product}/evidence", response_model=list[EvidencePath])
def evidence(product: str, state: AppState = Depends(get_state)) -> list[EvidencePath]:
    """log10 e-value after each draw (down-sampled): crossing 1.30 (×20) = evidence at α = 5 %."""
    code = _code(product)
    f, _, _ = _refresh(state, code)
    return [EvidencePath(product=code.value, component=n, path=[(int(a), float(b)) for a, b in c["path"]], threshold_log10=1.30103) for n, c in f.state["components"].items()]


@router.get("/{product}/scoreboard")
def track_record(product: str, state: AppState = Depends(get_state)) -> list[dict]:
    """Forecasts recorded before their draw and scored afterwards (``all`` for every product)."""
    return scoreboard(_dir(state), None if product == "all" else _code(product))
