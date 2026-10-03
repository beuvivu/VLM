"""ML probability reports. Synchronous handlers run in FastAPI's worker threads."""
from fastapi import APIRouter, Depends, HTTPException, Query

from vietlott_engine.api.deps import AppState, get_state
from vietlott_engine.core.products import get_product
from vietlott_engine.forecast.data import load_series
from vlm.forecast.pipeline import MLConfig
from vlm.forecast.service import directory_for, forecast_guard, model_config, refresh_snapshot

router = APIRouter(prefix='/ml/forecast', tags=['ML forecast'])


def _code(product: str):
    try:
        return get_product(product)
    except KeyError as exc:
        raise HTTPException(404, detail='Unknown product') from exc


def _run(state: AppState, product: str, *, config: MLConfig | None = None, refit: bool = False):
    code = _code(product)
    try:
        with forecast_guard(directory_for(state)):
            series = load_series(code, repository=state.repository, store=state.product_store(), seed_dir=state.settings.product_seed_dir)
            return refresh_snapshot(code.value, series, directory_for(state), config=config or model_config(state.settings), refit=refit)
    except ImportError as exc:
        raise HTTPException(503, detail='Requested ML backend is unavailable; install the ml extra') from exc
    except RuntimeError as exc:
        raise HTTPException(409, detail='Forecast state is being written by another process') from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, detail='Invalid historical data or ML checkpoint') from exc


@router.get('/{product}')
def next_draw(product: str, top_n: int = Query(5, ge=1, le=100),
              budget: int = Query(0, ge=0, le=1_000_000), state: AppState = Depends(get_state)) -> dict:
    model, _, summary = _run(state, product)
    return {**model.report(top_n, budget), 'learning':summary}


@router.post('/{product}/fit')
def fit(product: str, config: MLConfig, state: AppState = Depends(get_state)) -> dict:
    """An explicit fit resets prior online weights and live evidence for this product."""
    model, _, summary = _run(state, product, config=config, refit=True)
    return {**model.report(), 'learning':summary}
