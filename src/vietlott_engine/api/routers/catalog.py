"""All seven Vietlott products: catalogue, data coverage, draws, odds, randomness, sync.

Mega 6/45, Power 6/55 and Lotto 5/35 have their full feature set under ``/games/{game}``;
this router adds Keno, Bingo18, Max 3D (3D+) and Max 3D Pro.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from vietlott_engine.analytics.products import DigitReplication, ProductRandomnessReport, digit_replication, product_randomness
from vietlott_engine.api.deps import AppState, get_state
from vietlott_engine.api.schemas import ProductSyncRequest
from vietlott_engine.core.products import DIGIT_PRODUCTS, PRODUCT_INFO, SEED_FILES, ProductCode, get_product
from vietlott_engine.game_theory.fastgames import BingoBetOdds, KenoBacOdds, SideBetOdds, bingo18_odds, keno_odds

router = APIRouter(prefix="/products", tags=["products"])


class ProductEntry(BaseModel):
    code: str
    name: str
    draws: str
    schedule: str
    results_page: str
    endpoints: str
    stored_draws: int
    first_date: str | None = None
    last_date: str | None = None
    last_id: int | None = None
    missing_ids_inside_range: int | None = None
    source: str | None = None


class KenoOdds(BaseModel):
    bac: list[KenoBacOdds]
    side_bets: list[SideBetOdds]
    note: str


def _code(product: str, non_matrix: bool = True) -> ProductCode:
    try:
        code = get_product(product)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if non_matrix and code not in SEED_FILES:
        raise HTTPException(status_code=404, detail=f"{code.value}: use /games/{code.value}/… for Mega, Power and Lotto")
    return code


@router.get("", response_model=list[ProductEntry])
def catalogue(state: AppState = Depends(get_state)) -> list[ProductEntry]:
    """Every Vietlott product with its draw rules, schedule and the data stored for it."""
    out = []
    for code, info in PRODUCT_INFO.items():
        entry = ProductEntry(
            code=code.value, name=info.display_name, draws=info.draws, schedule=info.schedule,
            results_page="https://vietlott.vn" + info.results_path,
            endpoints=f"/games/{code.value}" if info.matrix_game else f"/products/{code.value}", stored_draws=0,
        )
        if info.matrix_game:
            draws = state.repository.load(code.value)
            if draws:
                entry.stored_draws = len(draws)
                entry.first_date, entry.last_date = str(draws[0].draw_date), str(draws[-1].draw_date)
                entry.last_id = draws[-1].draw_id
        else:
            try:
                cov = state.product_history(code).coverage()
            except Exception:  # noqa: BLE001 — an empty store is reported as 0 draws
                cov = {}
            if cov:
                entry.stored_draws = cov["draws"]
                entry.first_date, entry.last_date, entry.last_id = cov["first_date"], cov["last_date"], cov["last_id"]
                entry.missing_ids_inside_range, entry.source = cov["missing_ids_inside_range"], cov["source"]
        out.append(entry)
    return out


@router.get("/keno/odds", response_model=KenoOdds)
def keno() -> KenoOdds:
    """Exact hypergeometric odds and return to player of every Keno bậc and side bet."""
    bacs, sides = keno_odds()
    return KenoOdds(bac=bacs, side_bets=sides, note="Bảng giải tổng hợp từ đại lý/báo chí; mục verified=false có các phiên bản khác nhau (RTP từng phiên bản ở alternatives_rtp / variants).")


@router.get("/bingo18/odds", response_model=list[BingoBetOdds])
def bingo18() -> list[BingoBetOdds]:
    """Exact odds and return to player of every Bingo18 bet (216 equally likely outcomes)."""
    return bingo18_odds()


@router.get("/max3d/digit-check", response_model=DigitReplication)
async def max3d_digit_check(state: AppState = Depends(get_state)) -> DigitReplication:
    """Strongest digit bias found in Max 3D, tested out of sample on Max 3D Pro."""
    a, b = state.product_history(ProductCode.MAX3D), state.product_history(ProductCode.MAX3D_PRO)
    return await asyncio.to_thread(lambda: state._cached("rand:digit", lambda: digit_replication(a, b)))


@router.get("/{product}/draws")
def draws(product: str, limit: int = Query(50, ge=1, le=5000), state: AppState = Depends(get_state)) -> dict:
    """Most recent draws, newest first (Max 3D numbers as 3-digit strings, tiers 2/4/6/8)."""
    code = _code(product)
    h = state.product_history(code)
    rows = []
    for i in range(len(h) - 1, max(len(h) - limit, 0) - 1, -1):
        vals = [int(x) for x in h.values[i]]
        n = DIGIT_PRODUCTS.get(code)
        rows.append({"draw_id": int(h.draw_ids[i]), "date": str(h.dates[i]), "result": [f"{x:0{n}d}" for x in vals] if n else vals})
    return {"product": code.value, "coverage": h.coverage(), "draws": rows}


@router.get("/{product}/randomness", response_model=ProductRandomnessReport)
async def randomness(product: str, state: AppState = Depends(get_state)) -> ProductRandomnessReport:
    """Goodness-of-fit and independence tests against the exact law of a fair draw (BH-adjusted)."""
    code = _code(product)
    h = state.product_history(code)
    return await asyncio.to_thread(lambda: state._cached(f"rand:{code.value}", lambda: product_randomness(h)))


@router.post("/{product}/sync")
async def sync(product: str, body: ProductSyncRequest | None = None, state: AppState = Depends(get_state)) -> dict:
    """Fetch new draws (mirror / canonical / vietlott.vn) into the local product store."""
    from vlm.updates.service import sync_product

    code = _code(product)
    body = body or ProductSyncRequest()
    return await sync_product(state, code, source=body.source, max_pages=body.max_pages, full=body.full)
