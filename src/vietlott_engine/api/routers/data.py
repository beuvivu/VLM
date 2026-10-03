"""Health, game metadata, stored draws and data sync."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from vietlott_engine.api.deps import AppState, get_state, resolve_game
from vietlott_engine.api.schemas import SyncRequest
from vietlott_engine.core.games import GAMES, GameSpec

router = APIRouter()


@router.get('/updates/status', tags=['system'])
async def updates_status(state: AppState = Depends(get_state)) -> dict:
    from vlm.updates.service import update_status
    return update_status(state)


@router.get("/health", tags=["system"])
def health(state: AppState = Depends(get_state)) -> dict:
    return {
        "status": "ok",
        "storage": type(state.repository).__name__,
        "draws": {g.code.value: state.repository.count(g.code) for g in GAMES.values()},
    }


@router.get("/games", tags=["system"])
def list_games() -> list[dict]:
    return [
        {
            "code": g.code.value,
            "name": g.display_name,
            "pool_size": g.pool_size,
            "pick": g.pick,
            "bonus_mode": g.bonus_mode,
            "special_pool_size": g.bonus_pool_size,
            "ticket_price": g.ticket_price,
            "combinations": g.total_combinations,
            "draws_per_day": g.draws_per_day,
            "bao_levels": list(g.bao_levels),
            "p_any_prize": g.p_any_prize,
            "tiers": [
                {"name": t.name, "matches": t.main_matches, "matches_max": t.main_matches_max, "bonus": t.bonus_required, "prize": t.fixed_amount, "probability": g.tier_probabilities[t.name]}
                for t in g.tiers
            ],
            "min_jackpots": dict(g.min_jackpots),
        }
        for g in GAMES.values()
    ]


@router.get("/games/{game}/draws", tags=["data"])
def get_draws(spec: GameSpec = Depends(resolve_game), limit: int = Query(20, ge=1, le=5000), state: AppState = Depends(get_state)) -> dict:
    h = state.history(spec)
    rows = h.tail(limit).to_rows()
    return {"game": spec.code.value, "total": len(h), "id_gaps": h.id_gaps()[:20], "draws": rows[::-1]}


@router.post("/games/{game}/sync", tags=["data"])
async def sync(body: SyncRequest | None = None, spec: GameSpec = Depends(resolve_game), state: AppState = Depends(get_state)) -> dict:
    body = body or SyncRequest()
    from vlm.updates.service import sync_matrix
    return await sync_matrix(state, spec, source=body.source, full_refresh=body.full_refresh)
