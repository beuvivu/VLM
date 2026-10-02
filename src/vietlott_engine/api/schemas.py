"""Request/response schemas of the HTTP API (domain results are returned as-is)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SyncRequest(BaseModel):
    source: Literal["github_mirror", "vietlott"] | None = None
    full_refresh: bool = False


TICKET = Field(min_length=5, max_length=6, examples=[[3, 17, 22, 35, 41, 44]], description="6 numbers (Mega/Power) or 5 (Lotto 5/35)")
SPECIAL = Field(default=None, ge=1, le=12, description="Lotto 5/35 special number 1..12")
SOLD = Field(default=None, ge=0, description="tickets bought by other players; default: calibrated sales model / typical sales")


class EVRequest(BaseModel):
    ticket: list[int] = TICKET
    special: int | None = SPECIAL
    jackpot1: float | None = Field(default=None, ge=0, description="VND; defaults to the game's minimum jackpot")
    jackpot2: float | None = Field(default=None, ge=0, description="Power 6/55 only")
    tickets_sold: int | None = SOLD
    quick_pick_share: float | None = Field(default=None, ge=0, le=1)
    use_last_draw: bool = Field(default=True, description="model players copying the latest result")


class OptimizeRequest(BaseModel):
    n_tickets: int = Field(default=5, ge=1, le=50)
    jackpot1: float | None = Field(default=None, ge=0)
    jackpot2: float | None = Field(default=None, ge=0)
    tickets_sold: int | None = SOLD
    max_overlap: int = Field(default=2, ge=0, le=5)
    exclude: list[int] = Field(default_factory=list)
    iterations: int = Field(default=20_000, ge=100, le=200_000)
    seed: int | None = None


class WheelAPIRequest(BaseModel):
    pool: list[int] = Field(min_length=7, max_length=30, examples=[[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 42, 44]])
    guarantee: int = Field(default=3, ge=2, le=6)
    condition: int = Field(default=4, ge=2, le=6)
    seed: int | None = 0
    simulate: bool = True
    exact: bool = Field(default=True, description="run the ILP to improve / certify the design")
    time_limit: float = Field(default=10.0, gt=0, le=60)


class DecisionRequest(BaseModel):
    ticket: list[int] = TICKET
    special: int | None = SPECIAL
    jackpot1: float | None = Field(default=None, ge=0)
    jackpot2: float | None = Field(default=None, ge=0)
    tickets_sold: int | None = SOLD
    bankroll: float = Field(default=1e9, gt=0, description="VND available for lottery play")


class UncertaintyRequest(BaseModel):
    ticket: list[int] = TICKET
    special: int | None = SPECIAL
    use_calibration: bool = Field(default=True, description="draw crowd parameters from the fitted calibration's sampling distribution")
    jackpot1: float = Field(gt=0)
    jackpot2: float | None = Field(default=None, ge=0)
    tickets_sold_low: int = Field(default=1_000_000, gt=0)
    tickets_sold_high: int = Field(default=4_000_000, gt=0)
    sims: int = Field(default=300, ge=50, le=3000)
    seed: int | None = 0


class BacktestRequest(BaseModel):
    strategies: list[Literal["random", "hot", "cold", "overdue", "bayes", "markov", "gcn", "anti_popularity", "wheel"]] = Field(
        default_factory=lambda: ["random", "hot", "cold", "overdue", "bayes", "markov", "anti_popularity"]
    )
    start: int = Field(default=300, ge=50)
    end: int | None = None
    tickets_per_draw: int = Field(default=1, ge=1, le=50)
    seed: int = 0


class PrizeSyncRequest(BaseModel):
    source: Literal["auto", "canonical", "vietlott", "nhanaz", "v130", "compal"] = Field(default="auto", description="auto = first that answers of vietlott → canonical → nhanaz → v130; canonical = official detail-page records (GitHub); vietlott = vietlott.vn detail pages (Vietnamese IP only); nhanaz = community archive; v130 = Vietlott Quant Engine 1.3.0 snapshot (VQE_V130_DIR); compal = older third-party tables")
    last: int = Field(default=20, ge=1, le=500, description="source=vietlott: most recent stored draws to fetch")
    winners_base_url: str | None = None
    power_history_url: str | None = None


class ProductSyncRequest(BaseModel):
    source: Literal["auto", "vietlott", "nhanaz", "mirror", "canonical", "v130"] = Field(default="auto", description="auto = vietlott → nhanaz → mirror/canonical → v130; vietlott = vietlott.vn (Vietnamese IP only); nhanaz = NhanAZ-Data/vietlott-research; mirror = vietvudanh/vietlott-data; canonical = pqminh-4 (Max 3D only); v130 = 1.3.0 snapshot")
    max_pages: int = Field(default=50, ge=1, le=5000)
    full: bool = False


class BaoRequest(BaseModel):
    numbers: list[int] = Field(min_length=4, max_length=18, examples=[[5, 12, 19, 26, 33, 40, 44]], description="k−1 numbers (Bao 5 / Bao 4) or ≥ k numbers (Bao v)")
    specials: list[int] | None = Field(default=None, max_length=12, description="Lotto 5/35: special numbers (2–12 with 5 main numbers = bao số đặc biệt)")
    jackpot1: float | None = Field(default=None, ge=0)
    jackpot2: float | None = Field(default=None, ge=0)
    after_tax: bool = False
    tax_basis: Literal["ticket", "play"] = Field(default="ticket", description="tax the bao ticket's total (ticket) or each play separately (play)")
    strict: bool = Field(default=False, description="reject options Vietlott does not sell (e.g. Bao 16)")
    tickets_sold: int | None = Field(default=None, gt=0, description="add the EV with other players sharing jackpots")
    include_outcomes: bool = Field(default=False, description="return every (hits, bonus) outcome, not only the prize table")


class BaoCompareRequest(BaseModel):
    numbers: list[int] = Field(min_length=4, max_length=15, examples=[[3, 9, 14, 22, 31, 38, 41, 44]])
    specials: list[int] | None = Field(default=None, max_length=12)
    sims: int = Field(default=60_000, ge=5_000, le=300_000)
    guarantee: int = Field(default=3, ge=2, le=5, description="bao rút gọn: at least this many matches on one ticket…")
    condition: int = Field(default=3, ge=2, le=6, description="…whenever at least this many drawn numbers are in the set")
    seed: int | None = 0


class Max3DRequest(BaseModel):
    kind: Literal["co_ban", "bao_bo_so", "bao_nhieu_bo_so", "dao_so", "bao_vi_tri"] = "co_ban"
    numbers: list[str] = Field(min_length=1, max_length=20, examples=[["123", "456"]], description="three-digit numbers; bao_vi_tri accepts '*' wildcards")
    stake_multiple: int = Field(default=1, ge=1, le=20)
    sims: int = Field(default=100_000, ge=10_000, le=500_000)
    seed: int | None = 0


class CoverageRequest(BaseModel):
    budget: int = Field(default=10, ge=1, le=200, description="tickets per draw")
    min_tier: str | None = Field(default=None, description="count only prizes at or above this tier (default: any prize)")
    sim_draws: int = Field(default=20_000, ge=2_000, le=60_000)
    eval_draws: int = Field(default=100_000, ge=10_000, le=400_000)
    candidates: int = Field(default=3000, ge=200, le=8000)
    local_rounds: int = Field(default=2, ge=0, le=5)
    seed: int | None = 0


class RolldownRequest(BaseModel):
    jackpot: float = Field(gt=0, examples=[20e9], description="pot to be distributed (VND)")
    tickets_sold: int | None = Field(default=None, gt=0, description="expected tickets in the rolldown draw (default: recent rolldown sales)")
