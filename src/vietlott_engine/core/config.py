"""Application settings (12-factor): every value can be overridden with ``VQE_*`` env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VQE_", env_file=".env", extra="ignore")

    # --- storage
    data_dir: Path = Path("data")
    duckdb_file: str = "vietlott.duckdb"
    parquet_subdir: str = "parquet"

    # --- crawler
    http_backend: Literal["httpx", "curl_cffi"] = "httpx"
    source: Literal["github_mirror", "vietlott", "file", "nhanaz", "v130", "auto"] = "github_mirror"
    github_mirror_base_url: str = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data"
    prize_winners_base_url: str = "https://raw.githubusercontent.com/Compal123/vietlot-ai/main/data"
    prize_power_history_url: str = "https://raw.githubusercontent.com/leoodz/vn-vietlott/main/results/power6x55.json"
    vietlott_base_url: str = "https://www.vietlott.vn"  # serves Vietnamese IPs only (HTTP 403 elsewhere)
    vietlott_cookie_bootstrap: bool = True
    canonical_base_url: str = "https://raw.githubusercontent.com/pqminh-4/vietlott-data/main/data/canonical"
    product_mirror_base_url: str = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data"
    # fallback when vietlott.vn is unreachable (README §1.7)
    nhanaz_base_url: str = "https://raw.githubusercontent.com/NhanAZ-Data/vietlott-research/main/datasets"
    nhanaz_dir: Path | None = None  # local clone / ZIP of NhanAZ-Data/vietlott-research (or the 1.3.0 product cache)
    v130_dir: Path | None = None  # Vietlott Quant Engine 1.3.0 package root (its data/products snapshot)
    fallback_order: list[str] = ["vietlott", "nhanaz", "github_mirror", "v130"]
    http_timeout_s: float = 20.0
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_max_s: float = 30.0
    rate_limit_per_s: float = 2.0
    rate_limit_burst: int = 2
    max_concurrency: int = 4
    official_max_pages: int = 400
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) VietlottQuantEngine/1.0"

    # --- models
    default_decay: float = 1.0
    monte_carlo_sims: int = 1000
    random_seed: int = 20260930

    # --- game theory (see README: these are priors, calibrate when winner data is available)
    quick_pick_share: float = Field(default=0.35, ge=0.0, le=1.0)
    default_tickets_sold: int = 1_500_000  # fallback only: calibrated sales models are used when present
    calibration_dir: Path = Path("data/calibration")  # behaviour_*.json, market_*.json (vietlott market)
    anchors_file: Path = Path("data/seed/jackpot_anchors.json")  # published jackpot values used to close paths
    forecast_dir: Path | None = None  # self-learning forecaster state + ledger (v3.5); default <data_dir>/forecast
    product_seed_dir: Path = Path("data/seed")  # bundled Keno / Bingo18 / Max 3D histories (merged with data_dir/products)
    product_sync_max_pages: int = 50  # vietlott.vn results-list pages per product sync

    # --- API
    api_title: str = "Vietlott Quant Engine"
    # Không mở CORS cho mọi origin: API có lệnh ghi (sync, fit, record=true), nên "*" cho phép
    # bất kỳ trang nào người dùng đang mở gọi vào API chạy trên máy/mạng nội bộ của họ.
    # Cần thì khai rõ, ví dụ VQE_CORS_ORIGINS='["https://example.org"]'.
    cors_origins: list[str] = []
    # Đặt thì mọi lệnh ghi (POST/PUT/PATCH/DELETE và ?record=true) phải gửi
    # "Authorization: Bearer <token>". Đầu mục tuỳ biến buộc trình duyệt preflight, nên
    # một <form> hay <img> ở trang lạ không còn kích hoạt được lệnh ghi.
    api_token: str | None = None
    max_backtest_strategies: int = 8
    max_backtest_tickets: int = 50
    sync_on_startup: bool = False
    auto_update_enabled: bool = True
    auto_update_timeout_s: float = Field(default=180, gt=0)
    auto_update_dir: Path | None = None
    source_timeout_s: float = Field(default=20, gt=0)
    ml_auto_update_enabled: bool = True
    ml_bootstrap: int = Field(default=1000, ge=1, le=1000)
    ml_learning_timeout_s: float = Field(default=180, gt=0)
    ml_tree_every: int = Field(default=64, ge=1, le=1000)
    ml_search_nodes: int = Field(default=1000, ge=1, le=20000)
    ml_backends: list[Literal['rf', 'xgb', 'lgb']] = ['rf']
    storage_backend: Literal["duckdb", "memory"] = "duckdb"
    seed_file_dir: Path | None = None  # optional offline JSONL dir loaded on startup when the store is empty
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / self.duckdb_file

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / self.parquet_subdir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
