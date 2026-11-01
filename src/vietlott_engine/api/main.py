"""FastAPI application factory.

Run:  uvicorn vietlott_engine.api.main:app --host 127.0.0.1 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from vietlott_engine import __version__

import asyncio
import contextlib
import hmac
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from vietlott_engine.api.deps import AppState
from vietlott_engine.api.routers import analytics, catalog, data, forecast, inference, max3d, products, strategy
from vietlott_engine.core.config import Settings, get_settings
from vietlott_engine.core.exceptions import DataValidationError, InsufficientDataError, LookAheadError, SourceError, StorageError, VQEError
from vietlott_engine.core.games import GAMES
from vietlott_engine.core.logging import configure_logging, get_logger
from vietlott_engine.crawler.pipeline import SyncPipeline, build_http_client, build_source
from vietlott_engine.crawler.prize_sources import records_from_jsonl
from vietlott_engine.crawler.sources.mirror import JsonlFileSource
from vietlott_engine.crawler.storage import DrawRepository, DuckDBRepository, InMemoryRepository
from vlm.forecast.api import router as ml_router

log = get_logger(__name__)

_STATUS = {
    DataValidationError: 422,
    InsufficientDataError: 409,
    LookAheadError: 500,
    SourceError: 502,
    StorageError: 500,
}


def build_repository(settings: Settings) -> DrawRepository:
    if settings.storage_backend == "memory":
        return InMemoryRepository()
    try:
        return DuckDBRepository(settings.duckdb_path, settings.parquet_dir)
    except StorageError as exc:
        log.warning("DuckDB unavailable (%s); falling back to in-memory storage", exc)
        return InMemoryRepository()


async def _seed(state: AppState) -> None:
    """Load the bundled JSONL snapshot into an empty store (local, fast, blocking)."""
    s = state.settings
    if s.seed_file_dir is None or not s.seed_file_dir.exists():
        return
    for spec in GAMES.values():
        if state.repository.count(spec.code) == 0:
            report = await SyncPipeline(JsonlFileSource(s.seed_file_dir), state.repository).run(spec)
            log.info("seeded %s from %s: %d draws", spec.code.value, s.seed_file_dir, report.total_after)
        prize_file = s.seed_file_dir / f"prizes_{spec.code.value}.jsonl"
        if prize_file.exists() and not state.repository.load_prizes(spec.code):
            n = state.repository.upsert_prizes(records_from_jsonl(prize_file.read_text(encoding="utf-8")))
            log.info("seeded %d prize records for %s", n, spec.code.value)


async def _background_sync(state: AppState) -> None:
    """Incremental network sync after startup; the API serves stored data meanwhile."""
    for spec in GAMES.values():
        try:
            async with build_http_client(state.settings) as client:
                report = await SyncPipeline(build_source(state.settings, client), state.repository).run(spec)
            state.invalidate(spec)
            log.info("synced %s: +%d draws (total %d)", spec.code.value, report.inserted, report.total_after)
        except VQEError as exc:
            log.error("background sync of %s failed (serving stored data): %s", spec.code.value, exc)


def create_app(settings: Settings | None = None, repository: DrawRepository | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(settings=settings, repository=repository or build_repository(settings))
        app.state.vqe = state
        lease, task = None, None
        try:
            if settings.auto_update_enabled:
                from vlm.updates.lease import WriterLease
                from vlm.updates.service import journal_path, make_runner, replay_results
                lease = WriterLease(journal_path(state).parent / 'writer.lock')
                lease.__enter__()
            await _seed(state)
            if settings.auto_update_enabled:
                replay_results(state)
                state.updater = make_runner(state)
                task = asyncio.create_task(state.updater.run_forever(), name='vlm-periodic-results')
            else:
                task = asyncio.create_task(_background_sync(state)) if settings.sync_on_startup else None
            yield
        finally:
            try:
                try:
                    if task is not None:
                        if not task.done():
                            task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                finally:
                    state.updater = None
                    from vlm.updates.service import finish_learning
                    await finish_learning(state)
                    close = getattr(state.repository, "close", None)
                    if callable(close):
                        await asyncio.to_thread(close)
            finally:
                if lease is not None:
                    lease.__exit__()

    app = FastAPI(
        title=settings.api_title,
        version=__version__,
        description=(
            "Quantitative analytics for all Vietlott products — Mega 6/45, Power 6/55, Lotto 5/35, Keno, Bingo18, "
            "Max 3D / 3D+, Max 3D Pro and the Max 4D archive: exact odds, official prize data with a fallback chain when vietlott.vn is unreachable, "
            "import and cross-check of pages a person saved from vietlott.vn (its Cloudflare check is never bypassed), "
            "a self-learning next-draw forecaster for every product that reports anytime-valid evidence of whether it beats chance, "
            "crowd-behaviour calibration from winner counts, sales and jackpot models, anti-popularity EV, bao "
            "(system) tickets incl. Max 3D / 3D+ / 3D Pro, coverage portfolios, Lotto rolldowns, wheels, inference and "
            "walk-forward backtests. "
            "No endpoint can raise the probability of a ticket winning; see README §Kết luận định lượng."
        ),
        lifespan=lifespan,
    )
    _MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
    # GET làm mới dự báo ghi trạng thái; GET GCN huấn luyện lại ngay trong lượt yêu cầu.
    # Cả hai cần cùng cổng token/origin để khách lạ không tiêu CPU qua đường chỉ đọc.
    # /scoreboard chỉ đọc sổ nên không thuộc cổng này.
    _PROTECTED_GET = re.compile(
        r"^/(?:(?:ml/)?forecast/[^/]+(?:/evidence)?|games/[^/]+/ml/gcn/(?:skill|next))/?$"
    )
    _BOOL = TypeAdapter(bool)

    def _asks_to_record(request: Request) -> bool:
        """Đọc ``record`` ĐÚNG như FastAPI đọc tham số bool (pydantic): "on", "y", "t"… cũng là
        True, không chỉ "1"/"true"/"yes". Xét MỌI lần lặp của tham số. Giá trị không đọc được
        (FastAPI sẽ trả 422) vẫn tính là lệnh ghi: đóng an toàn."""
        for value in request.query_params.getlist("record"):
            try:
                if _BOOL.validate_python(value):
                    return True
            except ValidationError:
                return True
        return False

    def _sent_by_another_site(request: Request) -> bool:
        """Trình duyệt đánh dấu yêu cầu đến từ một trang khác (Fetch Metadata, hoặc Origin lạ).

        Client không phải trình duyệt (scheduler, curl) không gửi hai đầu mục này nên không bị
        ảnh hưởng. Cùng trang (Swagger UI ở /docs do chính API phục vụ) và Origin có trong
        VQE_CORS_ORIGINS là được tin."""
        site = request.headers.get("sec-fetch-site")
        if site in {"same-origin", "none"}:
            return False
        origin = request.headers.get("origin")
        if origin is not None:
            own = f"{request.url.scheme}://{request.url.netloc}"
            return origin != own and origin not in settings.cors_origins
        return site is not None

    @app.middleware("http")
    async def _require_token_for_writes(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Lệnh ghi cần token khi VQE_API_TOKEN được đặt; lệnh đọc giữ nguyên.

        Không đặt token thì lệnh ghi vẫn mở cho client cục bộ, nhưng KHÔNG cho trình duyệt gửi
        từ trang khác: CORS đóng chỉ ngăn đọc phản hồi, còn một "simple request" như
        ``GET /forecast/mega645?record=true`` từ trang lạ vẫn được gửi đi."""
        token = settings.api_token
        # Cùng đường dẫn tương đối mà router ASGI dùng: mount/reverse proxy có
        # root_path không được che một GET làm mới trạng thái khỏi xác thực.
        path = request.scope["path"]
        root_path = request.scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path):]
        writes = (request.method in _MUTATING or _asks_to_record(request)
                  or (request.method in {"GET", "HEAD"} and _PROTECTED_GET.match(path) is not None))
        if writes and token:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
                return JSONResponse(status_code=401, content={"error": "Unauthorized", "detail": "missing or invalid API token"})
        elif writes and _sent_by_another_site(request):
            return JSONResponse(status_code=403, content={"error": "Forbidden", "detail": "cross-site write without an API token"})
        return await call_next(request)

    # CORS thêm SAU cùng nên nằm NGOÀI cùng: phản hồi 401/403 trả sớm ở trên vẫn mang đầu mục
    # CORS cho trang được phép, thay vì thành lỗi mạng mờ.
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(VQEError)
    async def _vqe_error(_: Request, exc: VQEError) -> JSONResponse:
        status = next((code for cls, code in _STATUS.items() if isinstance(exc, cls)), 400)
        return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)})

    app.include_router(data.router)
    app.include_router(analytics.router)
    app.include_router(strategy.router)
    app.include_router(inference.router)
    app.include_router(products.router)
    app.include_router(max3d.router)
    app.include_router(catalog.router)
    app.include_router(forecast.router)
    app.include_router(ml_router)
    if settings.api_token:
        base_openapi = app.openapi

        def _openapi_with_bearer() -> dict:
            """Cho Swagger gửi token theo đúng các thao tác middleware bảo vệ."""
            schema = base_openapi()
            schema.setdefault("components", {}).setdefault("securitySchemes", {})["VqeBearer"] = {
                "type": "http",
                "scheme": "bearer",
                "description": "Nhập VQE_API_TOKEN trong Authorize; Swagger tự thêm tiền tố Bearer.",
            }
            for path, operations in schema["paths"].items():
                for method, operation in operations.items():
                    if method.upper() in _MUTATING or (
                        method.upper() in {"GET", "HEAD"} and _PROTECTED_GET.match(path) is not None
                    ):
                        operation["security"] = [{"VqeBearer": []}]
            return schema

        app.openapi = _openapi_with_bearer
    return app


app = create_app()
