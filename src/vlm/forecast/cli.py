"""Source-backed next/update/fit/benchmark and validated JSONL input."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from vietlott_engine.core.config import get_settings
from vietlott_engine.core.games import get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.models import Draw
from vietlott_engine.core.products import ProductCode, get_product, parse_product_rows
from vietlott_engine.forecast.data import CONFIGS, load_series, matrix_series, product_series
from vietlott_engine.paths import enter_project
from vlm.database.schema import DrawRecord
from vlm.forecast.pipeline import MLConfig, MLForecaster
from vlm.forecast.service import model_config, refresh_snapshot


def input_series(path: Path, code: ProductCode):
    raw = path.read_text(encoding='utf-8')
    rows = json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()]
    records = [DrawRecord.model_validate(row) if 'game_type' in row else DrawRecord.from_legacy(code.value, row) for row in rows]
    if any(r.game_type != code.value for r in records) or len({r.draw_id for r in records}) != len(records):
        raise ValueError('Mixed products or duplicate draw IDs in input')
    if code in (ProductCode.MEGA_645, ProductCode.POWER_655, ProductCode.LOTTO_535):
        spec = get_game(code.value)
        draws = [Draw(game=code.value, draw_id=r.draw_id, draw_date=r.draw_date.date(),
                      numbers=tuple(r.winning_numbers), bonus=r.bonus_number, source=str(path)) for r in records]
        return matrix_series(DrawHistory.from_draws(draws, spec))
    history, rejected = parse_product_rows(code, [{'id':r.draw_id, 'date':r.draw_date.date().isoformat(),
        'result':list(r.winning_numbers)} for r in records], str(path))
    if rejected:
        raise ValueError('Rejected input rows')
    return product_series(history)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='VLM ML: xác suất có kiểm định, học từng kỳ, benchmark ngoài mẫu')
    parser.add_argument('action', choices=['next', 'update', 'fit', 'benchmark'])
    parser.add_argument('--product', choices=['all', *[p.value for p in CONFIGS]], default='all')
    parser.add_argument('--input', type=Path, help='JSONL/JSON được validator kiểm tra; cần chọn một product')
    parser.add_argument('--dir', type=Path, help='forecast directory, with ML state in ml/')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--bootstrap', type=int)
    parser.add_argument('--backends', help='rf,xgb,lgb; boosters require pip install -e .[ml]')
    args = parser.parse_args(argv)
    enter_project()
    repo = None
    try:
        if args.input and args.product == 'all':
            raise ValueError('Select one product with --input')
        settings = get_settings()
        config = model_config(settings)
        overrides = config.model_dump()
        if args.bootstrap is not None:
            overrides['bootstrap'] = args.bootstrap
        if args.backends is not None:
            overrides['backends'] = tuple(args.backends.split(','))
        config = MLConfig.model_validate(overrides)
        directory = args.dir or settings.forecast_dir or settings.data_dir / 'forecast'
        products = list(CONFIGS) if args.product == 'all' else [get_product(args.product)]
        if not args.input:
            from vietlott_engine.api.deps import AppState
            from vietlott_engine.api.main import _seed
            from vietlott_engine.crawler.storage import DuckDBRepository, InMemoryRepository
            from vlm.updates.service import replay_results
            repo = InMemoryRepository() if settings.storage_backend == 'memory' else DuckDBRepository(settings.duckdb_path, settings.parquet_dir)
            seed_settings = settings.model_copy(update={'seed_file_dir':settings.seed_file_dir or settings.product_seed_dir})
            state = AppState(seed_settings, repo)
            asyncio.run(_seed(state))
            replay_results(state)
            store = state.product_store()
        reports = []
        for code in products:
            series = input_series(args.input, code) if args.input else load_series(code, repo, store, settings.product_seed_dir)
            started = time.perf_counter()
            if args.action == 'benchmark':
                model = MLForecaster(code, config)
                learned = model.update(series)
                summary = {'learned_draws':learned, 'issued':False, 'evaluation':'retrospective_prequential_exploratory'}
            else:
                model, _, summary = refresh_snapshot(code.value, series, directory, config=config,
                    refit=args.action == 'fit', issue=args.action == 'next', include_legacy=args.action == 'update',
                    require_config_match=args.backends is not None or args.bootstrap is not None)
            report = model.report(args.top_n, args.budget)
            reports.append({**report, 'learning':summary, 'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps(reports if args.product == 'all' else reports[0], ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Exception as exc:
        note = ('Dùng fit để đổi cấu hình đã lưu.' if isinstance(exc, ValueError) and str(exc) == 'config_change_requires_fit'
                else 'Kiểm tra lịch sử, cấu hình, dependency hoặc writer đang chạy.')
        print(json.dumps({'error':type(exc).__name__, 'note':note}, ensure_ascii=False))
        return 1
    finally:
        if repo is not None and callable(getattr(repo, 'close', None)):
            repo.close()


if __name__ == '__main__':
    raise SystemExit(main())
