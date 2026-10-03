"""Once for cron/Actions; continuous for a machine without the API."""
import argparse
import asyncio
import json
from pathlib import Path

from vietlott_engine.core.config import get_settings
from vietlott_engine.paths import enter_project
from vlm.updates.lease import WriterLease
from vlm.updates.schedule import LIVE_PRODUCTS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='VLM: tự cập nhật kết quả theo lịch giờ Việt Nam')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Kiểm tra các sản phẩm đến hạn một lần')
    mode.add_argument('--status', action='store_true', help='Đọc checkpoint; không mở CSDL')
    p.add_argument('--product', choices=['all', *LIVE_PRODUCTS], default='all')
    p.add_argument('--state-dir', type=Path)
    p.add_argument('--force', action='store_true', help='Thử tất cả target đã chọn ngay')
    args = p.parse_args(argv)
    enter_project()
    settings = get_settings()
    directory = args.state_dir or settings.auto_update_dir or settings.data_dir / 'updates'
    settings = settings.model_copy(update={'auto_update_dir':directory})
    if args.status:
        path = directory / 'status.json'
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version':1, 'products':{}}
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    from vietlott_engine.api.deps import AppState
    from vietlott_engine.api.main import _seed
    from vietlott_engine.crawler.storage import DuckDBRepository, InMemoryRepository
    from vlm.updates.service import finish_learning, make_runner, replay_results

    try:
        with WriterLease(directory / 'writer.lock'):
            # A locked DuckDB file is an error, never an implicit memory fallback.
            repo = InMemoryRepository() if settings.storage_backend == 'memory' else DuckDBRepository(settings.duckdb_path, settings.parquet_dir)
            try:
                state = AppState(settings, repo)
                async def run():
                    try:
                        await _seed(state)
                        replay_results(state)
                        runner = make_runner(state)
                        if args.product != 'all':
                            runner.products = (args.product,)
                        if args.once:
                            return await runner.tick(force=args.force)
                        await runner.run_forever()
                    finally:
                        await finish_learning(state)
                result = asyncio.run(run())
                if result is not None:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    unhealthy = {'behind_schedule', 'not_advancing', 'invalid_schedule'}
                    failed = result.get('worker_error') or any(
                        entry.get('error') or entry.get('freshness', {}).get('status') in unhealthy
                        for product, entry in result['products'].items()
                        if args.product == 'all' or product == args.product)
                    return 1 if failed else 0
            finally:
                close = getattr(repo, 'close', None)
                if close:
                    close()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Do not print a proxy URL/token embedded in a network exception.
        print(json.dumps({'error':type(exc).__name__}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
