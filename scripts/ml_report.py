"""Reproduce source-backed retrospective ML diagnostics; never certify them as live."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vietlott_engine.api.deps import AppState
from vietlott_engine.api.main import _seed
from vietlott_engine.core.config import Settings
from vietlott_engine.crawler.storage import InMemoryRepository
from vietlott_engine.forecast.data import CONFIGS, load_series
from vlm.forecast.pipeline import MLConfig, MLForecaster
from vlm.updates.service import replay_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--backends', default='rf')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = MLConfig(bootstrap=args.bootstrap, backends=tuple(args.backends.split(',')), search_nodes=50)
    settings = Settings(seed_file_dir=Path('data/seed'), auto_update_dir=Path('data/results'),
                        data_dir=Path('data'), auto_update_enabled=False, ml_auto_update_enabled=False)
    state = AppState(settings, InMemoryRepository())
    asyncio.run(_seed(state))
    replay_results(state)
    results = []
    for code in CONFIGS:
        series = load_series(code, state.repository, state.product_store(), settings.product_seed_dir)
        started = time.perf_counter()
        forecaster = MLForecaster(code, config)
        learned = forecaster.update(series)
        report = forecaster.report(5)
        results.append({**report, 'elapsed_seconds':time.perf_counter()-started,
                        'learned_this_run':learned})
        print(json.dumps({'product':code.value, 'available':len(series), 'learned':learned,
                          'last_id':report['last_id'], 'seconds':round(results[-1]['elapsed_seconds'], 2)}, ensure_ascii=False), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'config':config.model_dump(mode='json'), 'products':results,
            'evaluation':'retrospective_prequential_exploratory',
            'data':'bundled seed plus committed results journal; local product caches created by replay',
            'replay_conflicts':state.replay_conflicts}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
