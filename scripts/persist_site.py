#!/usr/bin/env python3
"""Publication barrier: fetch current main and durably push allowed result ledgers.

Run ``sync`` after checkout (before restoring cache/updating), then ``persist``
before any Pages build or cache save. A failed pull/rebase/push exits nonzero.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILES = ('data/results/results.jsonl', 'data/forecast/ledger.jsonl', 'data/forecast/ml-ledger.jsonl')


def git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, text=True, encoding='utf-8',
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def sync() -> None:
    git('pull', '--rebase', '--autostash', 'origin', 'main')


def persist() -> None:
    changed = git('status', '--porcelain', '--', *FILES)
    if changed:
        git('config', 'user.name', 'github-actions[bot]')
        git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
        present = [p for p in FILES if Path(p).exists()]
        git('add', '--', *present)
        stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M')
        git('commit', '-m', f'data: kết quả và sổ dự báo {stamp} [skip ci]')
    for attempt in range(3):
        # Unconditional even when no local result changed: queued jobs may be stale.
        sync()
        if git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main'):
            return
        try:
            git('push', 'origin', 'HEAD:main')
            # Confirm that the remote accepted precisely this publication commit.
            git('fetch', 'origin', 'main')
            if git('rev-parse', 'HEAD') != git('rev-parse', 'origin/main'):
                raise RuntimeError('Remote advanced during publication; retry required')
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
    raise RuntimeError('Unable to synchronize publication branch')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sync', 'persist'))
    action = parser.parse_args().action
    try:
        sync() if action == 'sync' else persist()
    except (subprocess.CalledProcessError, RuntimeError) as error:
        print('Publication stopped: ' + (error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)), file=sys.stderr)
        return 1
    print('Repository synchronized; publication barrier passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
