"""Bounded streaming features. A snapshot is always emitted before its observation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import numpy as np

from vietlott_engine.forecast.data import DigitSpec, SetSpec


def spec_dict(spec: SetSpec | DigitSpec) -> dict:
    return asdict(spec)


def read_spec(data: dict) -> SetSpec | DigitSpec:
    if data['kind'] == 'set':
        return SetSpec(**data)
    data = dict(data)
    data['position_labels'] = tuple(data['position_labels'])
    return DigitSpec(**data)


class FeatureState:
    names = ['symbol', 'position', 'freq_all', 'ma10', 'ma50', 'ema10', 'ema100',
             'log_gap', 'gap_censored', 'lag1', 'lag2', 'cycle10', 'pair_link',
             'triple_link', 'cluster_rate', 'last_sum', 'last_even', 'last_head',
             'last_tail', 'id_gap', 'elapsed_days']

    def __init__(self, spec: SetSpec | DigitSpec):
        self.spec = spec
        self.nodes = spec.n if isinstance(spec, SetSpec) else spec.positions * spec.alphabet
        self.base = spec.k / spec.n if isinstance(spec, SetSpec) else 1 / spec.alphabet
        self.seen = 0
        self.last_id: int | None = None
        self.last_date: str | None = None
        self.count = np.zeros(self.nodes)
        self.last_seen = np.zeros(self.nodes, dtype=np.int64)
        self.censored = np.ones(self.nodes, dtype=bool)
        self.ema = np.full((2, self.nodes), self.base)
        self.rows: list[np.ndarray] = []
        self.pairs = np.zeros((self.nodes, self.nodes))
        self.triples = np.zeros((self.nodes, self.nodes, self.nodes), dtype=np.float32)

    def validate(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=float)
        if row.shape != (self.nodes,) or not np.isfinite(row).all():
            raise ValueError('Invalid observation shape or nonfinite values')
        if isinstance(self.spec, SetSpec):
            if not np.isin(row, [0, 1]).all() or row.sum() != self.spec.k:
                raise ValueError('Invalid set observation')
        else:
            counts = row.reshape(self.spec.positions, self.spec.alphabet)
            if (row < 0).any() or (row != np.floor(row)).any() or not (counts.sum(axis=1) == self.spec.per_draw).all():
                raise ValueError('Invalid digit counts')
        return row

    def snapshot(self, next_id: int, next_date: str) -> np.ndarray:
        day = date.fromisoformat(str(next_date)[:10])
        if next_id < 1 or (self.last_id is not None and next_id <= self.last_id):
            raise ValueError('Draw IDs must strictly increase')
        if self.last_date is not None and day < date.fromisoformat(self.last_date):
            raise ValueError('Draw dates must not go backwards')
        n = self.nodes
        prev = self.rows[-1] if self.rows else np.zeros(n)
        prev2 = self.rows[-2] if len(self.rows) >= 2 else np.zeros(n)
        freq = (self.count + 2 * self.base) / (self.seen + 2)
        ma = [np.mean(self.rows[-m:], axis=0) if self.rows else np.full(n, self.base) for m in (10, 50)]
        support = np.flatnonzero(prev > 0)
        if len(support):
            pair = self.pairs[:, support].mean(axis=1) / max(1, self.seen)
            triple = self.triples[:, support][:, :, support].mean(axis=(1, 2)) / max(1, self.seen)
        else:
            pair = triple = np.zeros(n)
        graph = self.pairs.copy()
        np.fill_diagonal(graph, 0)
        neighbors = np.argsort(-graph, axis=1, kind='stable')[:, :min(3, n)]
        cluster = prev[neighbors].mean(axis=1)
        if isinstance(self.spec, SetSpec):
            symbols = np.arange(1, n + 1)
            positions = np.zeros(n)
            denom = self.spec.k
            symbol_scale = self.spec.n
        else:
            symbols = np.tile(np.arange(self.spec.alphabet) + self.spec.symbol_offset, self.spec.positions)
            positions = np.repeat(np.arange(self.spec.positions), self.spec.alphabet) / max(1, self.spec.positions - 1)
            denom = self.spec.positions
            symbol_scale = self.spec.alphabet
        gap = np.where(self.last_seen > 0, next_id - self.last_seen, next_id - (self.last_id or next_id) + self.seen)
        id_gap = max(0, next_id - self.last_id - 1) if self.last_id is not None else 0
        censored = self.censored | (id_gap > 0)
        cycle = self.rows[-10] if len(self.rows) >= 10 else np.zeros(n)
        context = [float(prev @ symbols / max(1, denom * symbol_scale)),
                   float(prev @ (symbols % 2 == 0) / max(1, denom)),
                   float(prev @ (symbols // 10) / max(1, denom * max(1, symbol_scale // 10))),
                   float(prev @ (symbols % 10) / max(1, denom * 9)),
                   float(np.log1p(id_gap)),
                   float((day - date.fromisoformat(self.last_date)).days) / 7 if self.last_date else 0.]
        out = np.column_stack([symbols / symbol_scale, positions, freq, *ma, *self.ema,
                               np.log1p(np.maximum(gap, 0)) / 5, censored, prev, prev2, cycle,
                               pair, triple, cluster, *[np.full(n, v) for v in context]])
        return np.clip(out, -5, 5).astype(np.float64)

    def observe(self, row: np.ndarray, draw_id: int, draw_date: str) -> None:
        row = self.validate(row)
        self.snapshot(draw_id, draw_date)  # validate ordering before mutating state
        if self.last_id is not None and draw_id > self.last_id + 1:
            self.censored[:] = True
        divisor = 1 if isinstance(self.spec, SetSpec) else self.spec.per_draw
        rate = row / divisor
        present = row > 0
        self.count += rate
        self.last_seen[present] = draw_id
        self.censored[present] = False
        decay = np.array([.5 ** (1 / 10), .5 ** (1 / 100)])[:, None]
        self.ema = decay * self.ema + (1 - decay) * rate
        self.pairs += np.outer(rate, rate)
        # Distinct nodes only: repeat dice faces at different positions are different nodes.
        support = np.flatnonzero(present)
        a, b, c = np.meshgrid(support, support, support, indexing='ij')
        distinct = (a != b) & (a != c) & (b != c)
        self.triples[a[distinct], b[distinct], c[distinct]] += (rate[a[distinct]] * rate[b[distinct]] * rate[c[distinct]]).astype(np.float32)
        self.rows = (self.rows + [rate.copy()])[-50:]
        self.seen += 1
        self.last_id, self.last_date = int(draw_id), str(draw_date)[:10]

    def to_dict(self) -> dict:
        return {'spec':spec_dict(self.spec), 'seen':self.seen, 'last_id':self.last_id,
                'last_date':self.last_date, **{name:getattr(self, name).tolist()
                                             for name in ('count', 'last_seen', 'censored', 'ema', 'pairs', 'triples')},
                'rows':[r.tolist() for r in self.rows]}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureState:
        out = cls(read_spec(data['spec']))
        out.seen, out.last_id, out.last_date = data['seen'], data['last_id'], data['last_date']
        for name in ('count', 'last_seen', 'censored', 'ema', 'pairs', 'triples'):
            current = getattr(out, name)
            value = np.array(data[name], dtype=current.dtype)
            if value.shape != current.shape or not np.isfinite(value).all():
                raise ValueError('Invalid feature checkpoint')
            setattr(out, name, value)
        out.rows = [np.array(r, dtype=float) for r in data['rows']]
        if len(out.rows) > 50 or any(r.shape != (out.nodes,) or not np.isfinite(r).all() for r in out.rows):
            raise ValueError('Invalid feature history')
        return out
