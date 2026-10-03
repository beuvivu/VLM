"""Proper mixture laws; bounded ranking never labels an unproved search as exact."""
from __future__ import annotations

import heapq
from itertools import combinations, islice, product
from math import comb, log

import numpy as np
from scipy.special import logsumexp

from vietlott_engine.forecast.data import DigitSpec, SetSpec
from vietlott_engine.forecast.engine import inclusion_probabilities
from vietlott_engine.game_theory.popularity import elementary_symmetric


class Law:
    def __init__(self, spec: SetSpec | DigitSpec, values: np.ndarray, mixture: np.ndarray):
        self.spec = spec
        self.values = np.asarray(values, dtype=float)
        self.mixture = np.asarray(mixture, dtype=float)
        nodes = (spec.n,) if isinstance(spec, SetSpec) else (spec.positions, spec.alphabet)
        if (self.values.shape != (len(self.mixture), *nodes) or len(self.mixture) == 0
                or (self.values <= 0).any() or not np.isfinite(self.values).all()
                or (self.mixture < 0).any() or not np.isfinite(self.mixture).all()
                or self.mixture.sum() <= 0):
            raise ValueError('Invalid probability law')
        self.mixture = self.mixture / self.mixture.sum()
        if isinstance(spec, SetSpec):
            self.values = self.values / self.values.mean(axis=1, keepdims=True)
            self.log_den = np.log(elementary_symmetric(self.values, spec.k)[:, spec.k])
            self.fair = 1 / comb(spec.n, spec.k)
        else:
            self.values = self.values / self.values.sum(axis=-1, keepdims=True)
            self.log_den = np.zeros(len(self.mixture))
            self.fair = spec.alphabet ** (-spec.positions)
        self.log_values = np.log(self.values)
        with np.errstate(divide='ignore'):
            self.log_mix = np.log(self.mixture)

    def expert_log_likelihood(self, row: np.ndarray, bonus: int = 0) -> np.ndarray:
        row = np.asarray(row, dtype=float)
        if isinstance(self.spec, SetSpec):
            if row.shape != (self.spec.n,) or not np.isin(row, [0, 1]).all() or row.sum() != self.spec.k:
                raise ValueError('Invalid set observation')
            ll = self.log_values @ row - self.log_den
            if self.spec.bonus_same_drum:
                if bonus < 1 or bonus > self.spec.n or row[bonus - 1]:
                    raise ValueError('Invalid bonus observation')
                ll += self.log_values[:, bonus - 1] - np.log(self.values @ (1 - row))
            return ll
        counts = row.reshape(self.spec.positions, self.spec.alphabet)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or not (counts.sum(axis=1) == self.spec.per_draw).all():
            raise ValueError('Invalid digit observation')
        # The multinomial combinatorial factor cancels from model/null ratios.
        return np.einsum('jla,la->j', self.log_values, counts)

    def log_likelihood(self, row: np.ndarray, bonus: int = 0) -> float:
        return float(logsumexp(self.log_mix + self.expert_log_likelihood(row, bonus)))

    def null_log_likelihood(self, row: np.ndarray, bonus: int = 0) -> float:
        if isinstance(self.spec, SetSpec):
            return log(self.fair) - (log(self.spec.n - self.spec.k) if self.spec.bonus_same_drum else 0)
        return self.spec.per_draw * log(self.fair)

    def probability(self, numbers: list[int], bonus: int | None = None) -> float:
        if any(type(v) not in (int, np.int64, np.int32) for v in numbers):
            raise ValueError('Symbols must be integers')
        if isinstance(self.spec, SetSpec):
            if len(numbers) != self.spec.k or len(set(numbers)) != len(numbers) or any(v < 1 or v > self.spec.n for v in numbers):
                raise ValueError('Invalid combination')
            idx = np.array(numbers) - 1
            lp = self.log_values[:, idx].sum(axis=1) - self.log_den
            if bonus is not None:
                if not self.spec.bonus_same_drum or bonus in numbers or bonus < 1 or bonus > self.spec.n:
                    raise ValueError('Invalid bonus')
                mask = np.ones(self.spec.n, bool)
                mask[idx] = False
                lp += self.log_values[:, bonus - 1] - np.log(self.values[:, mask].sum(axis=1))
        else:
            idx = np.array(numbers) - self.spec.symbol_offset
            if len(numbers) != self.spec.positions or (idx < 0).any() or (idx >= self.spec.alphabet).any() or bonus is not None:
                raise ValueError('Invalid digit combination')
            lp = self.log_values[:, np.arange(self.spec.positions), idx].sum(axis=1)
        return float(np.exp(logsumexp(self.log_mix + lp)))

    def marginals(self) -> np.ndarray:
        if isinstance(self.spec, SetSpec):
            return self.mixture @ inclusion_probabilities(self.values, self.spec.k)
        return np.einsum('j,jla->la', self.mixture, self.values).ravel()

    def top(self, count: int, max_nodes: int = 20000) -> tuple[list[dict], bool]:
        if count < 1 or count > 100 or max_nodes < 1:
            raise ValueError('Top N must be 1..100 with a positive search budget')
        sp = self.spec
        if isinstance(sp, DigitSpec):
            all_rows = list(product(range(sp.symbol_offset, sp.alphabet + sp.symbol_offset), repeat=sp.positions))
            ranked = sorted(((self.probability(list(c)), c) for c in all_rows), key=lambda r:(-r[0], r[1]))[:count]
            return [{'numbers':list(c), 'p_model':p, 'p_fair':self.fair} for p, c in ranked], True
        space = comb(sp.n, sp.k)
        count = min(count, space)
        if np.allclose(self.values, 1, atol=1e-14, rtol=0) or space <= max_nodes:
            iterator = islice(combinations(range(1, sp.n + 1), sp.k), count) if np.allclose(self.values, 1, atol=1e-14, rtol=0) else combinations(range(1, sp.n + 1), sp.k)
            ranked = sorted(((self.probability(list(c)), c) for c in iterator), key=lambda r:(-r[0], r[1]))[:count]
            return [{'numbers':list(c), 'p_model':p, 'p_fair':self.fair} for p, c in ranked], True
        # Seed feasible leaves so a budget exhaustion can always return candidates.
        candidates: dict[tuple[int, ...], float] = {}
        for w in self.values:
            order = np.argsort(-w, kind='stable')
            mode = sorted((order[:sp.k] + 1).tolist())
            candidates[tuple(mode)] = self.probability(mode)
            for take in range(min(sp.k, count)):
                for extra in order[sp.k:sp.k + count]:
                    c = tuple(sorted(mode[:take] + mode[take+1:] + [int(extra) + 1]))
                    candidates[c] = self.probability(list(c))
        # Bound a branch by each expert's largest attainable suffix product.
        # Those maxima can differ; the mixture bound is still conservative.
        def bound(prefix: tuple[int, ...], start: int) -> float:
            need = sp.k - len(prefix)
            suffix = np.sort(self.log_values[:, start:], axis=1)[:, -need:] if need else np.zeros((len(self.mixture), 0))
            lp = self.log_values[:, prefix].sum(axis=1) if prefix else np.zeros(len(self.mixture))
            return float(logsumexp(self.log_mix + lp + suffix.sum(axis=1) - self.log_den))
        queue = [(-bound((), 0), (), 0)]
        examined = 0
        exact = False
        while queue and examined < max_nodes:
            nb, prefix, start = heapq.heappop(queue)
            examined += 1
            ranked = sorted(candidates.items(), key=lambda r:(-r[1], r[0]))
            threshold = log(ranked[count - 1][1]) if len(ranked) >= count else -np.inf
            if -nb < threshold - 1e-12:
                exact = True
                break
            if len(prefix) == sp.k:
                c = tuple(i + 1 for i in prefix)
                candidates[c] = self.probability(list(c))
                continue
            need = sp.k - len(prefix)
            for i in range(start, sp.n - need + 1):
                child = prefix + (i,)
                cb = bound(child, i + 1)
                if cb >= threshold - 1e-12:
                    heapq.heappush(queue, (-cb, child, i + 1))
            if len(queue) > max_nodes * 4:
                break  # bounded memory; cannot certify the omitted branches
        if not queue:
            exact = True
        ranked = sorted(candidates.items(), key=lambda r:(-r[1], r[0]))[:count]
        return [{'numbers':list(c), 'p_model':p, 'p_fair':self.fair} for c, p in ranked], exact

    def to_dict(self) -> dict:
        return {'values':self.values.tolist(), 'mixture':self.mixture.tolist()}
