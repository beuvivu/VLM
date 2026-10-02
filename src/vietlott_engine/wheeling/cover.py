"""Wheeling system (bao lô tối ưu) via covering / lotto designs.

Problem L(v, k, m, t). Given a pool P of v chosen numbers, find the fewest
k-number tickets such that *whenever at least m of the drawn numbers lie in P,
some ticket matches at least t of them*. With m = t this is the classical
covering design C(v, k, t).

Algorithm
1. Greedy set cover over all m-subsets of the pool (bitmask + popcount,
   vectorised); candidate tickets are all C(v, k) k-subsets when that is
   tractable, otherwise a large random sample.
2. Redundancy pruning.
3. Simulated-annealing descent in the style of Nurmela & Östergård (1993): try to
   reach full coverage with one ticket fewer by moving single numbers between
   tickets, repeating while it keeps succeeding.
4. Exhaustive verification of the guarantee, plus the Schönheim lower bound
   when m = t so the gap to optimality is visible.

A wheel does **not** change expected value — every ticket in it has the same EV as
any other ticket. It reshapes the *distribution* of outcomes: it converts "the pool
hit m numbers" into a guaranteed tier, at the cost of buying many tickets.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb

import numpy as np
from pydantic import BaseModel, Field

from vietlott_engine.core.games import GameSpec


class WheelRequest(BaseModel):
    pool: list[int] = Field(min_length=2)
    ticket_size: int = 6
    guarantee: int = Field(ge=1, description="t: at least this many matches on one ticket")
    condition: int = Field(ge=1, description="m: ...whenever at least this many drawn numbers are in the pool")


class WheelStats(BaseModel):
    simulations: int
    p_condition_met: float
    p_any_prize: float
    expected_fixed_payout: float
    cost: int
    tier_hit_rates: dict[str, float]


class WheelResult(BaseModel):
    pool: list[int]
    ticket_size: int
    guarantee: int
    condition: int
    tickets: list[list[int]]
    n_tickets: int
    verified: bool
    uncovered: int
    schonheim_lower_bound: int | None
    lower_bound: int  # best certified bound: max(Schönheim, counting bound, ILP dual bound)
    optimality: str  # "proven_optimal" | "gap"
    optimality_gap: int
    method: str  # "heuristic" | "ilp"
    ilp_status: str | None = None
    exhaustive_candidates: bool
    stats: WheelStats | None = None


def schonheim_bound(v: int, k: int, t: int) -> int:
    """Schönheim lower bound on the covering number C(v, k, t)."""
    if t == 0:
        return 1
    return ceil(v / k * schonheim_bound(v - 1, k - 1, t - 1)) if t > 1 else ceil(v / k)


def counting_bound(v: int, k: int, t: int, m: int) -> int:
    """Every m-subset must be hit; one ticket hits at most Σ_{j≥t} C(k,j)·C(v−k, m−j) of them."""
    per_ticket = sum(comb(k, j) * comb(v - k, m - j) for j in range(t, min(k, m) + 1))
    return ceil(comb(v, m) / per_ticket)


def _masks(v: int, size: int) -> np.ndarray:
    return np.array([sum(1 << i for i in c) for c in combinations(range(v), size)], dtype=np.uint64)


def _popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x)


class CoveringDesigner:
    def __init__(self, v: int, k: int, t: int, m: int, max_candidates: int = 12_000, max_matrix: int = 150_000_000, seed: int | None = None) -> None:
        if not (1 <= t <= k and t <= m <= v and k <= v and v <= 60):
            raise ValueError(f"invalid design parameters v={v} k={k} m={m} t={t} (need t≤k, t≤m≤v, k≤v≤60)")
        self.v, self.k, self.t, self.m = v, k, t, m
        self.rng = np.random.default_rng(seed)
        self.targets = _masks(v, m)
        total = comb(v, k)
        cap = max(1, min(max_candidates, max_matrix // max(1, self.targets.size)))
        if total <= cap:
            self.candidates = _masks(v, k)
            self.exhaustive = True
        else:
            seen: set[int] = set()
            while len(seen) < cap:
                s = self.rng.choice(v, k, replace=False)
                seen.add(int(sum(1 << int(i) for i in s)))
            self.candidates = np.array(sorted(seen), dtype=np.uint64)
            self.exhaustive = False

    def covers(self, ticket_mask: int | np.uint64) -> np.ndarray:
        return _popcount(self.targets & np.uint64(ticket_mask)) >= self.t

    # ------------------------------------------------------------ construction
    def greedy(self) -> list[int]:
        cover = _popcount(self.candidates[:, None] & self.targets[None, :]) >= self.t
        uncovered = np.ones(self.targets.size, dtype=bool)
        chosen: list[int] = []
        while uncovered.any():
            gains = cover[:, uncovered].sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:  # sampled candidates cannot finish the job: add a direct fix
                target = int(self.targets[np.flatnonzero(uncovered)[0]])
                chosen.append(self._complete(target))
                uncovered &= ~self.covers(chosen[-1])
                continue
            chosen.append(int(self.candidates[best]))
            uncovered &= ~cover[best]
        return self.prune(chosen)

    def _complete(self, target: int) -> int:
        bits = [i for i in range(self.v) if target >> i & 1][: self.k]
        rest = [i for i in self.rng.permutation(self.v) if i not in bits]
        return int(sum(1 << i for i in (bits + rest)[: self.k]))

    def prune(self, tickets: list[int]) -> list[int]:
        counts = np.sum([self.covers(tk) for tk in tickets], axis=0)
        kept = list(tickets)
        for tk in sorted(tickets, key=lambda x: int(self.covers(x).sum())):
            cov = self.covers(tk)
            if np.all(counts[cov] >= 2):
                counts = counts - cov
                kept.remove(tk)
        return kept

    def anneal(self, tickets: list[int], iterations: int = 20_000, t0: float = 1.0, t1: float = 0.02) -> list[int] | None:
        """Search for a full cover of size len(tickets) starting from ``tickets``. None if not found."""
        bits = [[i for i in range(self.v) if tk >> i & 1] for tk in tickets]
        cov = np.array([self.covers(tk) for tk in tickets])
        counts = cov.sum(axis=0)
        cost = int(np.sum(counts == 0))
        for it in range(iterations):
            if cost == 0:
                return [int(sum(1 << i for i in b)) for b in bits]
            temp = t0 * (t1 / t0) ** (it / iterations)
            # focus moves on an uncovered target (as in Nurmela–Östergård)
            target = int(self.targets[self.rng.choice(np.flatnonzero(counts == 0))])
            j = int(self.rng.integers(len(bits)))
            members = bits[j]
            out_pos = int(self.rng.integers(self.k))
            candidates_in = [i for i in range(self.v) if (target >> i & 1) and i not in members]
            if not candidates_in:
                continue
            new = members.copy()
            new[out_pos] = int(self.rng.choice(candidates_in))
            new_mask = int(sum(1 << i for i in new))
            new_cov = self.covers(new_mask)
            new_counts = counts - cov[j] + new_cov
            new_cost = int(np.sum(new_counts == 0))
            if new_cost <= cost or self.rng.random() < np.exp((cost - new_cost) / temp):
                bits[j], cov[j], counts, cost = new, new_cov, new_counts, new_cost
        return [int(sum(1 << i for i in b)) for b in bits] if cost == 0 else None

    def solve(self, anneal_iterations: int = 50_000, max_rounds: int = 25, restarts: int = 2) -> list[int]:
        best = self.greedy()
        for _ in range(max_rounds):
            if self.m == self.t and len(best) <= schonheim_bound(self.v, self.k, self.t):
                break
            counts = np.sum([self.covers(tk) for tk in best], axis=0)
            unique = np.array([int(np.sum(self.covers(tk) & (counts == 1))) for tk in best])
            improved = None
            # drop the ticket(s) with the least unique coverage and try to repair
            for drop in np.argsort(unique)[: restarts]:
                start = [tk for i, tk in enumerate(best) if i != int(drop)]
                improved = self.anneal(start, anneal_iterations)
                if improved is not None:
                    break
            if improved is None:
                break
            best = improved
        return best

    def solve_ilp(self, time_limit: float = 10.0) -> tuple[list[int] | None, float | None, str]:
        """Exact set-cover ILP over the candidate tickets (HiGHS branch-and-cut).

        Returns (tickets or None, dual bound, status). The dual bound is a certified
        lower bound on the covering number only when the candidates are exhaustive.
        """
        from scipy import sparse
        from scipy.optimize import Bounds, LinearConstraint, milp

        cover = _popcount(self.candidates[:, None] & self.targets[None, :]) >= self.t
        a = sparse.csr_matrix(cover.T.astype(np.float64))
        n_var = cover.shape[0]
        res = milp(
            np.ones(n_var),
            constraints=LinearConstraint(a, lb=1, ub=np.inf),
            integrality=np.ones(n_var),
            bounds=Bounds(0, 1),
            options={"time_limit": time_limit, "disp": False, "presolve": True},
        )
        dual = getattr(res, "mip_dual_bound", None)
        status = "optimal" if res.status == 0 else ("time_limit" if res.status == 1 else f"status_{res.status}")
        if res.x is None:
            return None, dual, status
        chosen = [int(self.candidates[i]) for i in np.flatnonzero(res.x > 0.5)]
        return chosen, dual, status

    def uncovered(self, tickets: list[int]) -> int:
        covered = np.zeros(self.targets.size, dtype=bool)
        for tk in tickets:
            covered |= self.covers(tk)
        return int((~covered).sum())


def simulate_wheel(tickets: list[list[int]], pool: list[int], condition: int, spec: GameSpec, sims: int = 100_000, seed: int | None = None) -> WheelStats:
    """Monte Carlo outcome distribution of a wheel under uniformly random draws."""
    rng = np.random.default_rng(seed)
    n, k = spec.pool_size, spec.pick
    draws = np.argpartition(rng.random((sims, n)), k, axis=1)[:, :k]
    inc = np.zeros((sims, n + 1), dtype=bool)
    inc[np.arange(sims)[:, None], draws + 1] = True
    tk = np.asarray(tickets, dtype=int)
    matches = inc[:, tk].sum(axis=2)  # (sims, b)
    in_pool = inc[:, np.asarray(pool)].sum(axis=1)
    fixed = {t.main_matches: t for t in spec.tiers if t.fixed_amount is not None}
    payout = np.zeros(sims)
    for mm, tier in fixed.items():
        payout += (matches == mm).sum(axis=1) * tier.fixed_amount
    best = matches.max(axis=1)
    rates = {t.name: float(np.mean(best >= t.main_matches)) for t in spec.tiers if not t.bonus_required}
    return WheelStats(
        simulations=sims,
        p_condition_met=float(np.mean(in_pool >= condition)),
        p_any_prize=float(np.mean(best >= min(t.main_matches for t in spec.tiers))),
        expected_fixed_payout=float(payout.mean()),
        cost=len(tickets) * spec.ticket_price,
        tier_hit_rates=rates,
    )


def build_wheel(
    req: WheelRequest,
    spec: GameSpec | None = None,
    seed: int | None = None,
    anneal_iterations: int = 50_000,
    simulate: bool = True,
    exact: bool = True,
    time_limit: float = 10.0,
    max_ilp_size: int = 3_000_000,
) -> WheelResult:
    """Heuristic design, then (optionally) an exact ILP that either improves it, proves it
    optimal, or at least certifies a lower bound."""
    pool = sorted(set(req.pool))
    if len(pool) != len(req.pool):
        raise ValueError("pool contains duplicates")
    if spec is not None:
        if req.ticket_size != spec.pick:
            raise ValueError(f"{spec.display_name} tickets have {spec.pick} numbers")
        if any(not 1 <= x <= spec.pool_size for x in pool):
            raise ValueError(f"pool numbers must be in 1..{spec.pool_size}")
        if req.condition > spec.pick:
            raise ValueError("condition cannot exceed the number of drawn balls")
    v, k, t, m = len(pool), req.ticket_size, req.guarantee, req.condition
    designer = CoveringDesigner(v, k, t, m, seed=seed)
    masks = designer.solve(anneal_iterations)
    schon = schonheim_bound(v, k, t) if m == t else None
    lb = max(counting_bound(v, k, t, m), schon or 0)
    method, ilp_status = "heuristic", None
    if exact and len(masks) > lb and designer.candidates.size * designer.targets.size <= max_ilp_size:
        ilp_masks, dual, ilp_status = designer.solve_ilp(time_limit)
        if ilp_masks is not None and len(ilp_masks) < len(masks) and designer.uncovered(ilp_masks) == 0:
            masks, method = designer.prune(ilp_masks), "ilp"
        if designer.exhaustive and dual is not None and np.isfinite(dual):
            lb = max(lb, int(np.ceil(dual - 1e-6)))
        if designer.exhaustive and ilp_status == "optimal" and ilp_masks is not None:
            lb = max(lb, len(ilp_masks))
    tickets = [[pool[i] for i in range(v) if mk >> i & 1] for mk in masks]
    unc = designer.uncovered(masks)
    stats = simulate_wheel(tickets, pool, m, spec, seed=seed) if (simulate and spec is not None) else None
    return WheelResult(
        pool=pool,
        ticket_size=k,
        guarantee=t,
        condition=m,
        tickets=sorted(tickets),
        n_tickets=len(tickets),
        verified=unc == 0,
        uncovered=unc,
        schonheim_lower_bound=schon,
        lower_bound=lb,
        optimality="proven_optimal" if len(tickets) <= lb else "gap",
        optimality_gap=max(0, len(tickets) - lb),
        method=method,
        ilp_status=ilp_status,
        exhaustive_candidates=designer.exhaustive,
        stats=stats,
    )
