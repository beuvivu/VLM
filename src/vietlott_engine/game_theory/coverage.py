"""Portfolio optimiser: maximise the probability of winning *at least one* prize per draw
for a fixed budget of B tickets ("tăng tần suất trúng giải").

What can and cannot be improved
-------------------------------
* P(a given ticket wins any tier) is fixed by the rules (Mega 2.38 %, Power 1.33 %,
  Lotto 5/35 9.60 %). Nothing changes it.
* The expected payout of B tickets is B × the single-ticket EV for *any* choice of
  tickets (linearity of expectation) — except jackpot sharing, handled elsewhere.
* What *does* depend on the choice is the overlap between tickets: P(at least one
  prize) = P(∪ events) ≤ B·p. Tickets that share numbers tend to win together; tickets
  spread over the pool rarely do. A bao ticket is the extreme of overlap.

Method (sample-average approximation of a maximum-coverage problem):
1. Simulate S draws (bitmasks).
2. Greedy selection from a random candidate pool — a (1−1/e)-approximation for coverage.
3. Local search: replace single numbers (and Lotto special numbers) while the number of
   covered draws increases.
4. Re-evaluate on an *independent* sample (no optimisation bias) with a 95 % interval;
   compare with B independent quick picks and with the B·p upper bound.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vietlott_engine.core.games import GameSpec


class CoverageResult(BaseModel):
    game: str
    tickets: list[list[int]]
    specials: list[int] | None = None
    budget_tickets: int
    cost: int
    min_tier: str
    p_at_least_one: float
    p_at_least_one_ci95: tuple[float, float]
    p_random_tickets: float
    p_random_ci95: tuple[float, float]
    upper_bound: float  # min(1, B·p): disjoint events
    expected_prizes_per_draw: float  # identical for every portfolio of B tickets
    evaluation_draws: int
    note: str


def _masks(arr: np.ndarray) -> np.ndarray:
    return np.bitwise_or.reduce(np.left_shift(np.uint64(1), arr.astype(np.uint64) - np.uint64(1)), axis=1)


def _sim_draws(spec: GameSpec, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, k = spec.pool_size, spec.pick
    w = np.argpartition(rng.random((size, n)), k, axis=1)[:, :k] + 1
    sp = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, size) if spec.separate_special else np.zeros(size, int)
    return _masks(w), sp


def _win_matrix(spec: GameSpec, ticket_masks: np.ndarray, ticket_sp: np.ndarray, draw_masks: np.ndarray, draw_sp: np.ndarray, min_main: dict) -> np.ndarray:
    """(tickets, draws) bool: ticket wins a tier at or above the threshold."""
    m = np.bitwise_count(ticket_masks[:, None] & draw_masks[None, :])
    if spec.separate_special:
        hit = ticket_sp[:, None] == draw_sp[None, :]
        return np.where(hit, m >= min_main["hit"], m >= min_main["miss"])
    return m >= min_main["any"]


def _thresholds(spec: GameSpec, min_tier: str) -> dict:
    """Smallest main-match count that reaches ``min_tier`` (or better), per special condition."""
    order = [t.name for t in spec.tiers]
    if min_tier not in order:
        raise ValueError(f"unknown tier {min_tier!r}; choose from {order}")
    ok = set(order[: order.index(min_tier) + 1])
    out = {}
    for cond, hit in (("any", False), ("hit", True), ("miss", False)):
        cands = [m for m in range(spec.pick + 1) if (t := spec.classify(m, hit)) is not None and t.name in ok]
        out[cond] = min(cands) if cands else spec.pick + 1
    return out


def _wilson(p: float, n: int) -> tuple[float, float]:
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def optimise_coverage(
    spec: GameSpec,
    budget: int,
    min_tier: str | None = None,
    sim_draws: int = 30_000,
    eval_draws: int = 200_000,
    candidates: int = 4000,
    local_rounds: int = 3,
    seed: int | None = 0,
) -> CoverageResult:
    if budget < 1:
        raise ValueError("budget must be ≥ 1 ticket")
    n, k = spec.pool_size, spec.pick
    min_tier = min_tier or spec.tiers[-1].name
    thr = _thresholds(spec, min_tier)
    rng = np.random.default_rng(seed)
    dm, dsp = _sim_draws(spec, sim_draws, rng)
    b_sp = int(spec.bonus_pool_size or 1)

    cand = np.sort(np.argpartition(rng.random((candidates, n)), k, axis=1)[:, :k] + 1, axis=1)
    cand_sp = rng.integers(1, b_sp + 1, candidates) if spec.separate_special else np.zeros(candidates, int)
    cmask = _masks(cand)
    win = _win_matrix(spec, cmask, cand_sp, dm, dsp, thr)
    covered = np.zeros(sim_draws, dtype=np.int32)
    chosen: list[int] = []
    for _ in range(budget):
        gains = (win & (covered == 0)[None, :]).sum(axis=1)
        if chosen:
            gains[chosen] = -1
        i = int(np.argmax(gains))
        chosen.append(i)
        covered += win[i]
    tickets = cand[chosen].copy()
    specials = cand_sp[chosen].copy()

    def row_win(t: np.ndarray, s: int) -> np.ndarray:
        return _win_matrix(spec, _masks(t[None, :]), np.array([s]), dm, dsp, thr)[0]

    rows = [row_win(t, s) for t, s in zip(tickets, specials)]
    for _ in range(local_rounds):
        improved = False
        for i in range(budget):
            base = rows[i]
            covered_wo = covered - base
            cur = int(np.sum(base & (covered_wo == 0)))
            for _try in range(k * 4):
                t2 = tickets[i].copy()
                pos = int(rng.integers(k))
                choices = np.setdiff1d(np.arange(1, n + 1), t2)
                t2[pos] = int(rng.choice(choices))
                t2.sort()
                s2 = int(specials[i])
                if spec.separate_special and rng.random() < 0.3:
                    s2 = int(rng.integers(1, b_sp + 1))
                r2 = row_win(t2, s2)
                g2 = int(np.sum(r2 & (covered_wo == 0)))
                if g2 > cur:
                    tickets[i], specials[i], rows[i] = t2, s2, r2
                    covered = covered_wo + r2
                    covered_wo = covered - r2
                    cur = g2
                    improved = True
        if not improved:
            break

    # unbiased evaluation on fresh draws, in chunks; the baseline draws a *new* random
    # portfolio for every simulated draw, so it estimates E over quick-pick portfolios
    rng_eval = np.random.default_rng(None if seed is None else seed + 1)
    hits = 0
    hits_rand = 0
    tmask = _masks(tickets)
    done = 0
    chunk = max(1000, min(50_000, 4_000_000 // budget))
    while done < eval_draws:
        m = min(chunk, eval_draws - done)
        em, esp = _sim_draws(spec, m, rng_eval)
        hits += int(np.any(_win_matrix(spec, tmask, specials, em, esp, thr), axis=0).sum())
        rt = np.argpartition(rng_eval.random((m * budget, n)), k, axis=1)[:, :k] + 1
        rmask = _masks(rt).reshape(m, budget)
        cnt = np.bitwise_count(rmask & em[:, None])
        if spec.separate_special:
            rsp = rng_eval.integers(1, b_sp + 1, (m, budget))
            win_r = np.where(rsp == esp[:, None], cnt >= thr["hit"], cnt >= thr["miss"])
        else:
            win_r = cnt >= thr["any"]
        hits_rand += int(win_r.any(axis=1).sum())
        done += m
    p = hits / eval_draws
    pr = hits_rand / eval_draws
    order = [t.name for t in spec.tiers]
    p_tier = sum(spec.tier_probabilities[nm] for nm in order[: order.index(min_tier) + 1])
    return CoverageResult(
        game=spec.code.value,
        tickets=[[int(x) for x in t] for t in tickets],
        specials=[int(s) for s in specials] if spec.separate_special else None,
        budget_tickets=budget,
        cost=budget * spec.ticket_price,
        min_tier=min_tier,
        p_at_least_one=p,
        p_at_least_one_ci95=_wilson(p, eval_draws),
        p_random_tickets=pr,
        p_random_ci95=_wilson(pr, eval_draws),
        upper_bound=float(min(1.0, budget * p_tier)),
        expected_prizes_per_draw=float(budget * p_tier),
        evaluation_draws=eval_draws,
        note=(
            "Expected number of prizes (and expected payout) is the same for every portfolio of this size; "
            "the optimised tickets only reduce overlap so that wins are spread over more draws."
        ),
    )
