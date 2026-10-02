"""Gap (khoảng vắng) analysis.

Under H0 each number appears in a draw independently with p = k/n, so the gap
between appearances is Geometric(p) and the hazard — P(appears now | absent for g
draws) — is constant. A rising hazard is exactly what "overdue number" systems
assume; ``hazard_table`` measures it directly so the claim can be checked instead
of believed.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vietlott_engine.analytics.randomness import TestResult
from vietlott_engine.analytics.stats import pooled_gof, wilson_interval
from vietlott_engine.core.history import DrawHistory


class NumberGap(BaseModel):
    number: int
    appearances: int
    current_gap: int
    mean_gap: float | None
    max_gap: int
    tail_probability: float  # P(gap >= current_gap) under H0 — descriptive, NOT predictive


class HazardRow(BaseModel):
    gap: str
    at_risk: int
    events: int
    hazard: float
    ci_low: float
    ci_high: float


class GapReport(BaseModel):
    game: str
    draws: int
    expected_mean_gap: float
    numbers: list[NumberGap]
    geometric_fit: TestResult
    hazard: list[HazardRow]


def inter_arrival_gaps(h: DrawHistory) -> dict[int, np.ndarray]:
    """Gaps (in draws, ≥1) between consecutive appearances, per number."""
    out: dict[int, np.ndarray] = {}
    for i in range(h.n):
        idx = np.flatnonzero(h.incidence[:, i])
        out[i + 1] = np.diff(idx)
    return out


def current_gaps(h: DrawHistory) -> np.ndarray:
    """Draws since last appearance (0 = appeared in the latest draw; D if never)."""
    d = len(h)
    last = np.full(h.n, -1)
    rows, cols = np.nonzero(h.incidence)
    np.maximum.at(last, cols, rows)
    return np.where(last >= 0, d - 1 - last, d)


def hazard_table(h: DrawHistory, max_gap: int = 30) -> list[HazardRow]:
    """Empirical P(appear at next draw | current gap = g) pooled over numbers and time."""
    d, n = h.incidence.shape
    at_risk = np.zeros(max_gap + 2)
    events = np.zeros(max_gap + 2)
    gap = np.full(n, -1)  # unknown until first appearance
    for t in range(d):
        known = gap >= 0
        b = np.minimum(gap[known], max_gap + 1)
        np.add.at(at_risk, b, 1)
        np.add.at(events, b, h.incidence[t, known])
        gap = np.where(h.incidence[t], 0, np.where(gap >= 0, gap + 1, -1))
    rows = []
    for g in range(max_gap + 2):
        if at_risk[g] == 0:
            continue
        lo, hi = wilson_interval(events[g], at_risk[g])
        rows.append(
            HazardRow(
                gap=str(g) if g <= max_gap else f">{max_gap}",
                at_risk=int(at_risk[g]),
                events=int(events[g]),
                hazard=float(events[g] / at_risk[g]),
                ci_low=lo,
                ci_high=hi,
            )
        )
    return rows


def gap_report(h: DrawHistory, max_bin: int = 40) -> GapReport:
    h.require(20)
    p = h.spec.inclusion_probability
    gaps = inter_arrival_gaps(h)
    cur = current_gaps(h)
    numbers = []
    for i in range(1, h.n + 1):
        g = gaps[i]
        numbers.append(
            NumberGap(
                number=i,
                appearances=int(h.incidence[:, i - 1].sum()),
                current_gap=int(cur[i - 1]),
                mean_gap=float(g.mean()) if g.size else None,
                max_gap=int(g.max()) if g.size else int(cur[i - 1]),
                tail_probability=float((1 - p) ** cur[i - 1]),
            )
        )
    pooled = np.concatenate([g for g in gaps.values() if g.size])
    obs = np.bincount(np.minimum(pooled, max_bin + 1), minlength=max_bin + 2)[1:]
    gvals = np.arange(1, max_bin + 1)
    probs = np.append((1 - p) ** (gvals - 1) * p, (1 - p) ** max_bin)
    gof = pooled_gof(obs, probs, [str(g) for g in gvals] + [f">{max_bin}"])
    fit = TestResult(
        name="gap_geometric_fit",
        statistic=gof.g_statistic,
        df=gof.df,
        p_value=gof.p_value,
        method=f"G-test of pooled inter-arrival gaps vs Geometric(p={p:.4f})",
        detail={"observed_mean_gap": float(pooled.mean()), "expected_mean_gap": 1 / p, "n_gaps": int(pooled.size)},
    )
    return GapReport(
        game=h.spec.code.value,
        draws=len(h),
        expected_mean_gap=1 / p,
        numbers=numbers,
        geometric_fit=fit,
        hazard=hazard_table(h),
    )
