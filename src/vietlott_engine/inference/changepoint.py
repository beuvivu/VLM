"""Change-point / regime detection.

Vietlott rotates machines and ball sets; a *local* bias (one ball set, a few months)
would be diluted in whole-history tests. Two scans look for it, both calibrated by
Monte Carlo on simulated fair histories of the same length, so the p-values already
account for scanning thousands of windows (no look-elsewhere bias):

* **Window scan** — the covariance-corrected frequency χ² over sliding windows of
  ~4 months to ~2 years; statistic = max over all windows.
* **Per-number CUSUM bridge** — for each number, max_t |S_t − (t/D)·S_D| /
  √(D·p(1−p)) with S_t the cumulative count; detects a number whose rate *shifted*
  at some point. Statistic = max over numbers and time.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vietlott_engine.analytics.stats import monte_carlo_p, simulate_incidence
from vietlott_engine.core.history import DrawHistory


def _windows(d: int, lengths: tuple[int, ...]) -> list[tuple[int, int]]:
    out = []
    for length in lengths:
        if length > d:
            continue
        step = max(1, length // 4)
        out.extend((s, s + length) for s in range(0, d - length + 1, step))
    return out


def window_scan_stats(inc: np.ndarray, k: int, windows: list[tuple[int, int]]) -> np.ndarray:
    n = inc.shape[1]
    cum = np.vstack([np.zeros(n), np.cumsum(inc, axis=0)])
    s = np.array([w[0] for w in windows])
    e = np.array([w[1] for w in windows])
    counts = cum[e] - cum[s]  # (W, n)
    lengths = (e - s).astype(np.float64)
    exp = lengths * k / n
    ss = ((counts - exp[:, None]) ** 2).sum(axis=1)
    return ss * n * (n - 1) / (lengths * k * (n - k))


def cusum_bridge(inc: np.ndarray, k: int) -> np.ndarray:
    """(D, n) standardised CUSUM bridge."""
    d, n = inc.shape
    p = k / n
    s = np.cumsum(inc, axis=0) - p * np.arange(1, d + 1)[:, None]
    bridge = s - (np.arange(1, d + 1) / d)[:, None] * s[-1][None, :]
    return bridge / np.sqrt(d * p * (1 - p))


class WindowScan(BaseModel):
    statistic: float
    p_value: float
    window_draws: int
    start_date: str
    end_date: str
    hottest_numbers: list[int]
    coldest_numbers: list[int]


class CusumScan(BaseModel):
    statistic: float
    p_value: float
    number: int
    change_date: str
    direction: str  # "rate rose after" / "rate fell after"


class ChangepointReport(BaseModel):
    game: str
    draws: int
    windows_scanned: int
    simulations: int
    window_scan: WindowScan
    cusum: CusumScan
    interpretation: str


def changepoint_report(h: DrawHistory, lengths: tuple[int, ...] = (52, 104, 156, 312), sims: int = 500, seed: int | None = None) -> ChangepointReport:
    h.require(max(60, min(lengths)))
    inc = h.incidence.astype(np.float64)
    d, n, k = len(h), h.n, h.k
    wins = _windows(d, lengths)
    scan = window_scan_stats(inc, k, wins)
    best = int(np.argmax(scan))
    cb = cusum_bridge(inc, k)
    t_star, i_star = np.unravel_index(np.argmax(np.abs(cb)), cb.shape)
    cus_stat = float(np.abs(cb).max())

    rng = np.random.default_rng(seed)
    null_scan = np.empty(sims)
    null_cus = np.empty(sims)
    for s in range(sims):
        sim = simulate_incidence(d, n, k, rng).astype(np.float64)
        null_scan[s] = window_scan_stats(sim, k, wins).max()
        null_cus[s] = np.abs(cusum_bridge(sim, k)).max()

    s0, e0 = wins[best]
    wc = inc[s0:e0].sum(axis=0)
    ws = WindowScan(
        statistic=float(scan[best]),
        p_value=monte_carlo_p(float(scan[best]), null_scan, "greater"),
        window_draws=e0 - s0,
        start_date=str(h.dates[s0]),
        end_date=str(h.dates[e0 - 1]),
        hottest_numbers=[int(i + 1) for i in np.argsort(-wc)[:3]],
        coldest_numbers=[int(i + 1) for i in np.argsort(wc)[:3]],
    )
    # bridge > 0 at t*: the number ran hot before t* relative to its overall rate, i.e. its rate fell after
    cs = CusumScan(
        statistic=cus_stat,
        p_value=monte_carlo_p(cus_stat, null_cus, "greater"),
        number=int(i_star + 1),
        change_date=str(h.dates[t_star]),
        direction="rate fell after" if cb[t_star, i_star] > 0 else "rate rose after",
    )
    sig = [name for name, p in (("window scan", ws.p_value), ("CUSUM", cs.p_value)) if p < 0.05]
    text = (
        f"Most unusual {ws.window_draws}-draw window: {ws.start_date} → {ws.end_date} (scan-adjusted p = {ws.p_value:.2f}); "
        f"largest rate shift: number {cs.number} around {cs.change_date} (p = {cs.p_value:.2f}). "
        + ("No regime with a detectable local bias." if not sig else f"Significant after scan adjustment: {sig} — check machine/ball-set logs for that period.")
    )
    return ChangepointReport(
        game=h.spec.code.value, draws=d, windows_scanned=len(wins), simulations=sims, window_scan=ws, cusum=cs, interpretation=text
    )
