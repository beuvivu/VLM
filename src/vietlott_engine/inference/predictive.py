"""Forecast evaluation with correct inference.

* ``newey_west_variance`` — HAC long-run variance (Bartlett kernel, Newey–West 1994
  automatic lag), because walk-forward loss differentials are serially correlated
  (models refit on overlapping windows).
* ``diebold_mariano`` — test of equal predictive accuracy with the Harvey–Leybourne–
  Newbold small-sample correction and Student-t reference.
* ``spiegelhalter_z`` — calibration test for probability forecasts.
* ``reliability_table`` — binned predicted vs observed frequencies.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats


def newey_west_lag(t: int) -> int:
    return int(np.floor(4 * (t / 100) ** (2 / 9)))


def andrews_lag(x: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    t = x.size
    rho = float(np.dot(x[1:], x[:-1]) / max(np.dot(x[:-1], x[:-1]), 1e-300))
    rho = float(np.clip(rho, -0.97, 0.97))
    a1 = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(min(t - 1, max(1, np.ceil(1.1447 * (a1 * t) ** (1 / 3)))))


def newey_west_variance(x: np.ndarray, lag: int | None = None) -> float:
    """HAC estimate of the long-run variance of a mean-stationary series."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    t = x.size
    lag = newey_west_lag(t) if lag is None else lag
    v = float(np.dot(x, x) / t)
    for j in range(1, min(lag, t - 1) + 1):
        w = 1 - j / (lag + 1)
        v += 2 * w * float(np.dot(x[j:], x[:-j]) / t)
    return max(v, 1e-300)


class DMResult(BaseModel):
    mean_differential: float
    hac_standard_error: float
    statistic: float
    p_value_two_sided: float
    p_value_greater: float  # H1: first forecaster better (positive differential)
    lag: int
    observations: int


def diebold_mariano(differential: np.ndarray, horizon: int = 1, lag: int | None = None) -> DMResult:
    """DM test on d_t = score_model − score_benchmark (higher = better)."""
    d = np.asarray(differential, dtype=np.float64)
    t = d.size
    lag = max(horizon - 1, newey_west_lag(t), andrews_lag(d)) if lag is None else lag
    se = np.sqrt(newey_west_variance(d, lag) / t)
    stat = float(np.mean(d) / se)
    hln = np.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
    stat_adj = stat * hln
    return DMResult(
        mean_differential=float(np.mean(d)),
        hac_standard_error=float(se),
        statistic=stat_adj,
        p_value_two_sided=float(2 * stats.t.sf(abs(stat_adj), t - 1)),
        p_value_greater=float(stats.t.sf(stat_adj, t - 1)),
        lag=lag,
        observations=t,
    )


class CalibrationResult(BaseModel):
    spiegelhalter_z: float
    p_value: float
    bins: list[dict]


def spiegelhalter_z(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=np.float64).ravel()
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = float(num / den) if den > 0 else 0.0
    return z, float(2 * stats.norm.sf(abs(z)))


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"mean_predicted": float(p[m].mean()), "observed": float(y[m].mean()), "count": int(m.sum())})
    return out


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 8) -> CalibrationResult:
    z, pv = spiegelhalter_z(p, y)
    return CalibrationResult(spiegelhalter_z=z, p_value=pv, bins=reliability_table(p, y, bins))
