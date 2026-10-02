"""Ticket sales, jackpot accounting and jackpot dynamics.

Accounting identity (documented structure: a fixed share of revenue funds the prizes;
fixed prizes are paid first and most of the rest feeds the jackpot):

    ΔJ_t  =  s · R_t  −  F_t

ΔJ_t is the growth of all jackpot pots at draw t (relative to the previous pot, or to
the minimum after a win), F_t the fixed prizes paid, R_t the gross revenue and s the
effective share of revenue that ends up in that draw's prizes. Vietlott quotes 55 % of
revenue for "prizes and prize reserves"; the share that reaches the jackpot is lower.

``s`` is *estimated*, not assumed, from an exact identity: for any crowd behaviour,
E_W[P_m(W)] = H_m over uniformly random winning sets, hence E[y_m] = N·H_m and

    ŝ = Σ_t (ΔJ_t + F_t)/price · H_m  /  Σ_t y_tm .

Different tiers m give independent estimates; on Power 6/55 they agree (ŝ ≈ 0.41),
which validates the identity. Combined with the revenue Vietlott reports for its
matrix games this is the basis of the tickets-sold series N_t = R_t/price.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.core.games import GameSpec
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.prizes import PrizeHistory
from vietlott_engine.inference.predictive import newey_west_variance


def jackpot_growth(ph: PrizeHistory) -> np.ndarray:
    """ΔJ_t (D,): total growth of all jackpot pots at each draw (NaN when not computable)."""
    spec = ph.spec
    pots = ph.pots
    won = ph.jackpot_won()
    mins = np.array([spec.min_jackpots.get(j, 0) for j in ph.jackpot_tiers], dtype=np.float64)
    d = pots.shape[0]
    out = np.full(d, np.nan)
    for t in range(1, d):
        if np.isnan(pots[t]).any() or np.isnan(pots[t - 1]).any() or not ph.has_winners[t - 1]:
            continue
        base = np.where(won[t - 1], mins, pots[t - 1])
        out[t] = float(np.sum(pots[t] - base))
    return out


class PayoutShare(BaseModel):
    overall: float
    by_tier: dict[str, float]
    by_year: dict[str, float]
    ci95: tuple[float, float]
    draws: int


def estimate_payout_share(ph: PrizeHistory, dates: np.ndarray, tiers: tuple[str, ...] = ("third", "second")) -> PayoutShare:
    """Method-of-moments estimate of s (see module docstring), with a block-bootstrap CI."""
    spec = ph.spec
    acc = jackpot_growth(ph)
    fixed = ph.fixed_prizes_paid()
    ok = ~np.isnan(acc) & ~np.isnan(fixed) & (acc > 0)
    base = (acc + fixed) / spec.ticket_price
    hm = {t.name: spec.tier_probabilities[t.name] for t in spec.tiers}

    def share(mask: np.ndarray, tier_names: tuple[str, ...]) -> float:
        num = sum(np.sum(base[mask]) * hm[t] for t in tier_names)
        den = sum(np.nansum(ph.column(t)[mask]) for t in tier_names)
        return float(num / den)

    by_tier = {t.name: share(ok, (t.name,)) for t in spec.tiers if not t.is_jackpot}
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    by_year = {str(y): share(ok & (years == y), tiers) for y in sorted(set(years[ok]))}
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ok)
    boots = []
    block = 20
    for _ in range(500):
        starts = rng.integers(0, max(1, idx.size - block), int(np.ceil(idx.size / block)))
        pick = np.concatenate([idx[s : s + block] for s in starts])[: idx.size]
        m = np.zeros_like(ok)
        np.add.at(m, pick, True)
        num = sum(np.sum(base[pick]) * hm[t] for t in tiers)
        den = sum(np.sum(ph.column(t)[pick]) for t in tiers)
        boots.append(num / den)
    return PayoutShare(
        overall=share(ok, tiers),
        by_tier=by_tier,
        by_year=by_year,
        ci95=(float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        draws=int(ok.sum()),
    )


def tickets_sold_from_accounting(ph: PrizeHistory, share: float) -> np.ndarray:
    """N_t = (ΔJ_t + F_t) / (s · price); NaN where the jackpot series is incomplete."""
    acc = jackpot_growth(ph)
    n = (acc + ph.fixed_prizes_paid()) / (share * ph.spec.ticket_price)
    return np.where(np.isfinite(n) & (n > 0), n, np.nan)


# ------------------------------------------------------------- sales model
class SalesModel(BaseModel):
    """log N_t = a + b·x + c·x² + weekday + year effects, x = log J − log J_ref (OLS, HAC SEs).

    ``elasticity`` is the slope at the reference (median) jackpot; ``curvature`` c > 0 means
    sales accelerate at large jackpots ("jackpot fever", strong for Mega 6/45). Outside the
    fitted jackpot range the quadratic is continued along its tangent (no wild extrapolation)."""

    game: str
    elasticity: float
    elasticity_se: float
    intercept: float
    weekday_effects: dict[str, float]
    year_effects: dict[str, float]
    r2: float
    residual_sd: float
    draws: int
    reference_year: str
    curvature: float = 0.0
    curvature_se: float = 0.0
    log_jackpot_ref: float = 0.0
    x_range: tuple[float, float] = (-1e9, 1e9)

    def _shape(self, jackpot: float) -> tuple[float, float]:
        """(b·x + c·q(x), d/dx) with tangent continuation outside ``x_range``."""
        x = float(np.log(jackpot)) - self.log_jackpot_ref
        lo, hi = self.x_range
        xc = min(max(x, lo), hi)
        quad = self.curvature * (xc * xc + 2 * xc * (x - xc))
        return self.elasticity * x + quad, self.elasticity + 2 * self.curvature * xc

    def elasticity_at(self, jackpot: float) -> float:
        return self._shape(jackpot)[1]

    def predict(self, jackpot: float, weekday: int | None = None, year: str | None = None) -> float:
        yr = year or self.reference_year
        lg = self.intercept + self._shape(jackpot)[0] + self.year_effects.get(yr, 0.0)
        if weekday is not None:
            lg += self.weekday_effects.get(str(weekday), 0.0)
        else:
            lg += float(np.mean(list(self.weekday_effects.values()) or [0.0]))
        return float(np.exp(lg + 0.5 * self.residual_sd**2))


def advertised_jackpot(ph: PrizeHistory) -> np.ndarray:
    """Total jackpot on offer before each draw (pots carried in, or minimum after a win)."""
    spec = ph.spec
    mins = np.array([spec.min_jackpots.get(j, 0) for j in ph.jackpot_tiers], dtype=np.float64)
    won = ph.jackpot_won()
    out = np.full(ph.pots.shape[0], np.nan)
    for t in range(1, ph.pots.shape[0]):
        if np.isnan(ph.pots[t - 1]).any() or not ph.has_winners[t - 1]:
            continue
        out[t] = float(np.sum(np.where(won[t - 1], mins, ph.pots[t - 1])))
    return out


def fit_sales_model(h: DrawHistory, sold: np.ndarray, jackpot_before: np.ndarray) -> SalesModel:
    ok = np.isfinite(sold) & np.isfinite(jackpot_before) & (sold > 0) & (jackpot_before > 0)
    y = np.log(sold[ok])
    dates = h.dates[ok]
    years = (dates.astype("datetime64[Y]").astype(int) + 1970).astype(str)
    wd = ((dates.astype("datetime64[D]").astype(int) + 3) % 7).astype(int)  # 1970-01-01 was a Thursday
    uy = sorted(set(years))
    uw = sorted(set(wd))
    lj = np.log(jackpot_before[ok])
    ref = float(np.median(lj))
    xj = lj - ref
    cols = [np.ones(ok.sum()), xj, xj**2]
    names = ["const", "log_jackpot", "log_jackpot_sq"]
    for yv in uy[1:]:
        cols.append((years == yv).astype(float))
        names.append(f"y{yv}")
    for wv in uw[1:]:
        cols.append((wd == wv).astype(float))
        names.append(f"w{wv}")
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    # HAC standard error of the elasticity (sales shocks are serially correlated)
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]
    s = np.zeros((x.shape[1], x.shape[1]))
    lag = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    for j in range(lag + 1):
        g = u[j:].T @ u[: len(y) - j]
        wgt = 1.0 if j == 0 else 1 - j / (lag + 1)
        s += wgt * (g if j == 0 else g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    _ = newey_west_variance  # same Bartlett weighting as predictive.newey_west_variance
    year_eff = {uy[0]: 0.0, **{nm[1:]: float(b) for nm, b in zip(names, beta) if nm.startswith("y")}}
    wd_eff = {str(uw[0]): 0.0, **{nm[1:]: float(b) for nm, b in zip(names, beta) if nm.startswith("w")}}
    return SalesModel(
        game=h.spec.code.value,
        elasticity=float(beta[1]),
        elasticity_se=float(np.sqrt(cov[1, 1])),
        curvature=float(beta[2]),
        curvature_se=float(np.sqrt(cov[2, 2])),
        log_jackpot_ref=ref,
        x_range=(float(xj.min()), float(xj.max())),
        intercept=float(beta[0]),
        weekday_effects=wd_eff,
        year_effects=year_eff,
        r2=float(1 - resid.var() / y.var()),
        residual_sd=float(resid.std(ddof=x.shape[1])),
        draws=int(ok.sum()),
        reference_year=uy[-1],
    )


# ------------------------------------------------- jackpot reconstruction
class JackpotPath(BaseModel):
    game: str
    draw_ids: list[int]
    jackpot: list[float | None]  # pot value at each draw (after that draw's sales)
    anchor: str
    closure_error: float | None  # reconstructed − known value at the anchor (VND)


def reconstruct_jackpot(spec: GameSpec, ph: PrizeHistory, sold: np.ndarray, share: float, anchor_index: int | None = None, anchor_value: float | None = None) -> JackpotPath:
    """Rebuild a single-pot jackpot path from tickets sold, winners and s.

    Forward from every reset (pot = minimum + s·R − F at the first draw after a win);
    when an anchor (a published jackpot value) is given, the segment containing it is
    also rebuilt backwards from the anchor and the forward/backward mismatch at the
    anchor is returned as ``closure_error`` — a test of s, N_t and F_t together.
    """
    d = len(ph.draw_ids)
    j = np.full(d, np.nan)
    won = ph.jackpot_won()[:, 0]
    jmin = float(spec.min_jackpots["jackpot1"])
    fixed = ph.fixed_prizes_paid()
    acc = share * sold * spec.ticket_price - fixed
    started = False
    for t in range(d):
        if not np.isfinite(acc[t]):
            started = False
            continue
        if t > 0 and won[t - 1]:
            j[t] = jmin + acc[t]
            started = True
        elif started and np.isfinite(j[t - 1]):
            j[t] = j[t - 1] + acc[t]
        else:
            started = False
    closure = None
    if anchor_index is not None and anchor_value is not None:
        fwd = j[anchor_index]
        # backward fill inside the anchor's segment where the forward path is unknown
        jb = anchor_value
        for t in range(anchor_index, 0, -1):
            if np.isnan(j[t]):
                j[t] = jb
            if won[t - 1] or not np.isfinite(acc[t]):
                break
            jb -= acc[t]
        if np.isfinite(fwd):
            closure = float(fwd - anchor_value)
        j[anchor_index] = anchor_value
    return JackpotPath(
        game=spec.code.value,
        draw_ids=[int(x) for x in ph.draw_ids],
        jackpot=[None if np.isnan(v) else float(v) for v in j],
        anchor=f"draw index {anchor_index} = {anchor_value:,.0f}" if anchor_value else "none",
        closure_error=closure,
    )


def jackpot_hit_check(pop_ticket_prob_fn, sold: np.ndarray, won: np.ndarray, spec: GameSpec, winning: np.ndarray) -> dict:
    """Compare observed jackpot wins with the model: P(won_t) = 1 − exp(−N_t·p(W_t)).

    Returns observed wins, expected under the calibrated crowd and under uniform picking.
    """
    ok = np.isfinite(sold)
    p_pop = pop_ticket_prob_fn(winning[ok])
    exp_model = float(np.sum(1 - np.exp(-sold[ok] * p_pop)))
    exp_uniform = float(np.sum(1 - np.exp(-sold[ok] / spec.main_combinations)))
    obs = int(np.sum(won[ok]))
    return {
        "draws": int(ok.sum()),
        "observed_draws_with_jackpot_winner": obs,
        "expected_calibrated": exp_model,
        "expected_uniform": exp_uniform,
        "poisson_p_calibrated": float(2 * min(stats.poisson.cdf(obs, exp_model), stats.poisson.sf(obs - 1, exp_model))),
    }
