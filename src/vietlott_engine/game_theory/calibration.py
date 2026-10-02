"""Calibrate the player-behaviour (number-popularity) model on real winner counts.

Why winner counts identify behaviour
------------------------------------
Let p(c) be the share of all tickets that hold combination c. For a draw with winning
set W, the expected number of tickets with exactly m matches is

    E[y_m | W] = N · Σ_c p(c)·1{|c∩W| = m} = N · P_m(W).

If everybody picked uniformly at random, P_m(W) would equal the hypergeometric H_m for
every W. Real crowds over-pick some numbers (birthdays, lucky numbers…), so draws whose
winning numbers are "popular" produce *more* winners than N·H_m and unpopular draws
fewer. Across ~1,400 draws with different W this variation identifies the popularity
weights — no survey needed. Averaged over uniformly random W, E_W[P_m(W)] = H_m exactly
for *any* p, which also pins down the absolute scale of N (see ``sales.py``).

Model
-----
    p(c, σ) = q · U(c)·U(σ) + (1 − q) · Π_{i∈c} w_ti / e_k(w_t) · π_t(σ)

* q: share of quick-pick (uniform) tickets;
* w_ti = exp(β_i + Σ_d γ_d X_tid): static per-number weight β_i plus dynamic effects
  — X_t1 = number was drawn last draw, X_t2 = "hot" (count in the last 10 draws,
  standardised), X_t3 = "overdue" (standardised log gap);
* π_t(σ) ∝ exp(η_σ + γ_s·1{σ = previous special}) for Lotto 5/35's special number.

Tier probabilities are exact: P(|c∩W| = m) under the product form is
e_m(w_W)·e_{k−m}(w_L)/e_k(w) (elementary symmetric polynomials), with analytic
gradients ∂log e_j(S)/∂β_i = w_i·e_{j−1}(S∖i)/e_j(S).

Likelihood
----------
* Ticket sales N_t known up to a per-year scale (Power 6/55, from the jackpot accounting
  identity): y_tg ~ NegBin(mean = N_t·e^{c_year}·P_tg, dispersion φ_g). The negative
  binomial absorbs the heavy tails created by syndicates and large "bao" tickets.
* N_t unknown (Mega 6/45, Lotto 5/35): y_t· | Σ_g y_tg ~ Multinomial over tiers — the
  exact conditional likelihood when counts are Poisson; N_t then has the closed-form
  estimate N̂_t = Σ_g y_tg / Σ_g P_tg.

Ridge penalties keep the 35–55 static weights stable; standard errors are sandwich
(robust) estimates with draws as independent units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats
from scipy.special import expit, gammaln, logit, logsumexp

from vietlott_engine.core.games import GameCode, GameSpec, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.logging import get_logger
from vietlott_engine.core.prizes import PrizeHistory
from vietlott_engine.game_theory.popularity import PopularityModel, PopularityParams, divide_out, elementary_symmetric

log = get_logger(__name__)

Cond = Literal["any", "hit", "miss"]
DYNAMIC_FEATURES = ("last_draw", "hot10", "overdue")


# ------------------------------------------------------------------ tiers
def observation_groups(spec: GameSpec) -> list[tuple[str, tuple[str, ...], tuple[tuple[int, Cond], ...]]]:
    """(group label, data tiers summed, model components (m main matches, special condition))."""
    if spec.code == GameCode.POWER_655:
        # the bonus ball only splits 5-match tickets into JP2 / first prize: merged
        return [
            ("6", ("jackpot1",), ((6, "any"),)),
            ("5", ("jackpot2", "first"), ((5, "any"),)),
            ("4", ("second",), ((4, "any"),)),
            ("3", ("third",), ((3, "any"),)),
        ]
    groups = []
    for t in spec.tiers:
        if spec.separate_special:
            cond: Cond = "hit" if t.bonus_required else "miss"
            comps = tuple((m, cond) for m in t.main_range())
        else:
            comps = tuple((m, "any") for m in t.main_range())
        groups.append((t.name, (t.name,), comps))
    return groups


# --------------------------------------------------------------- features
def dynamic_features(h: DrawHistory) -> np.ndarray:
    """(D, n, 3) features known *before* each draw: last draw, hot (10 draws), overdue."""
    inc = h.incidence.astype(np.float64)
    d, n = inc.shape
    k = h.k
    x = np.zeros((d, n, len(DYNAMIC_FEATURES)))
    if d == 0:
        return x
    x[1:, :, 0] = inc[:-1]
    cs = np.vstack([np.zeros(n), np.cumsum(inc, axis=0)])
    window = 10
    for t in range(1, d):
        lo = max(0, t - window)
        cnt = cs[t] - cs[lo]
        exp = (t - lo) * k / n
        sd = np.sqrt((t - lo) * (k / n) * (1 - k / n))
        x[t, :, 1] = (cnt - exp) / sd
    gap = np.zeros(n)
    p = k / n
    mean_lg = float(np.mean(np.log1p(np.random.default_rng(0).geometric(p, 200_000) - 1)))
    sd_lg = float(np.std(np.log1p(np.random.default_rng(1).geometric(p, 200_000) - 1)))
    for t in range(d):
        if t > 0:
            x[t, :, 2] = (np.log1p(gap) - mean_lg) / sd_lg
        gap = np.where(inc[t] > 0, 0.0, gap + 1.0)
    return x


def next_draw_features(h: DrawHistory) -> np.ndarray:
    """(n, 3) features for the draw after the last one in ``h``."""
    if len(h) == 0:
        return np.zeros((h.n, len(DYNAMIC_FEATURES)))
    ext = DrawHistory.from_arrays(h.spec, np.vstack([h.numbers, h.numbers[-1:]]))
    return dynamic_features(ext)[-1]


STATIC_FEATURES = ("le31_birthday", "le12_month", "ends_in_8", "ends_in_9", "ends_in_6", "ends_in_4", "is_7", "is_13")


def static_feature_matrix(n: int) -> np.ndarray:
    x = np.arange(1, n + 1)
    return np.column_stack(
        [x <= 31, x <= 12, x % 10 == 8, x % 10 == 9, x % 10 == 6, x % 10 == 4, x == 7, x == 13]
    ).astype(np.float64)


# ------------------------------------------------------------------- data
@dataclass
class CalibrationData:
    spec: GameSpec
    winning: np.ndarray  # (T, k) 1-based
    special: np.ndarray  # (T,) 1-based special number (separate drum) or 0
    prev_special: np.ndarray  # (T,) previous draw's special or 0
    y: np.ndarray  # (T, G)
    n_known: np.ndarray  # (T,) tickets sold, NaN when unknown
    year_index: np.ndarray  # (T,)
    years: list[int]
    features: np.ndarray  # (T, n, D)
    draw_ids: np.ndarray
    dates: np.ndarray
    groups: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.y.shape[0])

    def subset(self, mask: np.ndarray) -> "CalibrationData":
        return CalibrationData(
            spec=self.spec,
            winning=self.winning[mask],
            special=self.special[mask],
            prev_special=self.prev_special[mask],
            y=self.y[mask],
            n_known=self.n_known[mask],
            year_index=self.year_index[mask],
            years=self.years,
            features=self.features[mask],
            draw_ids=self.draw_ids[mask],
            dates=self.dates[mask],
            groups=self.groups,
        )


def build_calibration_data(h: DrawHistory, ph: PrizeHistory, tickets_sold: np.ndarray | None = None) -> CalibrationData:
    spec = h.spec
    groups = observation_groups(spec)
    cols = {nm: i for i, nm in enumerate(ph.tier_names)}
    y_all = np.column_stack([np.sum([ph.winners[:, cols[t]] for t in g[1]], axis=0) for g in groups])
    ok = ~np.isnan(y_all).any(axis=1)
    feats = dynamic_features(h)
    years_all = h.dates.astype("datetime64[Y]").astype(int) + 1970
    years = sorted(set(int(y) for y in years_all[ok]))
    yidx = np.array([years.index(int(y)) if int(y) in years else 0 for y in years_all])
    sold = np.full(len(h), np.nan) if tickets_sold is None else np.asarray(tickets_sold, dtype=np.float64)
    prev_sp = np.concatenate([[0], h.bonus[:-1]]).astype(int) if spec.separate_special else np.zeros(len(h), int)
    sp = h.bonus.astype(int) if spec.separate_special else np.zeros(len(h), int)
    return CalibrationData(
        spec=spec,
        winning=h.numbers[ok].astype(int),
        special=sp[ok],
        prev_special=prev_sp[ok],
        y=y_all[ok],
        n_known=sold[ok],
        year_index=yidx[ok],
        years=years,
        features=feats[ok],
        draw_ids=h.draw_ids[ok],
        dates=h.dates[ok],
        groups=groups,
    )


# ------------------------------------------------------------------ model
@dataclass
class ParamLayout:
    n: int
    d: int
    b: int  # special-number pool (0 if none)
    years: int  # year scales for draws with known sales
    nu: int  # free log-scale per draw with unknown sales

    @property
    def size(self) -> int:
        return self.n + self.d + 1 + (self.b + 1 if self.b else 0) + self.years + self.nu

    def split(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        i = 0
        out = {}
        for name, size in (
            ("beta", self.n),
            ("gamma", self.d),
            ("a", 1),
            ("eta", self.b),
            ("gamma_s", 1 if self.b else 0),
            ("c", self.years),
            ("nu", self.nu),
        ):
            out[name] = theta[i : i + size]
            i += size
        return out


def nb_loglik(y: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return gammaln(y + phi) - gammaln(phi) - gammaln(y + 1) + phi * np.log(phi / (phi + mu)) + y * np.log(mu / (phi + mu))


def profile_log_scale(p: np.ndarray, y: np.ndarray, phi: np.ndarray, iters: int = 30) -> np.ndarray:
    """Per-draw MLE of ν in μ_g = e^ν·P_g under NB(φ_g) — vectorised Newton."""
    nu = np.log(np.maximum(y.sum(axis=1), 0.5) / p.sum(axis=1))
    for _ in range(iters):
        mu = np.exp(nu)[:, None] * p
        g = np.sum(phi * (y - mu) / (mu + phi), axis=1)
        h = -np.sum(phi * mu * (y + phi) / (mu + phi) ** 2, axis=1)
        step = g / np.where(h < 0, h, -1.0)
        nu = nu - np.clip(step, -2, 2)
        if np.max(np.abs(step)) < 1e-10:
            break
    return nu


@dataclass
class Penalty:
    """Gaussian priors (ridge penalties). ``beta_mean`` lets a game borrow strength from
    another game's calibration (same player population): β ~ N(β_prior, 1/beta)."""

    beta: float = 2.0
    gamma: float = 0.5
    eta: float = 1.0
    a_mean: float = float(logit(0.35))
    a_sd: float = 1.5
    beta_mean: np.ndarray | None = None
    gamma_mean: np.ndarray | None = None


def _r_and_dlogr(w: np.ndarray, winning: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """r (T, k+1) = P(m matches) under the product form; dlogr (T, k+1, n)."""
    t_count, n = w.shape
    rows = np.arange(t_count)[:, None]
    widx = winning - 1
    e_all = elementary_symmetric(w, k)  # (T, k+1)
    w_w = w[rows, widx]  # (T, k)
    e_w = elementary_symmetric(w_w, k)
    e_l = e_all.copy()
    for j in range(k):
        e_l = divide_out(e_l, w_w[:, j])
    ms = np.arange(k + 1)
    r = e_w[:, ms] * e_l[:, k - ms] / e_all[:, k : k + 1]
    # derivative of log e_k(all)
    h_all = divide_out(np.broadcast_to(e_all[:, None, :], (t_count, n, k + 1)), w)
    d_all = w * h_all[:, :, k - 1] / e_all[:, k : k + 1]  # (T, n)
    # members of L
    in_w = np.zeros((t_count, n), dtype=bool)
    in_w[rows, widx] = True
    h_l = divide_out(np.broadcast_to(e_l[:, None, :], (t_count, n, k + 1)), w)
    dlogr = np.zeros((t_count, k + 1, n))
    for m in range(k + 1):
        j = k - m
        if j >= 1:
            dlogr[:, m, :] = np.where(in_w, 0.0, w * h_l[:, :, j - 1] / e_l[:, j : j + 1])
    # members of W
    for pos in range(k):
        h_w = divide_out(e_w, w_w[:, pos])  # (T, k+1)
        col = widx[:, pos]
        for m in range(1, k + 1):
            dlogr[np.arange(t_count), m, col] += w_w[:, pos] * h_w[:, m - 1] / e_w[:, m]
    dlogr -= d_all[:, None, :]
    return r, dlogr


class BehaviourModel:
    def __init__(self, data: CalibrationData, penalty: Penalty | None = None, fit_q: bool = True) -> None:
        self.data = data
        self.spec = data.spec
        self.penalty = penalty or Penalty()
        self.fit_q = fit_q
        self.known = ~np.isnan(data.n_known)
        self.unknown_idx = np.flatnonzero(~self.known)
        self.layout = ParamLayout(
            n=self.spec.pool_size,
            d=data.features.shape[2],
            b=(self.spec.bonus_pool_size or 0) if self.spec.separate_special else 0,
            years=len(data.years) if self.known.any() else 0,
            nu=int((~self.known).sum()),
        )
        k = self.spec.pick
        self.h = np.array([self.spec.match_probability(m) for m in range(k + 1)])
        self.phi = np.full(len(data.groups), 1e6)  # NB dispersion per group (≈ Poisson initially)

    # -------------------------------------------------------------- core
    def _u(self, cond: Cond) -> float:
        if cond == "any":
            return 1.0
        b = float(self.spec.bonus_pool_size or 1)
        return 1.0 / b if cond == "hit" else 1.0 - 1.0 / b

    def initial_theta(self) -> np.ndarray:
        th = np.zeros(self.layout.size)
        parts = self.layout.split(th)
        parts["a"][:] = self.penalty.a_mean
        if self.penalty.beta_mean is not None:
            parts["beta"][:] = np.asarray(self.penalty.beta_mean) - np.mean(self.penalty.beta_mean)
        if self.penalty.gamma_mean is not None:
            parts["gamma"][:] = self.penalty.gamma_mean
        p, _ = self.probabilities(th, need_grad=False)
        if self.layout.years:
            for yi in range(self.layout.years):
                msk = self.known & (self.data.year_index == yi)
                if msk.any():
                    parts["c"][yi] = np.log(self.data.y[msk].sum() / np.sum(self.data.n_known[msk][:, None] * p[msk]))
        if self.layout.nu:
            u = self.unknown_idx
            parts["nu"][:] = profile_log_scale(p[u], self.data.y[u], self.phi[None, :])
        return th

    def scales(self, par: dict[str, np.ndarray]) -> np.ndarray:
        d = self.data
        sc = np.empty(d.size)
        if self.layout.years:
            sc[self.known] = d.n_known[self.known] * np.exp(par["c"][d.year_index[self.known]])
        if self.layout.nu:
            sc[self.unknown_idx] = np.exp(par["nu"])
        return sc

    def probabilities(self, theta: np.ndarray, need_grad: bool = True):
        """Group probabilities P (T, G) and the pieces needed for gradients."""
        par = self.layout.split(theta)
        d = self.data
        beta_t = par["beta"][None, :] + np.einsum("tnd,d->tn", d.features, par["gamma"])
        w = np.exp(beta_t - beta_t.mean(axis=1, keepdims=True))
        r, dlogr = _r_and_dlogr(w, d.winning, self.spec.pick) if need_grad else (_r_only(w, d.winning, self.spec.pick), None)
        q = float(expit(par["a"][0]))
        if self.layout.b:
            logits = np.broadcast_to(par["eta"], (d.size, self.layout.b)).copy()
            has_prev = d.prev_special > 0
            logits[np.where(has_prev)[0], d.prev_special[has_prev] - 1] += par["gamma_s"][0]
            pi = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            pi_s = pi[np.arange(d.size), d.special - 1]
        else:
            pi = None
            pi_s = np.ones(d.size)
        p = np.zeros((d.size, len(d.groups)))
        for g, (_, _, comps) in enumerate(d.groups):
            for m, cond in comps:
                v = pi_s if cond == "hit" else (1 - pi_s if cond == "miss" else np.ones(d.size))
                p[:, g] += q * self.h[m] * self._u(cond) + (1 - q) * r[:, m] * v
        return p, dict(r=r, dlogr=dlogr, q=q, pi=pi, pi_s=pi_s, par=par)

    def loglik(self, theta: np.ndarray, need_grad: bool = True, per_draw: bool = False):
        """NB log-likelihood; returns (total, gradient or per-draw scores, per-draw values)."""
        d = self.data
        p, aux = self.probabilities(theta, need_grad)
        p = np.maximum(p, 1e-300)
        par = aux["par"]
        sc = self.scales(par)
        mu = sc[:, None] * p
        phi = self.phi[None, :]
        ll_t = np.sum(nb_loglik(d.y, mu, phi), axis=1)
        if not need_grad:
            return float(ll_t.sum()), None, ll_t
        dmu = d.y / mu - (d.y + phi) / (mu + phi)
        score = dmu * sc[:, None]  # dℓ/dP
        dlogscale = np.sum(dmu * mu, axis=1)  # dℓ/dlog(scale)
        q, r, dlogr, pi, pi_s = aux["q"], aux["r"], aux["dlogr"], aux["pi"], aux["pi_s"]
        k = self.spec.pick
        coef = np.zeros((d.size, k + 1))
        da = np.zeros(d.size)
        dpis = np.zeros(d.size)
        for g, (_, _, comps) in enumerate(d.groups):
            for m, cond in comps:
                v = pi_s if cond == "hit" else (1 - pi_s if cond == "miss" else np.ones(d.size))
                coef[:, m] += score[:, g] * (1 - q) * v * r[:, m]
                da += score[:, g] * q * (1 - q) * (self.h[m] * self._u(cond) - r[:, m] * v)
                if cond == "hit":
                    dpis += score[:, g] * (1 - q) * r[:, m]
                elif cond == "miss":
                    dpis -= score[:, g] * (1 - q) * r[:, m]
        g_beta_t = np.einsum("tm,tmn->tn", coef, dlogr)
        blocks = [g_beta_t, np.einsum("tn,tnd->td", g_beta_t, d.features), (da if self.fit_q else 0 * da)[:, None]]
        if self.layout.b:
            onehot = np.zeros((d.size, self.layout.b))
            onehot[np.arange(d.size), d.special - 1] = 1.0
            deta = (dpis * pi_s)[:, None] * (onehot - pi)
            prev_oh = np.zeros_like(onehot)
            has_prev = d.prev_special > 0
            prev_oh[np.where(has_prev)[0], d.prev_special[has_prev] - 1] = 1.0
            blocks += [deta, np.sum(deta * prev_oh, axis=1, keepdims=True)]
        if self.layout.years:
            dc_t = np.zeros((d.size, self.layout.years))
            kn = np.where(self.known)[0]
            dc_t[kn, d.year_index[kn]] = dlogscale[kn]
            blocks.append(dc_t)
        if self.layout.nu:
            dnu = np.zeros((d.size, self.layout.nu))
            dnu[self.unknown_idx, np.arange(self.layout.nu)] = dlogscale[self.unknown_idx]
            blocks.append(dnu)
        per_t = np.concatenate(blocks, axis=1)
        if per_draw:
            return float(ll_t.sum()), per_t, ll_t
        return float(ll_t.sum()), per_t.sum(axis=0), ll_t

    def penalty_terms(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        par = self.layout.split(theta)
        pen = self.penalty
        b0 = np.zeros_like(par["beta"]) if pen.beta_mean is None else np.asarray(pen.beta_mean)
        g0 = np.zeros_like(par["gamma"]) if pen.gamma_mean is None else np.asarray(pen.gamma_mean)
        db = par["beta"] - par["beta"].mean() - (b0 - b0.mean())
        val = 0.5 * pen.beta * np.sum(db**2) + 0.5 * pen.gamma * np.sum((par["gamma"] - g0) ** 2)
        grad = np.zeros_like(theta)
        g = self.layout.split(grad)
        g["beta"][:] = pen.beta * (db - db.mean())
        g["gamma"][:] = pen.gamma * (par["gamma"] - g0)
        if self.fit_q:
            val += 0.5 * ((par["a"][0] - pen.a_mean) / pen.a_sd) ** 2
            g["a"][:] = (par["a"][0] - pen.a_mean) / pen.a_sd**2
        if self.layout.b:
            val += 0.5 * pen.eta * (np.sum(par["eta"] ** 2) + np.sum(par["gamma_s"] ** 2))
            g["eta"][:] = pen.eta * par["eta"]
            g["gamma_s"][:] = pen.eta * par["gamma_s"]
        return float(val), grad

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        ll, grad, _ = self.loglik(theta)
        pv, pg = self.penalty_terms(theta)
        return -(ll - pv), -(grad - pg)

    def update_dispersion(self, theta: np.ndarray) -> None:
        """Method-of-moments NB dispersion per tier group, Var = μ + μ²/φ.

        Draws with a free scale ν_t lose one degree of freedom; residuals of those draws
        are inflated by G/(G−1) to compensate."""
        p, aux = self.probabilities(theta, need_grad=False)
        mu = self.scales(aux["par"])[:, None] * p
        y = self.data.y
        g_count = y.shape[1]
        infl = np.where(self.known, 1.0, g_count / max(g_count - 1, 1))[:, None]
        excess = np.sum((y - mu) ** 2 * infl - mu, axis=0)
        phi = np.where(excess > 0, np.sum(mu**2, axis=0) / np.maximum(excess, 1e-12), 1e6)
        self.phi = np.clip(phi, 1.0, 1e6)

    def fit(self, rounds: int = 4, maxiter: int = 500) -> np.ndarray:
        theta = self.initial_theta()
        res = None
        for r in range(rounds):
            res = optimize.minimize(self.objective, theta, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
            theta = res.x
            old = self.phi.copy()
            self.update_dispersion(theta)
            log.info("calibration round %d: obj=%.2f, phi=%s, converged=%s", r, res.fun, np.round(self.phi, 1), res.success)
            if np.allclose(old, self.phi, rtol=0.05):
                break
        if res is not None and theta.size:
            res = optimize.minimize(self.objective, theta, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
            theta = res.x
        self.result = res
        return theta

    # ------------------------------------------------------------ inference
    def sandwich(self, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Robust covariance H⁻¹ (Σ_t s_t s_tᵀ) H⁻¹ with draws as independent units.

        The Hessian is formed by finite differences of the analytic gradient for the
        structural parameters; per-draw scale parameters ν_t are profiled out (Schur
        complement), which keeps the cost independent of the number of draws."""
        lay = self.layout
        n_struct = lay.size - lay.nu
        _, per_t, _ = self.loglik(theta, per_draw=True)
        hess = np.zeros((lay.size, lay.size))
        for i in range(n_struct):
            e = np.zeros(lay.size)
            e[i] = eps
            hess[i] = (self.objective(theta + e)[1] - self.objective(theta - e)[1]) / (2 * eps)
        if lay.nu:
            # ν-block is diagonal (each ν_t only enters its own draw): exact NB curvature
            par = lay.split(theta)
            p, _ = self.probabilities(theta, need_grad=False)
            u = self.unknown_idx
            mu = np.exp(par["nu"])[:, None] * p[u]
            y = self.data.y[u]
            phi = self.phi[None, :]
            diag = np.sum(phi * mu * (y + phi) / (mu + phi) ** 2, axis=1)
            off = n_struct + np.arange(lay.nu)
            hess[off, off] = diag
            hess[np.ix_(off, np.arange(n_struct))] = hess[np.ix_(np.arange(n_struct), off)].T  # fill ν rows from struct rows
        hess = 0.5 * (hess + hess.T)
        if lay.nu:
            a_ = hess[:n_struct, :n_struct]
            b_ = hess[:n_struct, n_struct:]
            dinv = 1.0 / np.maximum(np.diag(hess)[n_struct:], 1e-12)
            h_eff = a_ - (b_ * dinv) @ b_.T
            s_struct = per_t[:, :n_struct] - (per_t[:, n_struct:] * dinv) @ b_.T
            hinv = np.linalg.pinv(h_eff)
            cov_s = hinv @ (s_struct.T @ s_struct) @ hinv
            cov = np.zeros((lay.size, lay.size))
            cov[:n_struct, :n_struct] = cov_s
            return cov
        hinv = np.linalg.pinv(hess)
        return hinv @ (per_t.T @ per_t) @ hinv

    def null_theta(self) -> np.ndarray:
        """Uniform crowd (q → 1) with scales re-profiled for that model."""
        th = np.zeros(self.layout.size)
        par = self.layout.split(th)
        par["a"][:] = 30.0
        p, _ = self.probabilities(th, need_grad=False)
        for yi in range(self.layout.years):
            msk = self.known & (self.data.year_index == yi)
            if msk.any():
                par["c"][yi] = np.log(self.data.y[msk].sum() / np.sum(self.data.n_known[msk][:, None] * p[msk]))
        if self.layout.nu:
            u = self.unknown_idx
            par["nu"][:] = profile_log_scale(p[u], self.data.y[u], self.phi[None, :])
        return th

    def fitted_scale(self, theta: np.ndarray) -> np.ndarray:
        """Tickets sold implied by the fit (known N × year scale, or e^ν)."""
        return self.scales(self.layout.split(theta))


def _r_only(w: np.ndarray, winning: np.ndarray, k: int) -> np.ndarray:
    rows = np.arange(w.shape[0])[:, None]
    e_all = elementary_symmetric(w, k)
    w_w = w[rows, winning - 1]
    e_w = elementary_symmetric(w_w, k)
    e_l = e_all.copy()
    for j in range(k):
        e_l = divide_out(e_l, w_w[:, j])
    ms = np.arange(k + 1)
    return e_w[:, ms] * e_l[:, k - ms] / e_all[:, k : k + 1]


# ----------------------------------------------------------------- results
class Coefficient(BaseModel):
    name: str
    estimate: float
    std_error: float
    z: float
    p_value: float


class FitStats(BaseModel):
    draws: int
    loglik: float
    loglik_null: float
    lr_statistic: float
    df: int
    lr_p_value: float
    deviance_explained: float  # share of the null-model excess deviance removed
    holdout_draws: int | None = None
    holdout_loglik_gain_per_draw: float | None = None
    holdout_lr_p_value: float | None = None
    tier3_r2: float | None = None  # variance of log(observed / expected-under-uniform) explained


class BehaviourCalibration(BaseModel):
    game: str
    trained_on: str
    draws_used: int
    prior: str | None = None
    tickets_known: bool
    quick_pick_share: float
    quick_pick_share_se: float
    beta: list[float]  # static per-number log-weights (number 1..n), centred
    beta_se: list[float]
    gamma: dict[str, float]
    gamma_se: dict[str, float]
    special_probs: list[float] | None = None
    special_last_effect: float | None = None
    dispersion: list[float]
    year_scale: dict[str, float] = {}
    feature_effects: list[Coefficient]
    fit: FitStats
    popularity_spread: dict[str, float]  # distribution of model popularity ratios over random tickets
    most_popular_numbers: list[int]
    least_popular_numbers: list[int]

    # ----------------------------------------------------------- usage
    def popularity_model(self, history: DrawHistory | None = None, pattern_priors: bool = True, seed: int = 7) -> PopularityModel:
        """Popularity model for the *next* draw (dynamic effects from ``history``)."""
        spec = get_game(self.game)
        beta = np.asarray(self.beta, dtype=np.float64).copy()
        if history is not None and len(history):
            x = next_draw_features(history)
            beta = beta + x @ np.array([self.gamma[f] for f in DYNAMIC_FEATURES])
        params = PopularityParams(quick_pick_share=self.quick_pick_share)
        if not pattern_priors:
            params = PopularityParams(
                quick_pick_share=self.quick_pick_share, arithmetic_progression=0.0, slip_line=0.0, last_draw_exact=0.0, long_run=0.0, all_birthday=0.0, consecutive_pair=0.0
            )
        last = tuple(int(v) for v in history.numbers[-1]) if history is not None and len(history) else None
        special = None
        if self.special_probs is not None:
            special = np.asarray(self.special_probs)
            if history is not None and len(history) and self.special_last_effect:
                lg = np.log(special)
                lg[int(history.bonus[-1]) - 1] += self.special_last_effect
                special = np.exp(lg - logsumexp(lg))
        return PopularityModel(spec, params, last_draw=last, beta=beta, seed=seed, special_probs=special)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BehaviourCalibration":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _coef(name: str, est: float, se: float) -> Coefficient:
    z = est / se if se > 0 else 0.0
    return Coefficient(name=name, estimate=float(est), std_error=float(se), z=float(z), p_value=float(2 * stats.norm.sf(abs(z))))


def _holdout_gain(model_train: BehaviourModel, theta: np.ndarray, test: CalibrationData) -> tuple[float, float]:
    """Out-of-sample log-likelihood gain per draw vs the uniform crowd.

    Structural parameters come from the training fit; nuisance scales (year scales for
    known sales, ν_t per draw otherwise) are re-profiled on the test draws for *both*
    models, so only the behavioural part is being compared."""

    def ll_profiled(th: np.ndarray) -> np.ndarray:
        m = BehaviourModel(test, model_train.penalty, model_train.fit_q)
        m.phi = model_train.phi
        th2 = np.zeros(m.layout.size)
        lay_tr, lay_te = model_train.layout.split(th), m.layout.split(th2)
        for key in ("beta", "gamma", "a", "eta", "gamma_s"):
            lay_te[key][:] = lay_tr[key]
        p, _ = m.probabilities(th2, need_grad=False)
        for yi in range(m.layout.years):
            msk = m.known & (test.year_index == yi)
            if msk.any():
                lay_te["c"][yi] = np.log(test.y[msk].sum() / np.sum(test.n_known[msk][:, None] * p[msk]))
        if m.layout.nu:
            lay_te["nu"][:] = profile_log_scale(p[m.unknown_idx], test.y[m.unknown_idx], m.phi[None, :])
        return m.loglik(th2, need_grad=False)[2]

    gain = ll_profiled(theta) - ll_profiled(model_train.null_theta())
    se = gain.std(ddof=1) / np.sqrt(gain.size)
    p = float(stats.norm.sf(gain.mean() / se)) if se > 0 else 1.0
    return float(gain.mean()), p


def prior_from(other: "BehaviourCalibration", spec: GameSpec, strength: float = 8.0) -> Penalty:
    """Penalty centred on another game's calibration (numbers 1..n of the other game).

    Used for Mega 6/45, whose 161 draws with prize data carry little information alone:
    its players are the same population as Power 6/55's, so Power's per-number weights
    (restricted to 1–45) and quick-pick share are the prior mean."""
    b = np.asarray(other.beta[: spec.pool_size], dtype=np.float64)
    return Penalty(
        beta=strength,
        gamma=4.0,
        a_mean=float(logit(np.clip(other.quick_pick_share, 0.02, 0.98))),
        a_sd=0.5,
        beta_mean=b - b.mean(),
        gamma_mean=np.array([other.gamma[f] for f in DYNAMIC_FEATURES]),
    )


def calibrate(
    h: DrawHistory,
    ph: PrizeHistory,
    tickets_sold: np.ndarray | None = None,
    penalty: Penalty | None = None,
    holdout_fraction: float = 0.2,
    fit_q: bool = True,
    compute_se: bool = True,
    prior_source: str | None = None,
) -> BehaviourCalibration:
    """Fit on all draws with prize data; report a chronological hold-out evaluation."""
    data = build_calibration_data(h, ph, tickets_sold)
    if data.size < 50:
        raise ValueError(f"only {data.size} draws with prize data; need ≥ 50")
    spec = h.spec
    model = BehaviourModel(data, penalty, fit_q)
    theta = model.fit()
    par = model.layout.split(theta)
    ll, _, _ = model.loglik(theta, need_grad=False)
    ll0 = model.loglik(model.null_theta(), need_grad=False)[0]
    df = model.layout.n - 1 + model.layout.d + (model.layout.b + 0 if model.layout.b else 0) + (1 if fit_q else 0)
    lr = max(0.0, 2 * (ll - ll0))

    # hold-out: refit on the first (1 − f) of draws, score the rest
    holdout_draws = holdout_gain = holdout_p = None
    if holdout_fraction > 0:
        cut = int(data.size * (1 - holdout_fraction))
        tr = np.zeros(data.size, bool)
        tr[:cut] = True
        m_tr = BehaviourModel(data.subset(tr), penalty, fit_q)
        th_tr = m_tr.fit()
        holdout_gain, holdout_p = _holdout_gain(m_tr, th_tr, data.subset(~tr))
        holdout_draws = int((~tr).sum())

    cov = model.sandwich(theta) if compute_se else np.zeros((theta.size, theta.size))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    se_par = model.layout.split(se)
    q = float(expit(par["a"][0]))
    q_se = float(q * (1 - q) * se_par["a"][0])

    beta = par["beta"] - par["beta"].mean()
    # feature regression of β on interpretable static features (WLS, sandwich-propagated SEs)
    xf = np.column_stack([np.ones(spec.pool_size), static_feature_matrix(spec.pool_size)])
    keep = xf[:, 1:].std(axis=0) > 0
    xf = np.column_stack([xf[:, 0], xf[:, 1:][:, keep]])
    names = [nm for nm, kp in zip(STATIC_FEATURES, keep) if kp]
    cov_b = cov[: spec.pool_size, : spec.pool_size]
    proj = np.linalg.pinv(xf.T @ xf) @ xf.T
    coef = proj @ beta
    cov_c = proj @ cov_b @ proj.T
    feat = [_coef(nm, coef[i + 1], np.sqrt(max(cov_c[i + 1, i + 1], 0))) for i, nm in enumerate(names)]

    # tier-3 R² (known-N games): how much of the winner-count surprise the model explains
    r2 = None
    if model.known.any():
        g3 = [i for i, g in enumerate(data.groups) if g[0] in ("3", "third")][0]
        p, aux = model.probabilities(theta, need_grad=False)
        scale = data.n_known * np.exp(par["c"][data.year_index]) if model.layout.years else data.n_known
        m_ = model.known
        obs = np.log(np.maximum(data.y[m_, g3], 1) / (scale[m_] * model.h[3]))
        pred = np.log(p[m_, g3] / model.h[3])
        r2 = float(1 - np.var(obs - pred) / np.var(obs))

    # popularity spread over random tickets for the next draw
    cal_partial = dict(game=spec.code.value, quick_pick_share=q, beta=beta.tolist(), gamma={f: float(v) for f, v in zip(DYNAMIC_FEATURES, par["gamma"])})
    pop = PopularityModel(spec, PopularityParams(quick_pick_share=q), beta=beta + next_draw_features(h) @ par["gamma"], last_draw=None)
    rng = np.random.default_rng(0)
    sample = np.sort(np.argpartition(rng.random((20_000, spec.pool_size)), spec.pick, axis=1)[:, : spec.pick] + 1, axis=1)
    ratios = pop.popularity_ratio(sample)
    spread = {f"p{int(qq * 100):02d}": float(np.quantile(ratios, qq)) for qq in (0.01, 0.1, 0.5, 0.9, 0.99)}
    _ = cal_partial

    special_probs = special_last = None
    if model.layout.b:
        eta = par["eta"]
        special_probs = np.exp(eta - logsumexp(eta)).tolist()
        special_last = float(par["gamma_s"][0])

    return BehaviourCalibration(
        game=spec.code.value,
        trained_on=f"{str(data.dates[0])} → {str(data.dates[-1])}",
        draws_used=data.size,
        prior=prior_source,
        tickets_known=bool(model.known.any()),
        quick_pick_share=q,
        quick_pick_share_se=q_se,
        beta=beta.tolist(),
        beta_se=se_par["beta"].tolist(),
        gamma={f: float(v) for f, v in zip(DYNAMIC_FEATURES, par["gamma"])},
        gamma_se={f: float(v) for f, v in zip(DYNAMIC_FEATURES, se_par["gamma"])},
        special_probs=special_probs,
        special_last_effect=special_last,
        dispersion=model.phi.tolist(),
        year_scale={str(y): float(c) for y, c in zip(data.years, par["c"])} if model.layout.years else {},
        feature_effects=feat,
        fit=FitStats(
            draws=data.size,
            loglik=ll,
            loglik_null=ll0,
            lr_statistic=lr,
            df=df,
            lr_p_value=float(stats.chi2.sf(lr, df)),
            deviance_explained=float((ll - ll0) / max(abs(ll0), 1e-9)),
            holdout_draws=holdout_draws,
            holdout_loglik_gain_per_draw=holdout_gain,
            holdout_lr_p_value=holdout_p,
            tier3_r2=r2,
        ),
        popularity_spread=spread,
        most_popular_numbers=[int(i + 1) for i in np.argsort(-beta)[:8]],
        least_popular_numbers=[int(i + 1) for i in np.argsort(beta)[:8]],
    )


def load_calibration(game: str | GameCode, directory: Path) -> BehaviourCalibration | None:
    path = Path(directory) / f"behaviour_{get_game(game).code.value}.json"
    if not path.exists():
        return None
    try:
        return BehaviourCalibration.load(path)
    except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupted file
        log.warning("ignoring unreadable calibration %s: %s", path, exc)
        return None


# --------------------------------------------------------------- validation
class LevelCheck(BaseModel):
    """Does the model's predicted popularity of each winning set explain *how many*
    third-prize winners there were, after removing the smooth sales trend?"""

    game: str
    draws: int
    correlation: float
    permutation_p_value: float
    slope: float  # regression of detrended log winners on predicted log popularity (1 = perfect scale)
    method: str


def level_check(cal: BehaviourCalibration, h: DrawHistory, ph: PrizeHistory, window: int = 7, permutations: int = 5000, seed: int = 0) -> LevelCheck:
    """Non-parametric validation that does not need ticket sales.

    Sales move smoothly from draw to draw (jackpot size, weekday) while the winning set
    is random, so log y3_t minus a rolling median of its neighbours (self excluded)
    isolates the popularity effect. Under a uniform crowd its correlation with the
    model's predicted log(P3_t/H3) is zero.
    """
    spec = h.spec
    data = build_calibration_data(h, ph, None)
    g3 = [i for i, g in enumerate(data.groups) if g[0] in ("3", "third", "fifth")][0]
    model = BehaviourModel(data)
    th = np.zeros(model.layout.size)
    par = model.layout.split(th)
    par["beta"][:] = np.asarray(cal.beta)
    par["gamma"][:] = [cal.gamma[f] for f in DYNAMIC_FEATURES]
    par["a"][:] = logit(np.clip(cal.quick_pick_share, 1e-6, 1 - 1e-6))
    if model.layout.b and cal.special_probs is not None:
        par["eta"][:] = np.log(cal.special_probs)
        par["gamma_s"][:] = cal.special_last_effect or 0.0
    p, _ = model.probabilities(th, need_grad=False)
    base = sum(model.h[m] * model._u(c) for m, c in data.groups[g3][2])
    pred = np.log(p[:, g3] / base)
    ly = np.log(np.maximum(data.y[:, g3], 1.0))
    half = window // 2
    trend = np.array([np.median(np.concatenate([ly[max(0, t - half) : t], ly[t + 1 : t + 1 + half]])) for t in range(len(ly))])
    resid = ly - trend
    ok = np.isfinite(resid)
    x, yv = pred[ok], resid[ok]
    corr = float(np.corrcoef(x, yv)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rng.permutation(x), yv)[0, 1] for _ in range(permutations)])
    slope = float(np.polyfit(x, yv, 1)[0])
    return LevelCheck(
        game=spec.code.value,
        draws=int(ok.sum()),
        correlation=corr,
        permutation_p_value=float((np.sum(perm >= corr) + 1) / (permutations + 1)),
        slope=slope,
        method=f"log third-tier winners minus rolling median of {window - 1} neighbours vs predicted log popularity",
    )


def transfer_calibration(source: BehaviourCalibration, h: DrawHistory, ph: PrizeHistory) -> BehaviourCalibration:
    """Apply another game's calibrated crowd to this game (numbers 1..n) and evaluate it.

    Nothing is fitted on the target game except the per-draw sales scales ν_t, so the
    reported likelihood gain over a uniform crowd is an honest out-of-sample test."""
    spec = h.spec
    data = build_calibration_data(h, ph, None)
    b = np.asarray(source.beta[: spec.pool_size], dtype=np.float64)
    b = b - b.mean()
    model = BehaviourModel(data)
    # dispersion borrowed by match count from the source game (same crowd, same syndicates)
    src_groups = observation_groups(get_game(source.game))
    by_m = {comps[0][0]: phi for (_, _, comps), phi in zip(src_groups, source.dispersion)}
    model.phi = np.array([by_m.get(comps[0][0], 1e6) for _, _, comps in data.groups])
    th = np.zeros(model.layout.size)
    par = model.layout.split(th)
    par["beta"][:] = b
    par["gamma"][:] = [source.gamma[f] for f in DYNAMIC_FEATURES]
    par["a"][:] = logit(np.clip(source.quick_pick_share, 1e-6, 1 - 1e-6))
    p, _ = model.probabilities(th, need_grad=False)
    par["nu"][:] = profile_log_scale(p, data.y, model.phi[None, :])
    ll_t = model.loglik(th, need_grad=False)[2]
    null = model.null_theta()
    ll0_t = model.loglik(null, need_grad=False)[2]
    gain = ll_t - ll0_t
    se = gain.std(ddof=1) / np.sqrt(gain.size)
    pop = PopularityModel(spec, PopularityParams(quick_pick_share=source.quick_pick_share), beta=b + next_draw_features(h) @ par["gamma"])
    rng = np.random.default_rng(0)
    sample = np.sort(np.argpartition(rng.random((20_000, spec.pool_size)), spec.pick, axis=1)[:, : spec.pick] + 1, axis=1)
    ratios = pop.popularity_ratio(sample)
    return BehaviourCalibration(
        game=spec.code.value,
        trained_on=f"transferred from {source.game} ({source.trained_on}); evaluated on {str(data.dates[0])} → {str(data.dates[-1])}",
        draws_used=data.size,
        prior=f"transfer:{source.game}",
        tickets_known=False,
        quick_pick_share=source.quick_pick_share,
        quick_pick_share_se=source.quick_pick_share_se,
        beta=b.tolist(),
        beta_se=source.beta_se[: spec.pool_size],
        gamma=dict(source.gamma),
        gamma_se=dict(source.gamma_se),
        dispersion=model.phi.tolist(),
        feature_effects=source.feature_effects,
        fit=FitStats(
            draws=data.size,
            loglik=float(ll_t.sum()),
            loglik_null=float(ll0_t.sum()),
            lr_statistic=float(max(0.0, 2 * gain.sum())),
            df=0,
            lr_p_value=float(stats.norm.sf(gain.mean() / se)) if se > 0 else 1.0,
            deviance_explained=float(gain.sum() / max(abs(ll0_t.sum()), 1e-9)),
            holdout_draws=data.size,
            holdout_loglik_gain_per_draw=float(gain.mean()),
            holdout_lr_p_value=float(stats.norm.sf(gain.mean() / se)) if se > 0 else 1.0,
        ),
        popularity_spread={f"p{int(qq * 100):02d}": float(np.quantile(ratios, qq)) for qq in (0.01, 0.1, 0.5, 0.9, 0.99)},
        most_popular_numbers=[int(i + 1) for i in np.argsort(-b)[:8]],
        least_popular_numbers=[int(i + 1) for i in np.argsort(b)[:8]],
    )


def fitted_tickets_sold(cal: BehaviourCalibration, h: DrawHistory, ph: PrizeHistory) -> np.ndarray:
    """N̂_t (aligned with ``h``; NaN without prize data) from winner counts and the crowd model.

    Uses E[y_g] = N·P_g(W_t): the NB-weighted profile estimate of N for each draw."""
    data = build_calibration_data(h, ph, None)
    model = BehaviourModel(data)
    model.phi = np.asarray(cal.dispersion) if len(cal.dispersion) == len(data.groups) else np.full(len(data.groups), 20.0)
    th = np.zeros(model.layout.size)
    par = model.layout.split(th)
    par["beta"][:] = cal.beta
    par["gamma"][:] = [cal.gamma[f] for f in DYNAMIC_FEATURES]
    par["a"][:] = logit(np.clip(cal.quick_pick_share, 1e-6, 1 - 1e-6))
    if model.layout.b and cal.special_probs is not None:
        par["eta"][:] = np.log(cal.special_probs)
        par["gamma_s"][:] = cal.special_last_effect or 0.0
    p, _ = model.probabilities(th, need_grad=False)
    nu = profile_log_scale(p, data.y, model.phi[None, :])
    out = np.full(len(h), np.nan)
    pos = {int(d): i for i, d in enumerate(h.draw_ids)}
    for did, v in zip(data.draw_ids, nu):
        out[pos[int(did)]] = float(np.exp(v))
    return out
