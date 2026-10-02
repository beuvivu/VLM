"""Decision layer on top of the EV engine.

1. ``payout_distribution`` — the full distribution of one ticket's after-tax payout
   (fixed tiers plus the jackpot split K ~ Poisson(λ) among co-winners), not just its mean.
2. ``kelly`` — growth-optimal staking. With outcomes X_o (per 1 VND staked) the expected
   log-growth of betting a fraction f of bankroll is G(f) = Σ p_o log(1 + f(X_o − 1)).
   G'(0) = RTP − 1, so the optimal stake is exactly zero whenever RTP ≤ 1; when RTP > 1
   the optimum is found numerically and converted into "bankroll needed to justify one
   ticket".
3. ``ev_uncertainty`` — the EV depends on unobserved crowd behaviour and sales, so the
   point estimate is replaced by a distribution: parameters are drawn from explicit
   priors, the EV is recomputed for each draw, and the result reports quantiles,
   P(RTP > 1) and a Spearman sensitivity ranking of the inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats

from vietlott_engine.core.games import DEFAULT_TAX, GameSpec, TaxRule
from vietlott_engine.game_theory.ev import EVCalculator
from vietlott_engine.game_theory.popularity import PopularityModel, PopularityParams


# ------------------------------------------------------------- distribution
def payout_distribution(
    calc: EVCalculator, ticket: list[int] | tuple[int, ...], jackpot1: float | None, jackpot2: float | None, tickets_sold: int | None, special: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(values, probabilities) of the after-tax payout of one play, including 0."""
    spec, tax = calc.spec, calc.tax
    combo = calc._combo(ticket)
    j1, j2 = calc._jackpots(jackpot1, jackpot2)
    tickets_sold, _ = calc.resolve_sold(tickets_sold, j1)
    p_c = calc.ticket_probability(combo, calc._special(special))
    probs = spec.tier_probabilities
    vals: list[float] = []
    ps: list[float] = []
    for t in spec.tiers:
        if t.fixed_amount is not None:
            vals.append(tax.after_tax(t.fixed_amount))
            ps.append(probs[t.name])
            continue
        pot = j1 if t.name == "jackpot1" else float(j2 or 0.0)
        lam = tickets_sold * (p_c if t.name == "jackpot1" else p_c + 5 / spec.main_combinations)
        kmax = int(stats.poisson.ppf(1 - 1e-10, lam)) + 2 if lam > 0 else 0
        ks = np.arange(kmax + 1)
        pmf = stats.poisson.pmf(ks, lam) if lam > 0 else np.array([1.0])
        pmf /= pmf.sum()
        for kk, pk in zip(ks, pmf):
            vals.append(tax.after_tax(pot / (1 + kk)))
            ps.append(probs[t.name] * pk)
    vals.append(0.0)
    ps.append(1.0 - sum(ps))
    return np.asarray(vals), np.asarray(ps)


class PayoutProfile(BaseModel):
    expected_value: float
    standard_deviation: float
    coefficient_of_variation: float
    p_any_prize: float
    p_at_least_price: float
    p_at_least_100x: float
    median_payout: float


def payout_profile(values: np.ndarray, probs: np.ndarray, price: float) -> PayoutProfile:
    mean = float(np.sum(values * probs))
    sd = float(np.sqrt(max(np.sum(probs * (values - mean) ** 2), 0.0)))
    order = np.argsort(values)
    cdf = np.cumsum(probs[order])
    return PayoutProfile(
        expected_value=mean,
        standard_deviation=sd,
        coefficient_of_variation=sd / mean if mean > 0 else float("inf"),
        p_any_prize=float(probs[values > 0].sum()),
        p_at_least_price=float(probs[values >= price].sum()),
        p_at_least_100x=float(probs[values >= 100 * price].sum()),
        median_payout=float(values[order][np.searchsorted(cdf, 0.5)]),
    )


# -------------------------------------------------------------------- Kelly
class KellyResult(BaseModel):
    return_to_player: float
    optimal_fraction: float  # of bankroll per ticket
    growth_per_ticket_at_optimum: float  # expected log-growth (nats)
    tickets_for_bankroll: int
    bankroll: float
    min_bankroll_for_one_ticket: float | None
    note: str


def kelly(values: np.ndarray, probs: np.ndarray, price: float, bankroll: float) -> KellyResult:
    x = values / price  # gross return per 1 VND staked
    rtp = float(np.sum(probs * x))

    def neg_growth(f: float) -> float:
        return -float(np.sum(probs * np.log1p(f * (x - 1))))

    if rtp <= 1.0:
        return KellyResult(
            return_to_player=rtp,
            optimal_fraction=0.0,
            growth_per_ticket_at_optimum=0.0,
            tickets_for_bankroll=0,
            bankroll=bankroll,
            min_bankroll_for_one_ticket=None,
            note="RTP ≤ 1: every positive stake has negative expected log-growth; the growth-optimal bet is zero.",
        )
    res = optimize.minimize_scalar(neg_growth, bounds=(0.0, 1.0 - 1e-12), method="bounded", options={"xatol": 1e-15})
    f = float(res.x)
    return KellyResult(
        return_to_player=rtp,
        optimal_fraction=f,
        growth_per_ticket_at_optimum=-float(res.fun),
        tickets_for_bankroll=int(np.floor(f * bankroll / price)),
        bankroll=bankroll,
        min_bankroll_for_one_ticket=price / f if f > 0 else None,
        note=(
            "Positive expectation, but the payoff is so skewed that the growth-optimal stake is a tiny fraction of "
            "bankroll; buying more tickets than 'tickets_for_bankroll' lowers long-run growth."
        ),
    )


# -------------------------------------------------------------- uncertainty
class EVUncertainty(BaseModel):
    game: str
    ticket: list[int]
    simulations: int
    rtp_mean: float
    rtp_p05: float
    rtp_median: float
    rtp_p95: float
    prob_rtp_above_one: float
    co_winners_median: float
    sensitivity: dict[str, float]  # Spearman correlation of each input with RTP
    priors: dict[str, str]


PRIORS = {
    "quick_pick_share": "Uniform(0.15, 0.65)",
    "birthday_bonus": "Uniform(0.10, 0.60)",
    "month_bonus": "Uniform(0.00, 0.30)",
    "pattern_scale": "Uniform(0.5, 1.5) × pattern log-factors",
    "last_draw_bonus": "Uniform(0.00, 0.30)",
    "tickets_sold": "LogUniform(lo, hi)",
}


def ev_uncertainty(
    spec: GameSpec,
    ticket: list[int] | tuple[int, ...],
    jackpot1: float,
    tickets_sold_range: tuple[int, int],
    jackpot2: float | None = None,
    last_draw: tuple[int, ...] | None = None,
    sims: int = 300,
    seed: int | None = 0,
    tax: TaxRule = DEFAULT_TAX,
    special: int | None = None,
    calibration=None,
) -> EVUncertainty:
    """EV as a distribution over what is not known about the crowd and the sales.

    Without a calibration the crowd parameters are drawn from wide literature priors.
    With a ``BehaviourCalibration`` they are drawn from its sampling distribution
    (q ~ N(q̂, se), β_i ~ N(β̂_i, se_i)); only the uncalibrated combination-pattern
    factors keep a wide prior."""
    rng = np.random.default_rng(seed)
    base = PopularityParams()
    lo, hi = tickets_sold_range
    rows = []
    for _ in range(sims):
        scale = rng.uniform(0.5, 1.5)
        sold = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if calibration is None:
            params = replace(
                base,
                quick_pick_share=rng.uniform(0.15, 0.65),
                birthday_bonus=rng.uniform(0.10, 0.60),
                month_bonus=rng.uniform(0.0, 0.30),
                last_draw_bonus=rng.uniform(0.0, 0.30),
                arithmetic_progression=base.arithmetic_progression * scale,
                slip_line=base.slip_line * scale,
                long_run=base.long_run * scale,
                last_draw_exact=base.last_draw_exact * scale,
                normaliser_samples=8000,
            )
            model = PopularityModel(spec, params, last_draw=last_draw, seed=int(rng.integers(1 << 31)))
            drawn = (params.quick_pick_share, params.birthday_bonus, params.month_bonus, scale, params.last_draw_bonus, sold)
        else:
            q = float(np.clip(rng.normal(calibration.quick_pick_share, calibration.quick_pick_share_se), 0.02, 0.98))
            beta = np.asarray(calibration.beta) + rng.normal(0, 1, spec.pool_size) * np.asarray(calibration.beta_se[: spec.pool_size])
            params = replace(
                base,
                quick_pick_share=q,
                arithmetic_progression=base.arithmetic_progression * scale,
                slip_line=base.slip_line * scale,
                long_run=base.long_run * scale,
                last_draw_exact=base.last_draw_exact * scale,
                normaliser_samples=8000,
            )
            sp = np.asarray(calibration.special_probs) if calibration.special_probs is not None else None
            model = PopularityModel(spec, params, last_draw=last_draw, beta=beta, seed=int(rng.integers(1 << 31)), special_probs=sp)
            bday = float(np.mean(beta[: min(31, spec.pool_size)]) - np.mean(beta[min(31, spec.pool_size) :])) if spec.pool_size > 31 else 0.0
            drawn = (q, bday, 0.0, scale, 0.0, sold)
        calc = EVCalculator(spec, model, tax)
        b = calc.evaluate(ticket, jackpot1, jackpot2, int(sold), special=special)
        rows.append((b.return_to_player, b.expected_co_winners) + drawn)
    arr = np.asarray(rows)
    rtp = arr[:, 0]
    names = ["quick_pick_share", "birthday_bonus", "month_bonus", "pattern_scale", "last_draw_bonus", "tickets_sold"]
    sens = {}
    for i, nm in enumerate(names):
        col = arr[:, 2 + i]
        sens[nm] = float(stats.spearmanr(col, rtp)[0]) if np.std(col) > 0 else 0.0
    priors = dict(PRIORS)
    priors["tickets_sold"] = f"LogUniform({lo:,}, {hi:,})"
    if calibration is not None:
        priors.update(
            quick_pick_share=f"N({calibration.quick_pick_share:.3f}, {calibration.quick_pick_share_se:.3f}) — calibrated",
            birthday_bonus="per-number β ~ N(β̂, se) — calibrated (column = mean β 1–31 minus mean β 32+)",
            month_bonus="(in β)",
            last_draw_bonus="(in β via dynamic effects)",
        )
    return EVUncertainty(
        game=spec.code.value,
        ticket=sorted(int(x) for x in ticket),
        simulations=sims,
        rtp_mean=float(rtp.mean()),
        rtp_p05=float(np.quantile(rtp, 0.05)),
        rtp_median=float(np.median(rtp)),
        rtp_p95=float(np.quantile(rtp, 0.95)),
        prob_rtp_above_one=float(np.mean(rtp > 1.0)),
        co_winners_median=float(np.median(arr[:, 1])),
        sensitivity=dict(sorted(sens.items(), key=lambda kv: -abs(kv[1]))),
        priors=priors,
    )
