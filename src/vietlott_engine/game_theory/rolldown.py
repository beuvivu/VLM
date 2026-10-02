"""Lotto 5/35 "chia giải Độc đắc" (jackpot rolldown) — the one structural situation in
Vietlott where the *whole* jackpot must be paid out to ordinary prize tiers.

Rule: when the jackpot exceeds 12 tỷ with no winner, the 21:00 draw of the following
day is a rolldown draw: unless someone wins the jackpot in that draw, 1/3 of it goes to
the first-prize tier (5 main numbers) and 1/6 to each of tiers 2–5, shared equally among
the winning tickets of each tier on top of the fixed prize. A tier with no winner has
its share redistributed equally among the other tiers.

Expected value per ticket in a rolldown draw (N other tickets, tier probabilities P_τ):

    EV = Σ_τ P_τ · E[ net( fixed_τ + J·f_τ / (1 + K_τ) ) ],   K_τ ~ Poisson(N·P_τ)

With many winners E[1/(1+K)] ≈ 1/(N·P_τ), so the rolldown adds ≈ J/N per ticket:
the draw is favourable whenever sales stay below roughly J / (price·(1 − base RTP)).
Historical rolldown draws are identified from sales surges and their pots rebuilt from the
accrual rate implied by rolldown timing (see the history section).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats

from vietlott_engine.core.games import DEFAULT_TAX, GameSpec, TaxRule
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.prizes import PrizeHistory


def _share_factor(lam: float) -> float:
    """E[1/(1+K)], K ~ Poisson(λ) = (1 − e^{−λ}) / λ."""
    return 1.0 if lam <= 1e-12 else float((1 - np.exp(-lam)) / lam)


class RolldownEV(BaseModel):
    jackpot: float
    tickets_sold: int
    p_jackpot_won: float  # someone hits 5+special → no rolldown
    tier_fractions: dict[str, float]  # share of the pot for a tier *given it has a winner*, incl. expected top-up from empty tiers
    expected_value: float
    return_to_player: float
    base_return_to_player: float  # same draw without rolldown
    break_even_tickets_sold: float | None  # sales at which RTP = 1


def _ev_parts(spec: GameSpec, jackpot: float, sold: float, tax: TaxRule, popularity_ratio: float = 1.0) -> tuple[float, float, float, dict]:
    probs = spec.tier_probabilities
    rd = spec.rolldown
    assert rd is not None
    base = sum(probs[t.name] * tax.after_tax(t.fixed_amount) for t in spec.tiers if t.fixed_amount is not None)
    jp_prob = probs["jackpot1"]
    p_jp_won = 1 - np.exp(-sold * jp_prob)
    # expected redistribution: tiers that are empty pass their share to the others
    lam = {t: sold * probs[t] for t in rd.shares}
    p_empty = {t: np.exp(-lam[t]) for t in rd.shares}
    eff = {}
    for t, f in rd.shares.items():
        extra = sum(rd.shares[u] * p_empty[u] / (len(rd.shares) - 1) for u in rd.shares if u != t)
        eff[t] = f + extra
    roll = 0.0
    for t, f in eff.items():
        fixed = float(spec.tier(t).fixed_amount or 0)
        share_if_win = jackpot * f * _share_factor(lam[t])
        # tax applies to the whole winning (fixed + share) — use the mean share (convex effect small)
        roll += probs[t] * (tax.after_tax(fixed + share_if_win) - tax.after_tax(fixed))
    # jackpot tier itself (if won in this draw the winner takes the pot)
    jp_part = jp_prob * tax.after_tax(jackpot) * _share_factor(sold * jp_prob * popularity_ratio)
    ev_roll_draw = base + (1 - p_jp_won) * roll + jp_part
    return ev_roll_draw, base + jp_part, float(p_jp_won), eff


def rolldown_ev(spec: GameSpec, jackpot: float, tickets_sold: int, tax: TaxRule = DEFAULT_TAX) -> RolldownEV:
    if spec.rolldown is None:
        raise ValueError(f"{spec.display_name} has no rolldown rule")
    ev, base, p_jp, eff = _ev_parts(spec, jackpot, tickets_sold, tax)

    def gap(n: float) -> float:
        return _ev_parts(spec, jackpot, n, tax)[0] - spec.ticket_price

    be = None
    try:
        if gap(1e3) > 0 and gap(1e9) < 0:
            be = float(optimize.brentq(gap, 1e3, 1e9, xtol=10))
    except ValueError:
        be = None
    return RolldownEV(
        jackpot=jackpot,
        tickets_sold=int(tickets_sold),
        p_jackpot_won=p_jp,
        tier_fractions={k: float(v) for k, v in eff.items()},
        expected_value=ev,
        return_to_player=ev / spec.ticket_price,
        base_return_to_player=base / spec.ticket_price,
        break_even_tickets_sold=be,
    )


# ------------------------------------------------------------- history
#
# Reconstruction of past rolldowns (validated against press reports):
#
# 1. *Events* are read from the data, not from a model: the announced rolldown draw is the
#    21:00 draw whose sales jump ≥ 3× the trailing median (players buy for the rolldown).
#    Checked against the press: 5 rolldowns in the first 3 months (#38, #54, #96, #146,
#    #168; #70 was announced but the jackpot was won), and the winner counts of #38, #54,
#    #146 match the published per-tier rolldown prizes exactly.
# 2. *Accrual*: the advertised jackpot grows at a rate c per đồng of sales. The rolldown
#    timing pins c down for every segment: the pot crossed 12 tỷ during day d−1 but not
#    before, so c ∈ (6 tỷ / R(→d−1), 6 tỷ / R(→d−2)]. Segments that ended in a jackpot win
#    without an announcement give upper bounds. c is the robust centre of those intervals.
# 3. The pot actually distributed is larger than c alone explains: sales on the rolldown day
#    feed it at a higher rate c_r, estimated from published rolldown values. Accounting
#    check: fixed prizes ≈ 17 % + visible accrual ≈ 16 % + 6-tỷ restarts ≈ 22 % ≈ the 54 %
#    of sales Vietlott reported as prizes paid in 01–07/2026.


class RolldownEvent(BaseModel):
    draw_id: int
    date: str
    tickets_sold: float
    sales_multiple: float  # vs median of the previous 14 draws
    executed: bool  # False: the jackpot was won in the announced draw (no rolldown)
    jackpot_estimate: float
    jackpot_published: float | None = None
    tier_prizes: dict[str, float]  # per winning ticket, fixed + share, from the actual winner counts
    realised_rtp_pre_tax: float  # (pot + fixed prizes) / sales of that draw
    realised_rtp_conservative: float  # same with rolldown-day sales feeding the pot only at c
    model_rtp_after_tax: float  # expected-value model (rolldown_ev) at the same pot and sales


class RolldownHistory(BaseModel):
    game: str
    announced: int
    executed: int
    accrual_rate: float  # c: jackpot growth per đồng of sales (normal draws)
    accrual_rate_range: tuple[float, float]  # 10–90 % of the per-segment interval midpoints
    timing_consistency: float  # share of segments whose interval contains c (±10 %)
    rolldown_day_rate: float  # c_r: growth per đồng of sales on the rolldown day
    anchors: list[dict]
    events: list[RolldownEvent]
    mean_realised_rtp: float | None
    rtp_trend: dict
    note: str


def detect_rolldowns(h: DrawHistory, sold: np.ndarray, min_multiple: float = 3.0, window: int = 14) -> list[int]:
    """Indices of announced rolldown draws: last draw of its day with a sales surge."""
    dates = [d.astype("datetime64[D]").astype(date) for d in h.dates]
    out = []
    for t in range(1, len(sold)):
        prev = sold[max(0, t - window) : t]
        if not np.isfinite(sold[t]) or not np.isfinite(prev).any():
            continue
        last_of_day = t == len(sold) - 1 or dates[t + 1] != dates[t]
        if last_of_day and sold[t] >= min_multiple * np.nanmedian(prev):
            out.append(t)
    return out


def realised_tier_prizes(spec: GameSpec, winners: dict[str, float], jackpot: float) -> dict[str, float]:
    """Prize per winning ticket in a rolldown draw from the actual winner counts.
    Shares of tiers without winners are split equally among the tiers that have winners."""
    rd = spec.rolldown
    assert rd is not None
    live = [t for t in rd.shares if winners.get(t, 0) > 0]
    if not live:
        return {}
    dead = sum(f for t, f in rd.shares.items() if t not in live)
    out = {}
    for t in live:
        frac = rd.shares[t] + dead / len(live)
        out[t] = float(spec.tier(t).fixed_amount or 0) + jackpot * frac / winners[t]
    return out


def _interval_centre(lo: np.ndarray, hi: np.ndarray, upper_only: np.ndarray) -> float:
    """c minimising the squared log-distance to every interval (robust 'consensus' rate)."""
    grid = np.exp(np.linspace(np.log(0.02), np.log(0.8), 1500))
    lg = np.log(grid)[:, None]
    d = np.maximum(np.log(lo)[None, :] - lg, 0) + np.maximum(lg - np.log(hi)[None, :], 0)
    loss = (d**2).sum(axis=1)
    if len(upper_only):
        loss += (np.maximum(lg - np.log(upper_only)[None, :], 0) ** 2).sum(axis=1)
    return float(grid[int(np.argmin(loss))])


def rolldown_history(
    spec: GameSpec,
    h: DrawHistory,
    ph: PrizeHistory,
    sold: np.ndarray,
    anchors: list[tuple[int, float]] | None = None,
    tax: TaxRule = DEFAULT_TAX,
    min_multiple: float = 3.0,
) -> RolldownHistory:
    """Every announced rolldown with its estimated pot, per-tier prizes and realised return."""
    if spec.rolldown is None:
        raise ValueError(f"{spec.display_name} has no rolldown rule")
    ids = [int(x) for x in h.draw_ids]
    dates = [d.astype("datetime64[D]").astype(date) for d in h.dates]
    sold = np.asarray(sold, dtype=np.float64)
    fill = np.nanmedian(sold)
    n_t = np.where(np.isfinite(sold), sold, fill)
    rev = n_t * spec.ticket_price
    fixed = ph.fixed_prizes_paid()
    fixed = np.nan_to_num(fixed, nan=0.0)  # unknown → 0 (one draw)
    won = ph.jackpot_won()[:, 0]
    jmin = float(spec.min_jackpots["jackpot1"])
    gap = float(spec.rolldown.threshold) - jmin
    ann = detect_rolldowns(h, sold, min_multiple)
    ann_set = set(ann)
    executed = {t for t in ann if not won[t]}
    resets = sorted({t for t in range(len(ids)) if won[t]} | executed)

    # ---- timing intervals per segment
    lo, hi, upper_only, mids = [], [], [], []
    seg_of: dict[int, tuple[int, int]] = {}
    start = 0
    for r in resets + ([len(ids) - 1] if not resets or resets[-1] != len(ids) - 1 else []):
        seg = range(start, r + 1)
        d = dates[r]
        r_a = sum(rev[i] for i in seg if dates[i] <= d - timedelta(days=2))
        r_b = sum(rev[i] for i in seg if dates[i] <= d - timedelta(days=1))
        if r in ann_set and r_b > 0:
            lo.append(gap / r_b)
            hi.append(gap / r_a if r_a > 0 else 10.0)
            mids.append(np.sqrt(lo[-1] * min(hi[-1], 1.0)))
            seg_of[r] = (start, r)
        elif won[r] and r_a > 0:
            upper_only.append(gap / r_a)
        start = r + 1
    if lo:
        c = _interval_centre(np.array(lo), np.array(hi), np.array(upper_only))
        inside = float(np.mean([(lo_i / 1.1) < c <= hi_i * 1.1 for lo_i, hi_i in zip(lo, hi)]))
        c_rng = (float(np.quantile(mids, 0.1)), float(np.quantile(mids, 0.9)))
    else:
        c, inside, c_rng = 0.15, float("nan"), (float("nan"), float("nan"))

    # ---- rolldown-day rate from published pots
    anchors = anchors or []
    num = den = 0.0
    for a_id, a_val in anchors:
        if a_id not in ids or ids.index(a_id) not in seg_of:
            continue
        r = ids.index(a_id)
        s0, _ = seg_of[r]
        pre = sum(rev[i] for i in range(s0, r + 1) if dates[i] < dates[r])
        day = sum(rev[i] for i in range(s0, r + 1) if dates[i] == dates[r])
        y = a_val - jmin - c * pre
        num += y * day
        den += day * day
    c_r = float(num / den) if den > 0 else c

    # ---- path
    rd_days = {(dates[r]) for r in ann}
    path = np.full(len(ids), np.nan)
    cur = jmin
    for t in range(len(ids)):
        cur += (c_r if dates[t] in rd_days else c) * rev[t]
        path[t] = cur
        if t in executed or won[t]:
            cur = jmin

    anchor_rows = []
    for a_id, a_val in anchors:
        if a_id in ids:
            v = path[ids.index(a_id)]
            anchor_rows.append({"draw_id": a_id, "published": a_val, "reconstructed": float(v), "ratio": float(v / a_val)})

    events = []
    names = [t.name for t in spec.tiers]
    for r in ann:
        winners = {nm: float(ph.winners[r, names.index(nm)]) for nm in names}
        jp = float(path[r])
        prizes = realised_tier_prizes(spec, winners, jp) if r in executed else {}
        realised = (jp + fixed[r]) / rev[r]
        day_rev = sum(rev[i] for i in range(seg_of.get(r, (r, r))[0], r + 1) if dates[i] == dates[r])
        conservative = (jp - (c_r - c) * day_rev + fixed[r]) / rev[r]
        mult = float(sold[r] / np.nanmedian(sold[max(0, r - 14) : r]))
        pub = next((v for a_id, v in anchors if a_id == ids[r]), None)
        events.append(
            RolldownEvent(
                draw_id=ids[r],
                date=str(dates[r]),
                tickets_sold=float(n_t[r]),
                sales_multiple=mult,
                executed=r in executed,
                jackpot_estimate=jp,
                jackpot_published=pub,
                tier_prizes=prizes,
                realised_rtp_pre_tax=float(realised),
                realised_rtp_conservative=float(conservative),
                model_rtp_after_tax=rolldown_ev(spec, jp, int(n_t[r]), tax).return_to_player,
            )
        )
    rtps = np.array([e.realised_rtp_pre_tax for e in events if e.executed])
    trend = {}
    if len(rtps) >= 6:
        half = len(rtps) // 2
        x = np.arange(len(rtps))
        rho = stats.spearmanr(x, rtps)
        sales = np.array([e.tickets_sold for e in events if e.executed])
        trend = {
            "first_half_mean_rtp": float(rtps[:half].mean()),
            "second_half_mean_rtp": float(rtps[half:].mean()),
            "first_half_mean_sales": float(sales[:half].mean()),
            "second_half_mean_sales": float(sales[half:].mean()),
            "spearman_rho_vs_time": float(rho.statistic),
            "p_value": float(rho.pvalue),
            "share_above_1": float((rtps > 1).mean()),
            "share_above_1_conservative": float(np.mean([e.realised_rtp_conservative > 1 for e in events if e.executed])),
        }
    return RolldownHistory(
        game=spec.code.value,
        announced=len(ann),
        executed=len(executed),
        accrual_rate=c,
        accrual_rate_range=c_rng,
        timing_consistency=inside,
        rolldown_day_rate=c_r,
        anchors=anchor_rows,
        events=events,
        mean_realised_rtp=float(rtps.mean()) if len(rtps) else None,
        rtp_trend=trend,
        note=(
            "Events from sales surges (model-free); pot = 6 tỷ + c·sales since the last reset, with sales on the "
            "rolldown day counted at c_r. Realised RTP is before tax and counts the whole distributed pot plus "
            "fixed prizes against the sales of the rolldown draw only."
        ),
    )


class AccrualRegimes(BaseModel):
    """Growth of the advertised Lotto jackpot per đồng of sales, from the official series."""

    slow_rate: float  # pooled ΔJ / sales while the jackpot is in its slow phase
    fast_rate: float  # pooled ΔJ / sales in the fast phase
    slow_total_rate: float  # (ΔJ + fixed prizes) / sales, slow phase
    fast_total_rate: float  # (ΔJ + fixed prizes) / sales, fast phase
    switch_level_median: float  # jackpot level at which a segment turns fast
    switch_level_range: tuple[float, float]
    diverted_per_restart_median: float  # ≈ 6 tỷ if the slow phase refunds each restart
    diverted_per_segment: list[dict]  # sales × (fast − slow rate) before the switch vs restarts paid
    note: str


class OfficialRolldowns(BaseModel):
    game: str
    executed: int
    pre_empted: int  # announced (pot > 12 tỷ the day before) but the jackpot was won
    events: list[RolldownEvent]
    mean_realised_rtp: float | None
    rtp_trend: dict
    accrual: AccrualRegimes
    reconstruction_check: dict  # how the sales-surge reconstruction (no jackpot series) fared
    note: str


def rolldown_history_official(
    spec: GameSpec,
    h: DrawHistory,
    ph: PrizeHistory,
    sold: np.ndarray,
    tax: TaxRule = DEFAULT_TAX,
    reconstruction: RolldownHistory | None = None,
) -> OfficialRolldowns:
    """Rolldowns read from the official jackpot series (vietlott.vn detail pages).

    A rolldown is executed when the pot exceeded the threshold, nobody won it, and the
    next pot restarts near the minimum. The pot on the page of the rolldown draw is the
    amount distributed, so realised returns need no reconstruction."""
    if spec.rolldown is None:
        raise ValueError(f"{spec.display_name} has no rolldown rule")
    ids = [int(x) for x in h.draw_ids]
    dates = [d.astype("datetime64[D]").astype(date) for d in h.dates]
    pots = ph.pots[:, 0]
    won = ph.jackpot_won()[:, 0]
    fixed = np.nan_to_num(ph.fixed_prizes_paid(), nan=0.0)
    n_t = np.where(np.isfinite(sold), sold, np.nanmedian(sold))
    rev = n_t * spec.ticket_price
    thr = float(spec.rolldown.threshold)
    jmin = float(spec.min_jackpots["jackpot1"])
    names = [t.name for t in spec.tiers]
    # announcement: the pot exceeds the threshold after a draw of day d−1 → rolldown at the
    # last draw of day d, unless the jackpot is won on day d (pre-empted)
    executed, pre_empted = [], []
    day_last: dict[date, int] = {}
    for t, d in enumerate(dates):
        day_last[d] = t
    days = sorted(day_last)
    for k in range(1, len(days)):
        d, prev_d = days[k], days[k - 1]
        if (d - prev_d).days != 1:
            continue
        prev_idx = [t for t in range(len(ids)) if dates[t] == prev_d]
        last_prev = prev_idx[-1]
        # the pot standing after the last draw of d−1 decides: it only grows within a day
        # unless won (reset) or distributed by a rolldown executed at that draw (reset)
        if not (np.isfinite(pots[last_prev]) and pots[last_prev] > thr) or won[last_prev] or last_prev in executed:
            continue
        today = [t for t in range(len(ids)) if dates[t] == d]
        winners_today = [t for t in today if won[t]]
        if winners_today:
            pre_empted.append(winners_today[0])
        elif today and today[-1] + 1 < len(ids) and np.isfinite(pots[today[-1] + 1]) and pots[today[-1] + 1] < 0.6 * pots[today[-1]]:
            executed.append(today[-1])
    executed = sorted(set(executed))
    pre_empted = sorted(set(pre_empted) - set(executed))
    events = []
    for t in executed + pre_empted:
        winners = {nm: float(ph.winners[t, names.index(nm)]) for nm in names}
        jp = float(pots[t])
        prev = sold[max(0, t - 14) : t]
        events.append(
            RolldownEvent(
                draw_id=ids[t],
                date=str(dates[t]),
                tickets_sold=float(n_t[t]),
                sales_multiple=float(sold[t] / np.nanmedian(prev)) if np.isfinite(prev).any() else float("nan"),
                executed=t in executed,
                jackpot_estimate=jp,
                jackpot_published=jp,
                tier_prizes=realised_tier_prizes(spec, winners, jp) if t in executed else {},
                realised_rtp_pre_tax=float((jp + fixed[t]) / rev[t]),
                realised_rtp_conservative=float((jp + fixed[t]) / rev[t]),
                model_rtp_after_tax=rolldown_ev(spec, jp, int(n_t[t]), tax).return_to_player,
            )
        )
    events.sort(key=lambda e: e.draw_id)
    rtps = np.array([e.realised_rtp_pre_tax for e in events if e.executed])
    trend: dict = {}
    if len(rtps) >= 6:
        rho = stats.spearmanr(np.arange(len(rtps)), rtps)
        half = len(rtps) // 2
        trend = {
            "first_half_mean_rtp": float(rtps[:half].mean()),
            "second_half_mean_rtp": float(rtps[half:].mean()),
            "spearman_rho_vs_time": float(rho.statistic),
            "p_value": float(rho.pvalue),
            "share_above_1": float((rtps > 1).mean()),
            "min": float(rtps.min()),
            "max": float(rtps.max()),
        }

    # ---- accrual regimes
    resets = set(executed) | {t for t in range(len(ids)) if won[t]}
    rows = []
    for t in range(1, len(ids)):
        if won[t] or not np.isfinite(pots[t]) or not np.isfinite(sold[t]):
            continue
        base = jmin if (t - 1) in resets else pots[t - 1]
        if not np.isfinite(base):
            continue
        rows.append((t, base, pots[t] - base, rev[t], fixed[t]))
    arr = np.array(rows)
    rate = arr[:, 2] / arr[:, 3]
    fast = rate > 0.2
    switch_levels, diverted = [], []
    start = 0
    seg_resets = sorted(resets)
    last_switch = -1  # restarts between two fast phases are funded in the slow phase
    for r in seg_resets + [len(ids)]:
        seg = [i for i, row in enumerate(rows) if start <= row[0] < r]
        if seg:
            k = next((i for i in seg if fast[i]), None)
            if k is not None:
                switch_t = rows[k][0]
                switch_levels.append(rows[k][1])
                slow_rev = sum(row[3] for row in rows if last_switch < row[0] < switch_t and not fast[rows.index(row)])
                restarts = sum(1 for x in seg_resets if last_switch < x < switch_t) + (1 if last_switch < 0 else 0)
                diverted.append(
                    {"switch_draw": ids[switch_t], "level_at_switch": float(rows[k][1]), "slow_phase_sales": float(slow_rev), "restarts_since_previous_fast_phase": restarts}
                )
                last_switch = max(t for t, *_ in rows if start <= t < r and fast[[row[0] for row in rows].index(t)]) if any(fast[i] for i in seg) else last_switch
        start = r + 1
    slow_rate = float(arr[~fast, 2].sum() / arr[~fast, 3].sum())
    fast_rate = float(arr[fast, 2].sum() / arr[fast, 3].sum())
    for dv in diverted:
        dv["diverted"] = dv["slow_phase_sales"] * (fast_rate - slow_rate)
        dv["diverted_per_restart"] = dv["diverted"] / max(dv["restarts_since_previous_fast_phase"], 1)
    accrual = AccrualRegimes(
        slow_rate=slow_rate,
        fast_rate=fast_rate,
        slow_total_rate=float((arr[~fast, 2] + arr[~fast, 4]).sum() / arr[~fast, 3].sum()),
        fast_total_rate=float((arr[fast, 2] + arr[fast, 4]).sum() / arr[fast, 3].sum()),
        switch_level_median=float(np.median(switch_levels)) if switch_levels else float("nan"),
        switch_level_range=(float(np.min(switch_levels)), float(np.max(switch_levels))) if switch_levels else (float("nan"), float("nan")),
        diverted_per_restart_median=float(np.median([d["diverted_per_restart"] for d in diverted])) if diverted else float("nan"),
        diverted_per_segment=diverted,
        note=(
            "Each jackpot segment starts slow (the pot grows by a small share of sales) and switches to a fast phase "
            "in which practically the whole prize fund net of fixed prizes goes into the pot. The slow-phase difference "
            "is consistent with a reserve that funds the 6-tỷ restart of every jackpot; this is an inference from the "
            "published values, not a stated rule."
        ),
    )
    check: dict = {}
    if reconstruction is not None:
        rec = {e.draw_id: e for e in reconstruction.events}
        off = {e.draw_id: e for e in events}
        common = sorted(set(rec) & set(off))
        errs = [rec[i].jackpot_estimate / off[i].jackpot_estimate - 1 for i in common]
        check = {
            "official_events": len(off),
            "reconstructed_events": len(rec),
            "matched": len(common),
            "missed": sorted(set(off) - set(rec)),
            "spurious": sorted(set(rec) - set(off)),
            "pot_error_median_abs": float(np.median(np.abs(errs))) if errs else None,
            "pot_error_max_abs": float(np.max(np.abs(errs))) if errs else None,
            "executed_flag_agreement": float(np.mean([rec[i].executed == off[i].executed for i in common])) if common else None,
        }
    return OfficialRolldowns(
        game=spec.code.value,
        executed=len(executed),
        pre_empted=len(pre_empted),
        events=events,
        mean_realised_rtp=float(rtps.mean()) if len(rtps) else None,
        rtp_trend=trend,
        accrual=accrual,
        reconstruction_check=check,
        note="Pots, winners and per-tier rolldown prizes come from vietlott.vn detail pages; sales are estimated from winner counts.",
    )


class RolldownForecast(BaseModel):
    last_draw_id: int
    last_date: str
    pot: float
    phase: str  # "slow" | "fast"
    refund_outstanding: float  # restarts × refund still to be collected before the fast phase
    restarts_pending: int
    typical_sales_per_draw: float
    draws_to_fast_phase: float
    draws_to_threshold: float
    expected_rolldown_window: str
    note: str


def forecast_next_rolldown(spec: GameSpec, h: DrawHistory, ph: PrizeHistory, sold: np.ndarray, official: OfficialRolldowns) -> RolldownForecast:
    """When will the pot turn fast and cross the threshold? (official series + two-phase accrual)."""
    acc = official.accrual
    pots = ph.pots[:, 0]
    won = ph.jackpot_won()[:, 0]
    ids = [int(x) for x in h.draw_ids]
    t = int(np.flatnonzero(np.isfinite(pots))[-1])
    n_typ = float(np.nanmedian(sold[max(0, t - 28) : t + 1]))
    r_typ = n_typ * spec.ticket_price
    executed_ids = {e.draw_id for e in official.events if e.executed}
    resets = {i for i in range(t + 1) if won[i] or ids[i] in executed_ids}
    jmin = float(spec.min_jackpots["jackpot1"])
    rev = np.where(np.isfinite(sold), sold, n_typ) * spec.ticket_price
    rate = np.full(t + 1, np.nan)
    for i in range(1, t + 1):
        if won[i] or not np.isfinite(pots[i]):
            continue
        base = jmin if (i - 1) in resets else pots[i - 1]
        rate[i] = (pots[i] - base) / rev[i]
    fast_rows = [i for i in range(1, t + 1) if np.isfinite(rate[i]) and rate[i] > 0.2 and i not in resets]
    last_fast = fast_rows[-1] if fast_rows else -1
    fast_now = t == last_fast
    pending = [i for i in resets if i > last_fast]
    slow_sales = float(sum(rev[i] for i in range(last_fast + 1, t + 1) if np.isfinite(rate[i])))
    thr = float(spec.rolldown.threshold) if spec.rolldown else float("nan")
    if fast_now:
        refund, to_fast = 0.0, 0.0
        to_thr = max(0.0, (thr - pots[t]) / (acc.fast_rate * r_typ))
    else:
        refund = max(0.0, len(pending) * acc.diverted_per_restart_median - slow_sales * (acc.fast_rate - acc.slow_rate))
        to_fast = refund / ((acc.fast_rate - acc.slow_rate) * r_typ)
        to_thr_slow = max(0.0, (thr - pots[t]) / (acc.slow_rate * r_typ))
        if to_thr_slow <= to_fast:
            to_thr = to_thr_slow
        else:
            level_at_switch = pots[t] + to_fast * acc.slow_rate * r_typ
            to_thr = to_fast + max(0.0, (thr - level_at_switch) / (acc.fast_rate * r_typ))
    days = to_thr / 2.0  # two draws a day
    return RolldownForecast(
        last_draw_id=ids[t],
        last_date=str(h.dates[t]),
        pot=float(pots[t]),
        phase="fast" if fast_now else "slow",
        refund_outstanding=float(refund),
        typical_sales_per_draw=n_typ,
        draws_to_fast_phase=float(to_fast),
        draws_to_threshold=float(to_thr),
        restarts_pending=len(pending),
        expected_rolldown_window=f"khoảng {days:.0f}–{days + 1.5:.0f} ngày nữa (kỳ 21:00 của ngày sau khi vượt ngưỡng), nếu không ai trúng Độc đắc trước",
        note="Typical sales = median of the last 28 draws. Jackpot wins before the threshold restart the count.",
    )


def backtest_rolldown_forecast(spec: GameSpec, h: DrawHistory, ph: PrizeHistory, sold: np.ndarray, official: OfficialRolldowns, start: int = 150, step: int = 10) -> dict:
    """Forecast from every ``step``-th draw using only data up to it; compare with the draw at
    which the pot really crossed the threshold (cases ended by a jackpot win are skipped).
    The accrual rates come from the full series (a mild look-ahead on two constants)."""
    from vietlott_engine.core.prizes import PrizeHistory as _PH

    pots = ph.pots[:, 0]
    won = ph.jackpot_won()[:, 0]
    thr = float(spec.rolldown.threshold) if spec.rolldown else float("nan")
    rows = []
    for cut in range(start, len(h) - 1, step):
        sub = _PH(spec=ph.spec, draw_ids=ph.draw_ids[:cut], winners=ph.winners[:cut], pots=ph.pots[:cut], tier_names=ph.tier_names, jackpot_tiers=ph.jackpot_tiers)
        if not np.isfinite(pots[cut - 1]) or pots[cut - 1] > thr:
            continue
        f = forecast_next_rolldown(spec, h.upto(cut), sub, sold[:cut], official)
        actual = None
        for k in range(cut, len(h)):
            if won[k]:
                break
            if pots[k] > thr:
                actual = k - cut + 1
                break
        if actual is not None:
            rows.append((int(h.draw_ids[cut - 1]), f.draws_to_threshold, actual))
    if not rows:
        return {"cases": 0}
    pred = np.array([r[1] for r in rows])
    act = np.array([r[2] for r in rows])
    err = pred - act
    return {
        "cases": len(rows),
        "median_abs_error_draws": float(np.median(np.abs(err))),
        "mean_error_draws": float(err.mean()),
        "share_within_3_draws": float(np.mean(np.abs(err) <= 3)),
        "spearman_rho": float(stats.spearmanr(pred, act).statistic),
        "examples": [{"from_draw": r[0], "predicted": round(r[1], 1), "actual": r[2]} for r in rows[-8:]],
    }


def tickets_sold_ci(n: float, rel_sd: float = 0.15) -> tuple[float, float]:
    """Rough 95 % band for a sales estimate with relative error ``rel_sd``."""
    z = stats.norm.ppf(0.975)
    return float(n * np.exp(-z * rel_sd)), float(n * np.exp(z * rel_sd))
