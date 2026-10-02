"""Walk-forward backtesting engine (no look-ahead bias by construction).

For every target draw t in [start, end):
  1. view = history.upto(t).detached()  — an independent, read-only copy of draws < t
  2. strategy.fit(view)                 — every ``refit_every`` draws
  3. tickets = strategy.generate(view)  — validated (range, uniqueness, shape)
  4. tickets are scored against draw t

Structural guarantees: the strategy never holds a reference to arrays containing
draw t or later (the prefix is copied, so ``ndarray.base`` cannot reach the future),
and all arrays are read-only. ``LookAheadError`` is raised if a strategy returns
malformed tickets or the view length ever disagrees with t.

Statistics: per-ticket match counts are compared with the exact hypergeometric
null (G-test, z-test on the mean); every strategy is compared with the quick-pick
benchmark (Welch t-test on matches), with Benjamini–Hochberg control across
strategies; ROI has a block-bootstrap 95% CI.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.analytics.stats import benjamini_hochberg, pooled_gof
from vietlott_engine.inference.bootstrap import SPAResult, spa_test
from vietlott_engine.inference.multiple_testing import holm
from vietlott_engine.inference.power import certified_mean_margin, mde_mean
from vietlott_engine.inference.sequential import ticket_eprocess
from vietlott_engine.backtest.strategies import Strategy
from vietlott_engine.core.exceptions import InsufficientDataError, LookAheadError
from vietlott_engine.core.games import DEFAULT_TAX, TaxRule
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.logging import get_logger
from vietlott_engine.game_theory.popularity import PopularityModel

log = get_logger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    start: int = 300
    end: int | None = None
    tickets_per_draw: int = 1
    seed: int = 0
    apply_tax: bool = True
    bootstrap_samples: int = 1000
    bootstrap_block: int = 20
    spa_bootstrap: int = 2000
    alpha: float = 0.05


class StrategyResult(BaseModel):
    strategy: str
    draws: int
    tickets: int
    spend: float
    payout: float
    roi: float
    roi_ci95: tuple[float, float]
    match_histogram: list[int]
    expected_histogram: list[float]
    mean_matches: float
    expected_mean_matches: float
    z_mean_matches: float  # cluster-robust by draw (tickets of one draw may be correlated)
    z_iid: float  # naive, treats every ticket as independent (for reference)
    design_effect: float  # variance inflation from within-draw ticket correlation
    gof_p_value: float
    p_value_vs_null: float  # one-sided: more matches than chance?
    q_value_vs_null: float | None = None
    p_holm: float | None = None
    e_value_max: float  # max wealth of the anytime-valid bettor (≥ 1/α ⇒ reject for one strategy)
    e_value_final: float
    anytime_p_value: float
    anytime_p_holm: float | None = None  # Holm across strategies (several bettors were run)
    mde_matches: float  # smallest edge (matches/ticket) this backtest detects with 80% power
    certified_edge_bound: float  # with 1−α confidence |edge| < this (TOST), matches/ticket
    certified_edge_bound_relative: float
    tier_counts: dict[str, int]
    tier_expected: dict[str, float]
    jackpot_hits: int
    mean_popularity_ratio: float | None
    vs_random_p_value: float | None = None
    runtime_s: float


class BacktestReport(BaseModel):
    game: str
    start_index: int
    end_index: int
    start_date: str
    end_date: str
    draws_evaluated: int
    tickets_per_draw: int
    results: list[StrategyResult]
    spa: SPAResult | None = None
    conclusion: str

    def to_markdown(self) -> str:
        lines = [
            f"# Walk-forward backtest — {self.game}",
            f"Draws {self.start_date} → {self.end_date} ({self.draws_evaluated} draws, {self.tickets_per_draw} ticket(s)/draw unless the strategy fixes its own count)",
            "",
            "| Strategy | Tickets | Mean matches (null) | z (cluster) | deff | q (>chance) | max e-value (Holm p) | edge < (95%) | MDE | ROI | ROI 95% CI | Popularity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.results:
            q = "—" if r.q_value_vs_null is None else f"{r.q_value_vs_null:.3f}"
            pop = "—" if r.mean_popularity_ratio is None else f"{r.mean_popularity_ratio:.2f}"
            lines.append(
                f"| {r.strategy} | {r.tickets} | {r.mean_matches:.4f} ({r.expected_mean_matches:.4f}) | {r.z_mean_matches:+.2f} | "
                f"{r.design_effect:.2f} | {q} | {r.e_value_max:.1f} ({(r.anytime_p_holm if r.anytime_p_holm is not None else r.anytime_p_value):.2f}) | {r.certified_edge_bound_relative:.1%} | "
                f"{r.mde_matches / r.expected_mean_matches:.1%} | {r.roi:+.1%} | [{r.roi_ci95[0]:+.1%}, {r.roi_ci95[1]:+.1%}] | {pop} |"
            )
        if self.spa is not None:
            lines += [
                "",
                f"Superior Predictive Ability (Hansen 2005, {self.spa.bootstrap_samples} stationary-bootstrap draws): best = "
                f"{self.spa.best_strategy}, p (consistent) = {self.spa.p_value_consistent:.3f}, "
                f"p (lower/upper) = {self.spa.p_value_lower:.3f}/{self.spa.p_value_upper:.3f}, White Reality Check p = "
                f"{self.spa.reality_check_p_value:.3f}.",
            ]
        lines += ["", self.conclusion, ""]
        return "\n".join(lines)


class WalkForwardBacktester:
    def __init__(self, history: DrawHistory, config: BacktestConfig | None = None, popularity: PopularityModel | None = None, tax: TaxRule = DEFAULT_TAX) -> None:
        self.h = history
        self.cfg = config or BacktestConfig()
        self.spec = history.spec
        self.popularity = popularity
        self.tax = tax
        self.draw_means: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ scoring
    def _score(self, tickets: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (matches, tier_index or -1, payout) for tickets vs draw t."""
        inc = self.h.incidence[t]
        matches = inc[tickets - 1].sum(axis=1)
        bonus_hit = (tickets == self.h.bonus[t]).any(axis=1) if self.spec.has_bonus else np.zeros(len(tickets), bool)
        tier_idx = np.full(len(tickets), -1)
        for j, tier in enumerate(self.spec.tiers):
            mask = (tier_idx == -1) & (matches == tier.main_matches)
            if tier.bonus_required:
                mask &= bonus_hit
            tier_idx[mask] = j
        payout = 0.0
        for j in tier_idx[tier_idx >= 0]:
            tier = self.spec.tiers[j]
            amount = float(tier.fixed_amount) if tier.fixed_amount is not None else float(self.spec.min_jackpots[tier.name])
            payout += self.tax.after_tax(amount) if self.cfg.apply_tax else amount
        return matches, tier_idx, payout

    def _validate(self, tickets: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(tickets)
        if arr.ndim != 2 or arr.shape[1] != self.spec.pick or arr.shape[0] == 0:
            raise LookAheadError(f"{name} returned tickets of shape {arr.shape}")
        arr = np.sort(arr.astype(np.int64), axis=1)
        if arr.min() < 1 or arr.max() > self.spec.pool_size or (np.diff(arr, axis=1) == 0).any():
            raise LookAheadError(f"{name} returned invalid numbers")
        return arr

    # ---------------------------------------------------------------- run
    def run_strategy(self, strategy: Strategy, start: int, end: int) -> tuple[StrategyResult, np.ndarray]:
        t0 = time.perf_counter()
        rng = np.random.default_rng([self.cfg.seed, zlib.crc32(strategy.name.encode())])
        all_matches: list[np.ndarray] = []
        draw_mean = np.zeros(end - start)
        tiers: list[np.ndarray] = []
        spend = np.zeros(end - start)
        payout = np.zeros(end - start)
        pops: list[float] = []
        fitted_at: int | None = None
        for i, t in enumerate(range(start, end)):
            view = self.h.upto(t).detached()
            if len(view) != t or (len(view) and view.draw_ids[-1] != self.h.draw_ids[t - 1]):
                raise LookAheadError("history view does not end strictly before the target draw")
            if fitted_at is None or t - fitted_at >= strategy.refit_every:
                strategy.fit(view)
                fitted_at = t
            tickets = self._validate(strategy.generate(view, self.cfg.tickets_per_draw, rng), strategy.name)
            m, tier_idx, pay = self._score(tickets, t)
            all_matches.append(m)
            draw_mean[i] = m.mean()
            tiers.append(tier_idx)
            spend[i] = len(tickets) * self.spec.ticket_price
            payout[i] = pay
            if self.popularity is not None:
                pops.extend(self.popularity.popularity_ratio(tickets).tolist())
        matches = np.concatenate(all_matches)
        tier_arr = np.concatenate(tiers)
        n_tickets = matches.size
        k, n = self.spec.pick, self.spec.pool_size
        null = np.array(self.spec.match_distribution())
        hist = np.bincount(matches, minlength=k + 1)
        exp_hist = null * n_tickets
        gof = pooled_gof(hist, null)
        mu = k * k / n
        var = float(np.sum(null * (np.arange(k + 1) - mu) ** 2))
        z_iid = (matches.mean() - mu) / np.sqrt(var / n_tickets)
        # draws are the independent units (martingale differences under H0): cluster-robust SE
        t_draws = draw_mean.size
        w = np.array([len(x) for x in all_matches], dtype=np.float64)
        wmean = float(np.sum(w * draw_mean) / w.sum())
        resid = w * (draw_mean - wmean)
        se_cl = float(np.sqrt(np.sum(resid**2)) / w.sum() * np.sqrt(t_draws / max(t_draws - 1, 1)))
        se_cl = max(se_cl, 1e-12)
        z = (matches.mean() - mu) / se_cl
        deff = (se_cl**2) / (var / n_tickets)
        ep = ticket_eprocess(strategy.name, draw_mean, mu, self.cfg.alpha)
        bound = certified_mean_margin(float(matches.mean() - mu), se_cl, self.cfg.alpha)
        self.draw_means[strategy.name] = draw_mean
        probs = self.spec.tier_probabilities
        tier_counts = {t.name: int(np.sum(tier_arr == j)) for j, t in enumerate(self.spec.tiers)}
        roi = payout.sum() / spend.sum() - 1
        result = StrategyResult(
            strategy=strategy.name,
            draws=end - start,
            tickets=int(n_tickets),
            spend=float(spend.sum()),
            payout=float(payout.sum()),
            roi=float(roi),
            roi_ci95=self._bootstrap_roi(spend, payout),
            match_histogram=[int(x) for x in hist],
            expected_histogram=[float(x) for x in exp_hist],
            mean_matches=float(matches.mean()),
            expected_mean_matches=mu,
            z_mean_matches=float(z),
            z_iid=float(z_iid),
            design_effect=float(deff),
            gof_p_value=gof.p_value,
            p_value_vs_null=float(stats.norm.sf(z)),
            e_value_max=float(10**ep.max_log10_wealth),
            e_value_final=float(10**ep.final_log10_wealth),
            anytime_p_value=ep.anytime_p_value,
            mde_matches=mde_mean(se_cl, self.cfg.alpha, 0.8, one_sided=True),
            certified_edge_bound=bound,
            certified_edge_bound_relative=bound / mu,
            tier_counts=tier_counts,
            tier_expected={name: float(p * n_tickets) for name, p in probs.items()},
            jackpot_hits=int(sum(tier_counts[t.name] for t in self.spec.tiers if t.is_jackpot)),
            mean_popularity_ratio=float(np.mean(pops)) if pops else None,
            runtime_s=round(time.perf_counter() - t0, 2),
        )
        log.info("%s: %d tickets, mean matches %.4f (null %.4f), ROI %+.1f%%", strategy.name, n_tickets, matches.mean(), mu, 100 * roi)
        return result, matches

    def _bootstrap_roi(self, spend: np.ndarray, payout: np.ndarray) -> tuple[float, float]:
        rng = np.random.default_rng(self.cfg.seed + 1)
        n, b = spend.size, max(1, min(self.cfg.bootstrap_block, spend.size))
        starts_max = n - b + 1
        n_blocks = int(np.ceil(n / b))
        rois = np.empty(self.cfg.bootstrap_samples)
        for s in range(self.cfg.bootstrap_samples):
            idx = (rng.integers(0, starts_max, n_blocks)[:, None] + np.arange(b)[None, :]).ravel()[:n]
            rois[s] = payout[idx].sum() / spend[idx].sum() - 1
        return (float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975)))

    def run(self, strategies: list[Strategy]) -> BacktestReport:
        if not strategies:
            raise ValueError("no strategies given")
        start = max(self.cfg.start, max(s.min_history for s in strategies))
        end = len(self.h) if self.cfg.end is None else min(self.cfg.end, len(self.h))
        if end - start < 10:
            raise InsufficientDataError(f"backtest window [{start}, {end}) too short")
        results: list[StrategyResult] = []
        matches: dict[str, np.ndarray] = {}
        for s in strategies:
            r, m = self.run_strategy(s, start, end)
            results.append(r)
            matches[s.name] = m
        # primary test: each strategy vs the exact hypergeometric null (cluster-robust), FDR and FWER
        pv = [r.p_value_vs_null for r in results]
        for r, q, ph, pa in zip(results, benjamini_hochberg(pv), holm(pv), holm([r.anytime_p_value for r in results])):
            r.q_value_vs_null, r.p_holm, r.anytime_p_holm = float(q), float(ph), float(pa)
        # secondary: vs the quick-pick realisation actually played
        ref = matches.get("random")
        if ref is not None:
            for r in results:
                if r.strategy != "random":
                    r.vs_random_p_value = float(stats.ttest_ind(matches[r.strategy], ref, equal_var=False).pvalue)
        # data-snooping-robust: is the *best* strategy better than chance?
        mu = self.spec.pick**2 / self.spec.pool_size
        names = [r.strategy for r in results]
        diffs = np.column_stack([self.draw_means[nm] - mu for nm in names])
        spa = spa_test(diffs, names, bootstrap=self.cfg.spa_bootstrap, mean_block=self.cfg.bootstrap_block, seed=self.cfg.seed)
        beats = [r.strategy for r in results if r.q_value_vs_null is not None and r.q_value_vs_null < self.cfg.alpha]
        anytime = [r.strategy for r in results if r.anytime_p_holm is not None and r.anytime_p_holm < self.cfg.alpha]
        streaks = [r.strategy for r in results if r.e_value_max >= 1 / self.cfg.alpha and r.strategy not in anytime]
        worst_bound = max(r.certified_edge_bound_relative for r in results)
        if beats or anytime or spa.p_value_consistent < self.cfg.alpha:
            conclusion = (
                f"Candidate edge — BH: {beats or 'none'}; anytime e-value: {anytime or 'none'}; SPA p = "
                f"{spa.p_value_consistent:.3f} (best {spa.best_strategy}). Treat as a hypothesis and confirm on draws "
                "after this window before trusting it."
            )
        else:
            conclusion = (
                f"No strategy matched more numbers than chance: every BH q ≥ {self.cfg.alpha}, no anytime-valid "
                f"e-process is significant after Holm across the {len(results)} bettors, and the best strategy "
                "(data-snooping-adjusted SPA) has p = "
                f"{spa.p_value_consistent:.2f}. Equivalence (TOST, {1 - self.cfg.alpha:.0%}): every strategy's edge is "
                f"below {worst_bound:.1%} of the chance level of {mu:.3f} matches/ticket. ROI differences come from a "
                "handful of rare prize hits; the only structural lever is jackpot sharing (Popularity column)."
            )
        if streaks:
            conclusion += (
                f" Temporary lucky streaks (e-value ≥ {1 / self.cfg.alpha:.0f}× at some point, not significant once all "
                f"{len(results)} bettors are accounted for): "
                + ", ".join(f"{r.strategy} (peak {r.e_value_max:.0f}×, final {r.e_value_final:.2f}×)" for r in results if r.strategy in streaks)
                + "."
            )
        if off_null := [r.strategy for r in results if r.gof_p_value < self.cfg.alpha / len(results)]:
            conclusion += (
                f" Match histogram deviates from the null for {off_null} (Bonferroni); with overlapping tickets the "
                "multinomial GOF is anti-conservative — rely on the cluster-robust z instead."
            )
        return BacktestReport(
            game=self.spec.code.value,
            start_index=start,
            end_index=end,
            start_date=str(self.h.dates[start]),
            end_date=str(self.h.dates[end - 1]),
            draws_evaluated=end - start,
            tickets_per_draw=self.cfg.tickets_per_draw,
            results=results,
            spa=spa,
            conclusion=conclusion,
        )
