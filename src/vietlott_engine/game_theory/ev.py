"""Expected-value engine and Anti-Popularity ticket optimiser.

EV of one play (after tax), with J₁ the jackpot and K the number of *other*
winning tickets on the same combination:

    EV = Σ_fixed P(tier)·net(prize) + P(jackpot)·E_K[ net(J₁ / (1 + K)) ]

K ~ Poisson(λ), λ = N·p(c): N tickets sold to others, p(c) the population
probability of the combination (``PopularityModel``). For Power 6/55 Jackpot 2 the
co-winners are the six tickets "5 main numbers + bonus"; we use
λ₂ = N·(p(c) + 5/C(n,k)) — our own combination's popularity plus five average
neighbours.

Win probabilities do not depend on the combination, so maximising EV is exactly
minimising λ: the optimiser searches for *unpopular* combinations. It cannot make
EV positive unless the jackpot is large enough; ``break_even_jackpot`` says how
large.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats

from vietlott_engine.core.exceptions import DataValidationError
from vietlott_engine.core.games import DEFAULT_TAX, GameSpec, TaxRule
from vietlott_engine.core.models import validate_numbers
from vietlott_engine.game_theory.popularity import PopularityModel


class TierEV(BaseModel):
    tier: str
    probability: float
    gross_prize: float
    expected_net_prize_if_won: float
    ev_contribution: float


class EVBreakdown(BaseModel):
    game: str
    ticket: list[int]
    special: int | None = None
    ticket_price: int
    jackpot1: float
    jackpot2: float | None
    tickets_sold: int
    tickets_sold_source: str = "given"
    popularity_ratio: float
    expected_co_winners: float
    tiers: list[TierEV]
    expected_value: float
    return_to_player: float
    expected_loss: float
    break_even_jackpot1: float | None
    break_even_jackpot1_with_sales_response: float | None = None


def expected_net_share(pot: float, lam: float, tax: TaxRule) -> float:
    """E[ net(pot / (1 + K)) ], K ~ Poisson(lam) — exact summation of the pmf."""
    if lam <= 0:
        return tax.after_tax(pot)
    if lam > 1e8:  # share concentrated at pot/(1+λ) (relative spread 1/√λ): E[1/(1+K)] = (1 − e^−λ)/λ
        return tax.after_tax(pot * (1.0 - np.exp(-lam)) / lam)
    # window holding all but ~1e-12 of the mass on each side: O(√λ) terms, not O(λ)
    if lam < 1e3:
        kmin, kmax = 0, int(stats.poisson.ppf(1 - 1e-12, lam)) + 2
    else:
        half = 15.0 * np.sqrt(lam)
        kmin, kmax = max(0, int(lam - half)), int(lam + half) + 1
    ks = np.arange(kmin, kmax + 1)
    pmf = stats.poisson.pmf(ks, lam)
    shares = pot / (1 + ks)
    net = shares - tax.rate * np.maximum(0.0, shares - tax.threshold)
    return float(np.sum(pmf * net) / pmf.sum())


class EVCalculator:
    """EV of one play. ``sales`` (a fitted ``SalesModel``) makes ticket sales respond to
    the jackpot when ``tickets_sold`` is not given — essential for break-even analysis,
    because a bigger jackpot attracts more co-winners."""

    DEFAULT_SOLD = 1_500_000

    def __init__(self, spec: GameSpec, popularity: PopularityModel | None = None, tax: TaxRule = DEFAULT_TAX, sales=None) -> None:
        self.spec = spec
        self.popularity = popularity or PopularityModel(spec)
        self.tax = tax
        self.sales = sales

    def _jackpots(self, jackpot1: float | None, jackpot2: float | None) -> tuple[float, float | None]:
        j1 = jackpot1 if jackpot1 is not None else self.spec.min_jackpots["jackpot1"]
        j2 = None
        if "jackpot2" in self.spec.min_jackpots:
            j2 = jackpot2 if jackpot2 is not None else self.spec.min_jackpots["jackpot2"]
        return float(j1), (float(j2) if j2 is not None else None)

    def _combo(self, ticket: tuple[int, ...] | list[int]) -> np.ndarray:
        try:
            validate_numbers(tuple(int(x) for x in ticket), self.spec)
        except ValueError as exc:
            raise DataValidationError(str(exc)) from exc
        return np.asarray(sorted(ticket), dtype=np.int64)[None, :]

    def _special(self, special: int | None) -> int | None:
        if not self.spec.separate_special:
            return None
        if special is None:
            raise DataValidationError(f"{self.spec.display_name} tickets need a special number 1..{self.spec.bonus_pool_size}")
        if not 1 <= int(special) <= int(self.spec.bonus_pool_size or 0):
            raise DataValidationError(f"special number must be in 1..{self.spec.bonus_pool_size}")
        return int(special)

    def resolve_sold(self, tickets_sold: int | None, jackpot1: float) -> tuple[int, str]:
        if tickets_sold is not None:
            return int(tickets_sold), "given"
        if self.sales is not None:
            return int(self.sales.predict(jackpot1)), "sales_model"
        return self.DEFAULT_SOLD, "default"

    def ticket_probability(self, combo: np.ndarray, special: int | None) -> float:
        if self.spec.separate_special:
            return float(self.popularity.ticket_probability_joint(combo, np.array([special]))[0])
        return float(self.popularity.ticket_probability(combo)[0])

    def _tiers(self, combo: np.ndarray, special: int | None, j1: float, j2: float | None, sold: int) -> tuple[list[TierEV], float, float]:
        p_c = self.ticket_probability(combo, special)
        lam1 = sold * p_c
        lam2 = sold * (p_c + 5 / self.spec.main_combinations)
        probs = self.spec.tier_probabilities
        cap = self.spec.jackpot_cap
        j2_eff = j2
        if cap and j2 is not None and j1 > cap:
            # Power 300-tỷ rule: if JP2 is won and JP1 is not, JP2 also receives JP1 − cap
            j2_eff = j2 + (j1 - cap) * float(np.exp(-sold / self.spec.main_combinations))
        tiers: list[TierEV] = []
        ev = 0.0
        for t in self.spec.tiers:
            if t.fixed_amount is not None:
                gross = float(t.fixed_amount)
                net = self.tax.after_tax(gross)
            elif t.name == "jackpot1":
                gross, net = j1, expected_net_share(j1, lam1, self.tax)
            else:
                gross, net = float(j2_eff or 0.0), expected_net_share(float(j2_eff or 0.0), lam2, self.tax)
            contrib = probs[t.name] * net
            ev += contrib
            tiers.append(TierEV(tier=t.name, probability=probs[t.name], gross_prize=gross, expected_net_prize_if_won=net, ev_contribution=contrib))
        return tiers, ev, p_c

    def evaluate(
        self,
        ticket: tuple[int, ...] | list[int],
        jackpot1: float | None = None,
        jackpot2: float | None = None,
        tickets_sold: int | None = None,
        special: int | None = None,
    ) -> EVBreakdown:
        combo = self._combo(ticket)
        sp = self._special(special)
        j1, j2 = self._jackpots(jackpot1, jackpot2)
        sold, src = self.resolve_sold(tickets_sold, j1)
        tiers, ev, p_c = self._tiers(combo, sp, j1, j2, sold)
        denom = self.spec.main_combinations * ((self.spec.bonus_pool_size or 1) if self.spec.separate_special else 1)
        return EVBreakdown(
            game=self.spec.code.value,
            ticket=[int(x) for x in combo[0]],
            special=sp,
            ticket_price=self.spec.ticket_price,
            jackpot1=j1,
            jackpot2=j2,
            tickets_sold=sold,
            tickets_sold_source=src,
            popularity_ratio=p_c * denom,
            expected_co_winners=sold * p_c,
            tiers=tiers,
            expected_value=ev,
            return_to_player=ev / self.spec.ticket_price,
            expected_loss=self.spec.ticket_price - ev,
            break_even_jackpot1=self.break_even_jackpot(ticket, jackpot2, sold, special=sp),
            break_even_jackpot1_with_sales_response=self.break_even_jackpot(ticket, jackpot2, None, special=sp) if self.sales is not None else None,
        )

    def break_even_jackpot(
        self, ticket: tuple[int, ...] | list[int], jackpot2: float | None = None, tickets_sold: int | None = 1_500_000, upper: float = 1e14, special: int | None = None
    ) -> float | None:
        """Lowest Jackpot 1 at which EV = ticket price (None if not reached below ``upper``).

        With ``tickets_sold=None`` and a sales model, sales respond to the jackpot —
        the realistic (and higher) break-even. EV is then not monotone in the jackpot (a
        convex sales curve extrapolated far beyond the data brings EV back down), so the
        first crossing is located on a log grid before refining with Brent's method."""

        def gap(j: float) -> float:
            sold = tickets_sold if tickets_sold is not None else self.resolve_sold(None, max(j, 1.0))[0]
            return self.evaluate_ev_only(ticket, j, jackpot2, sold, special=special) - self.spec.ticket_price

        lo = float(self.spec.min_jackpots["jackpot1"])
        if gap(lo) >= 0:
            return lo
        prev = lo
        for j in np.geomspace(lo, upper, 160)[1:]:
            if gap(float(j)) >= 0:
                return float(optimize.brentq(gap, prev, float(j), xtol=1e3))
            prev = float(j)
        return None

    def evaluate_ev_only(self, ticket: tuple[int, ...] | list[int], jackpot1: float, jackpot2: float | None, tickets_sold: int, special: int | None = None) -> float:
        combo = self._combo(ticket)
        _, j2 = self._jackpots(jackpot1, jackpot2)
        return self._tiers(combo, self._special(special) if self.spec.separate_special else None, float(jackpot1), j2, int(tickets_sold))[1]


# ------------------------------------------------------------------ optimiser
class OptimizedTicket(BaseModel):
    numbers: list[int]
    special: int | None = None
    popularity_ratio: float
    popularity_percentile: float  # share of uniformly random combos that are *more* popular
    expected_value: float
    return_to_player: float


class OptimizationResult(BaseModel):
    game: str
    tickets: list[OptimizedTicket]
    average_ticket_ev: float
    random_ticket_ev: float
    max_pairwise_overlap: int
    note: str


def optimize_tickets(
    calc: EVCalculator,
    n_tickets: int = 5,
    jackpot1: float | None = None,
    jackpot2: float | None = None,
    tickets_sold: int | None = 1_500_000,
    max_overlap: int = 2,
    exclude: tuple[int, ...] = (),
    iterations: int = 20_000,
    seed: int | None = None,
) -> OptimizationResult:
    """Simulated annealing: minimise Σ log popularity subject to pairwise overlap ≤ ``max_overlap``."""
    spec, model = calc.spec, calc.popularity
    n, k = spec.pool_size, spec.pick
    rng = np.random.default_rng(seed)
    allowed = np.array([x for x in range(1, n + 1) if x not in set(exclude)])
    if allowed.size < k:
        raise ValueError("too many excluded numbers")

    penalty = 10.0
    tickets = np.array([rng.choice(allowed, k, replace=False) for _ in range(n_tickets)])
    member = np.zeros((n_tickets, n + 1), dtype=np.int64)
    member[np.arange(n_tickets)[:, None], tickets] = 1
    logp = np.log(model.ticket_probability(np.sort(tickets, axis=1)))

    def overlap_excess(mem: np.ndarray) -> float:
        ov = mem @ mem.T
        return float(np.maximum(0, ov[np.triu_indices(len(mem), 1)] - max_overlap).sum())

    current = float(logp.sum() + penalty * overlap_excess(member))
    best, best_score = tickets.copy(), current
    t0, t1 = 1.0, 1e-3
    for it in range(iterations):
        temp = t0 * (t1 / t0) ** (it / max(1, iterations - 1))
        j = int(rng.integers(n_tickets))
        pos = int(rng.integers(k))
        old_num = int(tickets[j, pos])
        new_num = int(rng.choice(allowed[member[j, allowed] == 0]))
        # incremental update of ticket j only
        cand_row = tickets[j].copy()
        cand_row[pos] = new_num
        new_logp_j = float(np.log(model.ticket_probability(np.sort(cand_row)[None, :]))[0])
        mem_row = member[j].copy()
        mem_row[old_num], mem_row[new_num] = 0, 1
        others = np.delete(member, j, axis=0)
        old_ex = np.maximum(0, others @ member[j] - max_overlap).sum()
        new_ex = np.maximum(0, others @ mem_row - max_overlap).sum()
        s = current - logp[j] + new_logp_j + penalty * float(new_ex - old_ex)
        if s < current or rng.random() < np.exp((current - s) / temp):
            tickets[j] = cand_row
            member[j] = mem_row
            logp[j] = new_logp_j
            current = s
            if s < best_score - 1e-12:
                best, best_score = tickets.copy(), s
    best = np.sort(best, axis=1)

    ref = np.sort(np.argpartition(rng.random((20_000, n)), k, axis=1)[:, :k] + 1, axis=1)
    ref_ratio = np.sort(model.popularity_ratio(ref))
    j1, j2 = calc._jackpots(jackpot1, jackpot2)
    sold, _ = calc.resolve_sold(tickets_sold, j1)
    special = None
    if spec.separate_special:
        pi = model.special_probs if model.special_probs is not None else np.full(int(spec.bonus_pool_size or 1), 1.0)
        special = int(np.argmin(pi)) + 1  # least-picked special number
    rand_specials = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, 200) if spec.separate_special else [None] * 200
    ev_random = float(np.mean([calc.evaluate_ev_only(tuple(r), j1, j2, sold, special=int(sp) if sp is not None else None) for r, sp in zip(ref[:200], rand_specials)]))
    out = []
    for t in best:
        ratio = float(model.popularity_ratio(t[None, :])[0])
        ev = calc.evaluate_ev_only(tuple(t), j1, j2, sold, special=special)
        out.append(
            OptimizedTicket(
                numbers=[int(x) for x in t],
                special=special,
                popularity_ratio=ratio,
                popularity_percentile=float(1 - np.searchsorted(ref_ratio, ratio) / ref_ratio.size),
                expected_value=ev,
                return_to_player=ev / spec.ticket_price,
            )
        )
    overlaps = [len(np.intersect1d(best[a], best[b])) for a in range(len(best)) for b in range(a + 1, len(best))]
    return OptimizationResult(
        game=spec.code.value,
        tickets=out,
        average_ticket_ev=float(np.mean([t.expected_value for t in out])),
        random_ticket_ev=ev_random,
        max_pairwise_overlap=int(max(overlaps) if overlaps else 0),
        note=(
            "Win probability is identical for every combination; these tickets only reduce the expected "
            "number of co-winners who would split a jackpot. Return-to-player below 1.0 means a loss in expectation."
        ),
    )
