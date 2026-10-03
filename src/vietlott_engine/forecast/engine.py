"""Fixed-share Bayesian mixture of experts, its anytime-valid evidence, persistence, ledger.

Learning rule (Herbster–Warmuth fixed share): after draw t with expert likelihoods p_j(x_t),

    v_j ← (1−a) · v_j p_j(x_t) / Σ_i v_i p_i(x_t) + a · prior_j,

a = 1 / (10 × draws per year), fixed by design (about one switch per ten years): the mixture
can still move to another expert if the machines change, at a cost of ≈ log(1/a) nats of
evidence per switch. On the real data, 2 years and no switching lead to the same conclusions
for every product, and the two findings (Max 3D, Max 3D Pro) also clear 60 = 20 × 3 settings.
The mixture's forecast is Σ_j v_j p_j(·), a proper law built
from past draws only, so its cumulative likelihood ratio against the fair-machine law,

    W_T = Π_t Σ_j v_j p_j(x_t) / p₀(x_t),

is a non-negative martingale under H0 and P(sup_T W_T ≥ 1/α) ≤ α (Ville). W is reported in
log10; "evidence" means W has reached 1/α = 20 (log10 1.30) at some point.

Picks rule scored on every past draw: set games — the k numbers with the highest
mixture-averaged normalised weight (cheap, close to the exact inclusion probabilities); digit
games — the most likely symbol at each position. The forecast for the next draw, and the
ledger, use the exact inclusion probabilities. Under H0 any predictable picks hit k²/n numbers (set) or
m·L/A symbols (digit) per draw on average, so "hits / expected" is a fair score.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import product as iproduct
from math import comb, log, sqrt
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from vietlott_engine.core.products import PRODUCT_INFO, ProductCode, get_product
from vietlott_engine.forecast.data import CONFIGS, DigitSpec, Series, SetSpec
from vietlott_engine.forecast.experts import DigitExperts, DigitTilts, SetExperts, SetTilts, log_comb
from vietlott_engine.game_theory.popularity import divide_out, elementary_symmetric

VERSION = "3.5.0"
ALPHA = 0.05
RECENT = 500
PATH_POINTS = 800


# ============================================================================ outputs
class ExpertWeight(BaseModel):
    name: str
    weight: float
    log10_wealth_alone: float = Field(description="log10 likelihood ratio of this expert alone vs the fair machine, over all draws seen")


class Evidence(BaseModel):
    draws: int
    log10_wealth: float
    max_log10_wealth: float
    threshold_log10: float
    anytime_p_value: float
    found: bool
    valid: bool = True
    text: str


class PickScore(BaseModel):
    draws: int
    hits_per_draw: float
    expected_per_draw: float
    ratio: float
    z: float
    recent_draws: int
    recent_hits_per_draw: float


class NumberProb(BaseModel):
    symbol: str
    p: float
    lift: float


class KenoTicket(BaseModel):
    bac: int
    numbers: list[int]
    rtp_model: float
    rtp_fair: float
    p_any_prize_model: float
    p_any_prize_fair: float


class SetForecast(BaseModel):
    numbers: list[NumberProb]
    ticket: list[int]
    p_ticket_model: float
    p_ticket_fair: float
    ticket_lift: float
    match_dist_model: list[float]
    match_dist_fair: list[float]
    keno: list[KenoTicket] = []


class PositionForecast(BaseModel):
    label: str
    symbols: list[NumberProb]


class BetForecast(BaseModel):
    bet: str
    p_model: float
    p_fair: float
    rtp_model: float
    rtp_fair: float


class DigitForecast(BaseModel):
    positions: list[PositionForecast]
    top_numbers: list[NumberProb]
    p_top_appears_model: float
    p_top_appears_fair: float
    bets: list[BetForecast] = []


class ComponentForecast(BaseModel):
    name: str
    kind: str
    evidence: Evidence
    experts: list[ExpertWeight]
    uniform_weight: float
    picks: list[int] | list[str]
    pick_score: PickScore
    set: SetForecast | None = None
    digit: DigitForecast | None = None


class ForecastReport(BaseModel):
    product: str
    display_name: str
    version: str = VERSION
    made_at: str
    draws: int
    last_id: int | None
    last_date: str | None
    target_id: int | None
    evidence: Evidence
    components: list[ComponentForecast]
    verdict: str
    digest: str = ""
    recorded: bool | None = None  # written to the ledger before its draw (None: not asked)
    record_note: str = ""


# ============================================================================ helpers
def vn(x: float, digits: int = 2) -> str:
    """Vietnamese number format: 1.234,56."""
    return f"{x:,.{digits}f}".replace(",", "\0").replace(".", ",").replace("\0", ".")


def vn_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def inclusion_probabilities(w: np.ndarray, k: int) -> np.ndarray:
    """π_i = w_i e_{k−1}(w_{−i}) / e_k(w) for CB weights ``w`` (J, n)."""
    e = elementary_symmetric(w, k)
    pi = np.empty_like(w)
    for i in range(w.shape[1]):
        h = divide_out(e, w[:, i])
        pi[:, i] = w[:, i] * h[:, k - 1] / e[:, k]
    return pi


def match_distribution(w: np.ndarray, ticket: np.ndarray, k: int) -> np.ndarray:
    """(J, s+1): P(the ticket's s numbers contain r of the k drawn), r = 0..s, under CB weights."""
    n = w.shape[1]
    s = len(ticket)
    mask = np.zeros(n, dtype=bool)
    mask[ticket] = True
    e_in = elementary_symmetric(w[:, mask], min(s, k))
    e_out = elementary_symmetric(w[:, ~mask], k)
    e_all = elementary_symmetric(w, k)[:, k]
    out = np.zeros((w.shape[0], s + 1))
    for r in range(0, min(s, k) + 1):
        if 0 <= k - r <= n - s:
            out[:, r] = e_in[:, r] * e_out[:, k - r] / e_all
    return out


def hypergeom(n: int, k: int, s: int) -> np.ndarray:
    return np.array([comb(s, r) * comb(n - s, k - r) / comb(n, k) if 0 <= k - r <= n - s else 0.0 for r in range(s + 1)])


def _evidence(draws: int, lw: float, mlw: float, window: dict | None = None) -> Evidence:
    thr = log(1 / ALPHA, 10)
    valid = window is None
    found = valid and mlw >= thr
    if not valid:
        text = (
            f"Không dùng làm bằng chứng: mô hình học từ {vn_int(window['last'])} kỳ cuối do người dùng chọn (--last). "  # type: ignore[index]
            "E-value chỉ hợp lệ khi điểm bắt đầu được định trước khi xem dữ liệu; chạy `forecast fit` không có --last để học toàn bộ lịch sử."
        )
    elif found:
        text = f"Có bằng chứng kết quả lệch khỏi máy quay công bằng theo hướng mô hình đoán: e-value đạt 10^{vn(mlw)} ≥ 1/α = 20 (α = 5%, hợp lệ ở mọi thời điểm)."
    else:
        text = (
            f"Chưa có bằng chứng kết quả lệch khỏi máy quay công bằng: e-value cao nhất 10^{vn(mlw)}, hiện 10^{vn(lw)} (ngưỡng 10^{vn(thr)}). "
            "Xác suất của mô hình không đáng tin hơn ngẫu nhiên."
        )
    return Evidence(draws=draws, log10_wealth=round(lw, 4), max_log10_wealth=round(mlw, 4), threshold_log10=round(thr, 4), anytime_p_value=float(min(1.0, 10 ** (-max(mlw, 0.0)))), found=found, valid=valid, text=text)


def _jsonable(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _from_jsonable(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.asarray(obj["__nd__"], dtype=obj.get("dtype", "float64"))
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


# ============================================================================ component
class Component:
    def __init__(self, name: str, spec: SetSpec | DigitSpec, batch: int, switch_rate: float) -> None:
        self.name, self.spec, self.switch = name, spec, switch_rate
        if isinstance(spec, SetSpec):
            self.groups = [SetExperts(spec, batch), SetTilts(spec)]
            self.null = -log_comb(spec.n, spec.k) - (log(spec.n - spec.k) if spec.bonus_same_drum else 0.0)
            p = spec.k / spec.n
            self.expected = spec.k * p
            self.variance = spec.k * p * (1 - p) * (spec.n - spec.k) / (spec.n - 1)
        else:
            self.groups = [DigitExperts(spec), DigitTilts(spec)]
            self.null = -spec.per_draw * spec.positions * log(spec.alphabet)
            q = 1 / spec.alphabet
            self.expected = spec.per_draw * spec.positions * q
            self.variance = spec.per_draw * spec.positions * q * (1 - q)
        self.names = [n for g in self.groups for n in g.names]
        self.j_ad = len(self.groups[0].names)
        j = len(self.names)
        prior = np.empty(j)
        prior[0] = 0.25  # the fair-machine expert
        prior[1 : self.j_ad] = 0.375 / (self.j_ad - 1)
        prior[self.j_ad :] = 0.375 / (j - self.j_ad)
        self.prior = prior

    def init_state(self) -> dict:
        return {
            "groups": [g.init_state() for g in self.groups], "v": self.prior.copy(), "log_wealth": 0.0, "max_log_wealth": 0.0,
            "cum_ll": np.zeros(len(self.names)), "draws": 0, "hits": 0.0, "recent": [], "path": [], "stride": 1,
        }

    def process(self, st: dict, obs: dict, ids: np.ndarray) -> tuple[np.ndarray, dict]:
        t_len = len(ids)
        lls, preds, groups = [], [], []
        for g, gs in zip(self.groups, st["groups"]):
            ll, pred, gs2 = g.run(gs, obs)
            lls.append(ll)
            preds.append(pred)
            groups.append(gs2)
        big_ll = np.hstack(lls)
        p_ad, m_tilt = preds[0], self.groups[1].constant_pred
        mx = big_ll.max(axis=1)
        e = np.exp(big_ll - mx[:, None])
        v = st["v"].copy()
        a, prior, j_ad = self.switch, self.prior, self.j_ad
        incr = np.empty(t_len)
        hits = np.empty(t_len)
        is_set = isinstance(self.spec, SetSpec)
        if is_set:
            target = np.asarray(obs["X"], dtype=np.float64)
            k = self.spec.k
        else:
            target = np.asarray(obs["C"], dtype=np.float64)
            big_l, big_a = self.spec.positions, self.spec.alphabet
            rows = np.arange(big_l)
        lw, mlw = float(st["log_wealth"]), float(st["max_log_wealth"])
        path, stride, draws = list(st["path"]), int(st["stride"]), int(st["draws"])
        for t in range(t_len):
            score = v[:j_ad] @ p_ad[t] + v[j_ad:] @ m_tilt
            if is_set:
                pick = np.argpartition(-score, k - 1)[:k]
                hits[t] = target[t, pick].sum()
            else:
                pick = score.reshape(big_l, big_a).argmax(axis=1)
                hits[t] = target[t, rows, pick].sum()
            p = v * e[t]
            s = p.sum()
            incr[t] = log(s) + mx[t] - self.null
            v = (1 - a) * p / s + a * prior
            lw += incr[t]
            mlw = max(mlw, lw)
            draws += 1
            if draws % stride == 0:
                path.append([int(ids[t]), round(lw / log(10), 4)])
                if len(path) > PATH_POINTS:
                    path = path[1::2]
                    stride *= 2
        new = {
            "groups": groups, "v": v, "log_wealth": lw, "max_log_wealth": mlw,
            "cum_ll": st["cum_ll"] + (big_ll - self.null).sum(axis=0), "draws": draws,
            "hits": float(st["hits"] + hits.sum()), "recent": (list(st["recent"]) + hits.tolist())[-RECENT:],
            "path": path, "stride": stride,
        }
        return incr, new

    # ------------------------------------------------------------------ next draw
    def predictions(self, st: dict) -> np.ndarray:
        empty = {"X": np.zeros((0, self.spec.n), dtype=bool), "bonus": np.zeros(0, dtype=np.int16)} if isinstance(self.spec, SetSpec) else {
            "C": np.zeros((0, self.spec.positions, self.spec.alphabet))
        }
        _, pred, _ = self.groups[0].run(st["groups"][0], empty)
        return np.vstack([pred[0], self.groups[1].constant_pred])

    def picks(self, st: dict, preds: np.ndarray) -> np.ndarray:
        score = st["v"] @ preds
        if isinstance(self.spec, SetSpec):
            return np.sort(np.argpartition(-score, self.spec.k - 1)[: self.spec.k])
        return score.reshape(self.spec.positions, self.spec.alphabet).argmax(axis=1)

    def forecast(self, st: dict, window: dict | None = None) -> ComponentForecast:
        preds = self.predictions(st)
        v = st["v"]
        ln10 = log(10)
        ev = _evidence(int(st["draws"]), st["log_wealth"] / ln10, st["max_log_wealth"] / ln10, window)
        order = np.argsort(-v)
        experts = [ExpertWeight(name=self.names[i], weight=round(float(v[i]), 6), log10_wealth_alone=round(float(st["cum_ll"][i] / ln10), 3)) for i in order[:8]]
        d = max(int(st["draws"]), 1)
        rec = np.asarray(st["recent"], dtype=np.float64)
        ps = PickScore(
            draws=int(st["draws"]), hits_per_draw=round(st["hits"] / d, 5), expected_per_draw=round(self.expected, 5),
            ratio=round(st["hits"] / d / self.expected, 4) if st["draws"] else 1.0,
            z=round((st["hits"] - self.expected * st["draws"]) / sqrt(self.variance * d), 3) if st["draws"] else 0.0,
            recent_draws=int(rec.size), recent_hits_per_draw=round(float(rec.mean()), 4) if rec.size else 0.0,
        )
        out = ComponentForecast(name=self.name, kind=self.spec.kind, evidence=ev, experts=experts, uniform_weight=round(float(v[0]), 6), picks=[], pick_score=ps)
        if isinstance(self.spec, SetSpec):
            out.set = self._set_forecast(v, preds)
            out.picks = list(out.set.ticket)
        else:
            off = self.spec.symbol_offset
            out.picks = [str(int(a) + off) for a in self.picks(st, preds)]
            out.digit = self._digit_forecast(v, preds)
        return out

    def _set_forecast(self, v: np.ndarray, w: np.ndarray) -> SetForecast:
        n, k = self.spec.n, self.spec.k
        pi = v @ inclusion_probabilities(w, k)
        order = np.argsort(-pi, kind="stable")
        ticket = np.sort(order[:k])  # the k most likely numbers (exact inclusion probabilities)
        p0 = k / n
        nums = [NumberProb(symbol=str(i + 1), p=round(float(pi[i]), 6), lift=round(float(pi[i] / p0), 5)) for i in np.argsort(-pi)]
        logp = np.log(w[:, ticket]).sum(axis=1) - np.log(elementary_symmetric(w, k)[:, k])
        p_t = float(v @ np.exp(logp))
        fair = 1 / comb(n, k)
        md = v @ match_distribution(w, ticket, k)
        sf = SetForecast(
            numbers=nums, ticket=[int(i) + 1 for i in ticket], p_ticket_model=p_t, p_ticket_fair=fair, ticket_lift=round(p_t / fair, 6),
            match_dist_model=[round(float(x), 8) for x in md], match_dist_fair=[round(float(x), 8) for x in hypergeom(n, k, k)],
        )
        if n == 80 and k == 20:
            from vietlott_engine.game_theory.fastgames import KENO_TABLE

            for bac in range(1, 11):
                tk = np.sort(order[:bac])
                dm, df = v @ match_distribution(w, tk, k), hypergeom(n, k, bac)
                table = KENO_TABLE[bac]
                sf.keno.append(KenoTicket(
                    bac=bac, numbers=[int(i) + 1 for i in tk],
                    rtp_model=round(sum(dm[r] * table.get(r, 0) for r in range(bac + 1)) / 10_000, 5),
                    rtp_fair=round(sum(df[r] * table.get(r, 0) for r in range(bac + 1)) / 10_000, 5),
                    p_any_prize_model=round(sum(dm[r] for r in range(bac + 1) if table.get(r, 0) > 0), 6),
                    p_any_prize_fair=round(sum(df[r] for r in range(bac + 1) if table.get(r, 0) > 0), 6),
                ))
        return sf

    def _digit_forecast(self, v: np.ndarray, preds: np.ndarray) -> DigitForecast:
        sp = self.spec
        big_l, big_a, m, off = sp.positions, sp.alphabet, sp.per_draw, sp.symbol_offset
        q = preds.reshape(-1, big_l, big_a)
        qm = np.einsum("j,jla->la", v, q)
        positions = [
            PositionForecast(label=sp.position_labels[pos], symbols=[NumberProb(symbol=str(a + off), p=round(float(qm[pos, a]), 6), lift=round(float(qm[pos, a] * big_a), 5)) for a in np.argsort(-qm[pos])])
            for pos in range(big_l)
        ]
        combos = np.array(list(iproduct(range(big_a), repeat=big_l)))  # (A^L, L)
        logq = np.log(q)
        lp = sum(logq[:, pos, combos[:, pos]] for pos in range(big_l))  # (J, A^L)
        pj = np.exp(lp)
        pm = v @ pj
        top = np.argsort(-pm)[:5]
        fmt = (lambda c: "-".join(str(x + off) for x in c)) if off else (lambda c: "".join(str(x) for x in c))
        fair = big_a ** (-big_l)
        out = DigitForecast(
            positions=positions,
            top_numbers=[NumberProb(symbol=fmt(combos[i]), p=float(pm[i]), lift=round(float(pm[i] / fair), 5)) for i in top],
            p_top_appears_model=float(v @ (1 - (1 - pj[:, top[0]]) ** m)), p_top_appears_fair=float(1 - (1 - fair) ** m),
        )
        if big_a == 6 and big_l == 3:  # Bingo18: price 10.000 đ
            from vietlott_engine.game_theory.fastgames import BINGO_PRICE, BINGO_SUM_PAYOUT

            sums = (combos + 1).sum(axis=1)
            triple = (combos[:, 0] == combos[:, 1]) & (combos[:, 1] == combos[:, 2])
            bets = [(f"Tổng {s}", sums == s, BINGO_SUM_PAYOUT[s]) for s in range(3, 19)]
            bets += [("Nhỏ (3–9)", sums <= 9, 15_000), ("Hòa (10–11)", (sums >= 10) & (sums <= 11), 20_000), ("Lớn (12–18)", sums >= 12, 15_000), ("Bộ ba bất kỳ", triple, 200_000)]
            rows = []
            for name, mask, pay in bets:
                p_m, p_f = float(pm[mask].sum()), float(mask.mean())
                rows.append(BetForecast(bet=name, p_model=round(p_m, 6), p_fair=round(p_f, 6), rtp_model=round(p_m * pay / BINGO_PRICE, 5), rtp_fair=round(p_f * pay / BINGO_PRICE, 5)))
            out.bets = sorted(rows, key=lambda r: -r.rtp_model)
        return out


# ============================================================================ forecaster
class Forecaster:
    """All components of one product, their state, and the ledger of issued forecasts."""

    SWITCH_YEARS = 10.0

    def __init__(self, product: str | ProductCode, keep_trace: bool = False) -> None:
        self.product = get_product(product)
        self.keep_trace = keep_trace
        self.trace: list[tuple[np.ndarray, np.ndarray]] = []  # (draw ids, log10 wealth) at full resolution, when asked
        cfg = CONFIGS[self.product]
        switch = 1.0 / (self.SWITCH_YEARS * cfg.draws_per_year)
        self.components = {name: Component(name, spec, cfg.logistic_batch, switch) for name, spec in cfg.components.items()}
        self.state: dict = {
            "product": self.product.value, "version": VERSION, "last_id": None, "last_date": None, "draws": 0,
            "log_wealth": 0.0, "max_log_wealth": 0.0, "components": {n: c.init_state() for n, c in self.components.items()},
        }

    # ------------------------------------------------------------------ learning
    def update(self, series: Series, chunk: int = 2000) -> int:
        """Learn from every draw newer than the last one seen; returns how many."""
        if series.product != self.product:
            raise ValueError(f"series is {series.product.value}, forecaster is {self.product.value}")
        new = series.after(self.state["last_id"])
        for start in range(0, len(new), chunk):
            part = Series(new.product, new.draw_ids[start : start + chunk], new.dates[start : start + chunk], {c: {k: v[start : start + chunk] for k, v in o.items()} for c, o in new.obs.items()})
            total = np.zeros(len(part))
            for name, comp in self.components.items():
                incr, self.state["components"][name] = comp.process(self.state["components"][name], part.obs[name], part.draw_ids)
                total += incr
            lw = self.state["log_wealth"] + np.cumsum(total)
            if self.keep_trace:
                self.trace.append((part.draw_ids.copy(), lw / log(10)))
            self.state["max_log_wealth"] = float(max(self.state["max_log_wealth"], lw.max()))
            self.state["log_wealth"] = float(lw[-1])
        if len(new):
            self.state["last_id"] = int(new.draw_ids[-1])
            self.state["last_date"] = str(new.dates[-1])
            self.state["draws"] = int(self.state["draws"] + len(new))
        return len(new)

    # ------------------------------------------------------------------ forecasting
    def forecast(self) -> ForecastReport:
        window = self.state.get("window")
        comps = [c.forecast(self.state["components"][n], window) for n, c in self.components.items()]
        if self.product == ProductCode.MAX3D and comps[0].digit is not None:
            from vietlott_engine.game_theory.max3d import get_product as max3d_product
            from vietlott_engine.game_theory.max3d import play_distribution

            top = comps[0].digit.top_numbers[0]
            fair = play_distribution(max3d_product("max3d"), [int(top.symbol)]).return_to_player
            comps[0].digit.bets.append(BetForecast(bet=f"Max 3D, số {top.symbol} (RTP mô hình ≈ RTP × hệ số)", p_model=top.p, p_fair=0.001, rtp_model=round(fair * top.lift, 5), rtp_fair=round(fair, 5)))
        if self.product == ProductCode.MAX3D_PRO and comps[0].digit is not None:
            from vietlott_engine.game_theory.max3d import get_product as max3d_product
            from vietlott_engine.game_theory.max3d import play_distribution

            a, b = comps[0].digit.top_numbers[:2]
            fair = play_distribution(max3d_product("max3dpro"), [int(a.symbol), int(b.symbol)]).return_to_player
            comps[0].digit.bets.append(BetForecast(bet=f"Max 3D Pro, cặp {a.symbol} {b.symbol} (RTP mô hình ≤ RTP × hệ số₁ × hệ số₂)", p_model=a.p * b.p, p_fair=1e-6, rtp_model=round(fair * a.lift * b.lift, 5), rtp_fair=round(fair, 5)))
        ln10 = log(10)
        ev = _evidence(int(self.state["draws"]), self.state["log_wealth"] / ln10, self.state["max_log_wealth"] / ln10, window)
        verdict = self._verdict(ev, comps)
        rep = ForecastReport(
            product=self.product.value, display_name=PRODUCT_INFO[self.product].display_name, made_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            draws=int(self.state["draws"]), last_id=self.state["last_id"], last_date=self.state["last_date"],
            target_id=(self.state["last_id"] + 1) if self.state["last_id"] is not None and self.product != ProductCode.MAX4D else None,  # Max 4D: discontinued
            evidence=ev, components=comps, verdict=verdict,
        )
        rep.digest = hashlib.sha256(rep.model_dump_json(exclude={"digest", "made_at"}).encode()).hexdigest()[:16]
        return rep

    def _verdict(self, ev: Evidence, comps: list[ComponentForecast]) -> str:
        if not ev.valid:
            head = "Cửa sổ học do người dùng chọn (--last): không dùng làm bằng chứng; các hệ số × dưới đây là mô hình tự đánh giá."
        elif ev.found:
            head = f"Mô hình tự học ĐÃ tìm thấy độ lệch có ý nghĩa thống kê so với máy quay công bằng (e-value cao nhất 10^{vn(ev.max_log10_wealth)} ≥ 20, α = 5%); lợi thế nhỏ, xem RTP."
        else:
            head = (
                f"Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^{vn(ev.max_log10_wealth)} < 20): "
                "mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận."
            )
        parts, rtps = [], []
        for c in comps:
            if c.set is not None:
                sf = c.set
                if len(sf.ticket) <= 6:  # Keno tickets are judged per bậc below
                    parts.append(f"bộ đề xuất {' '.join(f'{x:02d}' for x in sf.ticket)}: P(trúng cả bộ) theo mô hình 1/{vn_int(1 / sf.p_ticket_model)} (×{vn(sf.ticket_lift, 3)}; ngẫu nhiên 1/{vn_int(1 / sf.p_ticket_fair)})")
                if sf.keno:
                    b = max(sf.keno, key=lambda x: x.rtp_model)
                    parts.append(f"vé Keno tốt nhất theo mô hình: bậc {b.bac}, RTP {vn(b.rtp_model, 4)} (ngẫu nhiên {vn(b.rtp_fair, 4)})")
                    rtps += [x.rtp_model for x in sf.keno]
            if c.digit is not None:
                df = c.digit
                strongest = max(((p.label, s) for p in df.positions for s in p.symbols), key=lambda x: x[1].lift)
                parts.append(f"mạnh nhất là {strongest[1].symbol} ở vị trí {strongest[0]} (×{vn(strongest[1].lift, 3)}); số đề xuất {df.top_numbers[0].symbol} ×{vn(df.top_numbers[0].lift, 3)}")
                if df.bets:
                    b = max(df.bets, key=lambda x: x.rtp_model)
                    parts.append(f"cửa tốt nhất theo mô hình: {b.bet}, RTP {vn(b.rtp_model, 4)} (ngẫu nhiên {vn(b.rtp_fair, 4)})")
                    rtps += [x.rtp_model for x in df.bets]
        tail = " Mọi cửa vẫn có kỳ vọng âm (RTP < 1)." if rtps and max(rtps) < 1 else ""
        body = "; ".join(parts)
        return head + " " + body[:1].upper() + body[1:] + "." + tail

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return _jsonable(self.state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Forecaster":
        data = _from_jsonable(json.loads(path.read_text(encoding="utf-8")))
        f = cls(data["product"])
        if data.get("version") != VERSION:
            raise ValueError(f"state {path} is from version {data.get('version')}; refit with `forecast fit`")
        for comp in data["components"].values():
            for g in comp["groups"]:
                for key in ("seen", "m10", "m100", "c", "acc_n"):
                    if key in g:
                        g[key] = float(g[key])
        f.state = data
        return f


def state_path(directory: Path, product: ProductCode) -> Path:
    return Path(directory) / f"{product.value}.json"


def load_or_new(directory: Path, product: ProductCode) -> Forecaster:
    p = state_path(directory, product)
    return Forecaster.load(p) if p.exists() else Forecaster(product)


# ============================================================================ ledger
LEDGER = "ledger.jsonl"


def record(directory: Path, rep: ForecastReport, pre_draw: bool = True, target_time: str | None = None) -> dict:
    from vlm.forecast.service import forecast_guard
    with forecast_guard(directory):
        return _record_unlocked(directory, rep, pre_draw, target_time)


def _record_unlocked(directory: Path, rep: ForecastReport, pre_draw: bool = True, target_time: str | None = None) -> dict:
    """Append a forecast *before* its draw: what was predicted, for which draw, when.

    ``pre_draw=False`` marks an entry forced in after the draw may have started; the scoreboard
    leaves such entries out."""
    entry = {
        "product": rep.product, "made_at": rep.made_at, "based_on_id": rep.last_id, "based_on_date": rep.last_date, "target_id": rep.target_id, "digest": rep.digest,
        "evidence_valid": rep.evidence.valid, "pre_draw": bool(pre_draw), "target_draw_time": target_time,
        "log10_wealth": rep.evidence.log10_wealth,
        "picks": {c.name: c.picks for c in rep.components},
        "ticket": next((c.set.ticket for c in rep.components if c.set is not None), None),
        "top": {c.name: ([x.symbol for x in c.set.numbers[:10]] if c.set else [x.symbol for x in c.digit.top_numbers]) for c in rep.components},  # type: ignore[union-attr]
        "score": None,
    }
    path = Path(directory) / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # the same forecast for the same draw is recorded once
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old = json.loads(line)
                if (old["product"], old["target_id"], old["digest"]) == (entry["product"], entry["target_id"], entry["digest"]):
                    return old
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _score_entry(entry: dict, series: Series) -> dict | None:
    idx = np.flatnonzero(series.draw_ids == entry["target_id"])
    if not idx.size:
        return None
    t = int(idx[0])
    cfg = CONFIGS[get_product(entry["product"])]
    out: dict = {"draw_id": entry["target_id"], "date": str(series.dates[t]), "components": {}}
    for name, spec in cfg.components.items():
        picks = entry["picks"].get(name, [])
        obs = series.obs[name]
        if isinstance(spec, SetSpec):
            drawn = set((np.flatnonzero(obs["X"][t]) + 1).tolist())
            hits = len(drawn & set(picks))
            out["components"][name] = {"drawn": sorted(drawn), "hits": hits, "expected": spec.k * spec.k / spec.n}
        else:
            c = obs["C"][t]
            hits = int(sum(c[pos, int(s) - spec.symbol_offset] for pos, s in enumerate(picks)))
            out["components"][name] = {"hits": hits, "expected": spec.per_draw * spec.positions / spec.alphabet}
    return out


def score_ledger(directory: Path, series: Series) -> list[dict]:
    from vlm.forecast.service import forecast_guard
    with forecast_guard(directory):
        return _score_ledger_unlocked(directory, series)


def _score_ledger_unlocked(directory: Path, series: Series) -> list[dict]:
    """Score every recorded forecast of this product whose target draw has arrived."""
    path = Path(directory) / LEDGER
    if not path.exists():
        return []
    entries = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    scored = []
    for e in entries:
        if e["product"] == series.product.value and e.get("score") is None and e.get("target_id") is not None:
            s = _score_entry(e, series)
            if s is not None:
                e["score"] = s
                scored.append(e)
    if scored:
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    return scored


def scoreboard(directory: Path, product: ProductCode | None = None) -> list[dict]:
    from vlm.forecast.service import forecast_guard
    with forecast_guard(directory):
        return _scoreboard_unlocked(directory, product)


def _scoreboard_unlocked(directory: Path, product: ProductCode | None = None) -> list[dict]:
    """Live track record per product: forecasts recorded before their draw, then scored."""
    path = Path(directory) / LEDGER
    if not path.exists():
        return []
    rows: dict[str, dict] = {}
    seen: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if (product is not None and e["product"] != product.value) or (e["product"], e["target_id"]) in seen or e.get("pre_draw") is False:
            continue  # only the first forecast recorded before its draw counts
        seen.add((e["product"], e["target_id"]))
        r = rows.setdefault(e["product"], {"product": e["product"], "recorded": 0, "scored": 0, "hits": 0.0, "expected": 0.0, "last": None})
        r["recorded"] += 1
        if e.get("score"):
            r["scored"] += 1
            for c in e["score"]["components"].values():
                r["hits"] += c["hits"]
                r["expected"] += c["expected"]
            r["last"] = e["score"]
    for r in rows.values():
        r["ratio"] = round(r["hits"] / r["expected"], 4) if r["expected"] else None
    return list(rows.values())


def refresh(product: str | ProductCode, series: Series, directory: Path, refit: bool = False, last: int | None = None) -> tuple[Forecaster, int, list[dict]]:
    from vlm.forecast.service import forecast_guard
    with forecast_guard(directory):
        return _refresh_unlocked(product, series, directory, refit, last)


def _refresh_unlocked(product: str | ProductCode, series: Series, directory: Path, refit: bool = False, last: int | None = None) -> tuple[Forecaster, int, list[dict]]:
    """Load the saved model (or start one), learn every new draw, score the ledger, save."""
    code = get_product(product)
    f = None
    if not refit:
        try:
            f = load_or_new(directory, code)
        except (ValueError, KeyError, json.JSONDecodeError):
            f = None  # state from another version, or damaged: learn again from the data
    if f is None:
        f = Forecaster(code)
        if last:
            series = series.tail(last)
            f.state["window"] = {"last": int(last), "first_id": int(series.draw_ids[0]) if len(series) else None}
    learned = f.update(series)
    scored = score_ledger(directory, series)
    f.save(state_path(directory, code))
    return f, learned, scored
