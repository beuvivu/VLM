"""Experts: each turns *past* draws into a full probability law for the next draw.

Every group processes a chunk of T draws and returns

* ``ll``   (T, J)        log-probability each expert gave to each draw, issued before it;
* ``pred`` (T+1, J, D)   each expert's normalised prediction before each draw, the last row
                         being the forecast for the next, unseen draw (set games: CB weights
                         summing to 1 over the n numbers; digit games: per-position
                         probabilities flattened to L·A). ``None`` for fixed experts, whose
                         prediction is ``constant_pred``.

State is carried between chunks, so running a history in one chunk or many gives the same
result (tested) — that is what lets ``forecast update`` learn draw by draw.

k-of-n draws use the conditional-Bernoulli law P(S) = Π_{i∈S} w_i / e_k(w) (exactly uniform
when all w_i are equal); Power's bonus ball, drawn from the remaining n−k numbers, has
P(b | S) = w_b / Σ_{i∉S} w_i. Digit games use independent categorical positions.
"""

from __future__ import annotations

from math import lgamma, log

import numpy as np
from scipy.signal import lfilter

from vietlott_engine.forecast.data import DigitSpec, SetSpec
from vietlott_engine.game_theory.popularity import elementary_symmetric

TILT_EPS = (-0.25, -0.1, 0.1, 0.25)


def log_comb(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _running_sum(x: np.ndarray, start: np.ndarray) -> np.ndarray:
    """(T+1, …): value before each of the T rows, then after the last."""
    c = np.cumsum(np.asarray(x, dtype=np.float64), axis=0)
    return np.concatenate([start[None], start[None] + c], axis=0)


def _running_ewma(x: np.ndarray, s0: np.ndarray, lam: float) -> np.ndarray:
    """(T+1, …) exponentially weighted sums s ← λ·s + x, before each row then after the last."""
    if len(x) == 0:
        return s0[None].astype(np.float64)
    y, _ = lfilter([1.0], [1.0, -lam], np.asarray(x, dtype=np.float64), axis=0, zi=(lam * np.asarray(s0, dtype=np.float64))[None])
    return np.concatenate([np.asarray(s0, dtype=np.float64)[None], y], axis=0)


def _lam(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


# ====================================================================== k-of-n games
class SetExperts:
    """Adaptive experts for k distinct numbers out of n (+ same-drum bonus)."""

    HALF_LIVES = (10, 100)

    def __init__(self, spec: SetSpec, batch: int = 20, lr: float = 0.05, l2: float = 1e-4) -> None:
        self.spec, self.batch, self.lr, self.l2 = spec, batch, lr, l2
        self.p0 = spec.k / spec.n
        self.names = [
            "uniform", "hot_all_a50", "hot_all_a500", "cold_all_a50", "cold_all_a500",
            "hot_ewma10", "hot_ewma100", "cold_ewma10", "cold_ewma100",
            "overdue", "anti_overdue", "repeat", "anti_repeat", "logistic",
        ]
        self.feature_names = ["freq_all", "freq_ewma10", "freq_ewma100", "gap_z", "in_last", "in_last2"]
        self.constant_pred = None

    def init_state(self) -> dict:
        n = self.spec.n
        z = np.zeros(n)
        return {
            "seen": 0.0, "count": z.copy(), "s10": z.copy(), "m10": 0.0, "s100": z.copy(), "m100": 0.0,
            "last": np.full(n, -1.0), "prev": z.copy(), "prev2": z.copy(),
            "theta": np.zeros(len(self.feature_names)), "c": float(_logit(self.p0)),
            "G": np.zeros(len(self.feature_names) + 1), "acc_g": np.zeros(len(self.feature_names) + 1), "acc_n": 0.0,
        }

    # ------------------------------------------------------------------ features
    def _features(self, st: dict, x: np.ndarray) -> tuple[dict, dict]:
        t = len(x)
        p0 = self.p0
        xf = x.astype(np.float64)
        seen = st["seen"] + np.arange(t + 1, dtype=np.float64)
        count = _running_sum(xf, st["count"])
        ew = {}
        for h in self.HALF_LIVES:
            lam = _lam(h)
            ew[h] = (_running_ewma(xf, st[f"s{h}"], lam), _running_ewma(np.ones(t), np.array(st[f"m{h}"]), lam))
        g = st["seen"] + np.arange(t, dtype=np.float64)
        last = np.maximum.accumulate(np.concatenate([st["last"][None], np.where(x, g[:, None], -1.0)], axis=0), axis=0)
        gap = seen[:, None] - last
        prev_all = np.concatenate([st["prev2"][None], st["prev"][None], xf], axis=0)
        f = {
            "seen": seen, "count": count, "gap": gap, "prev": prev_all[1:], "prev2": prev_all[:-1],
            **{f"s{h}": ew[h][0] for h in self.HALF_LIVES}, **{f"m{h}": ew[h][1] for h in self.HALF_LIVES},
        }
        new = dict(st)
        new.update({
            "seen": float(seen[-1]), "count": count[-1], "last": last[-1], "prev": prev_all[-1], "prev2": prev_all[-2],
            **{f"s{h}": ew[h][0][-1] for h in self.HALF_LIVES}, **{f"m{h}": float(ew[h][1][-1]) for h in self.HALF_LIVES},
        })
        mu, sd = 1 / p0, np.sqrt(1 - p0) / p0
        f["gap_z"] = np.clip((gap - mu) / sd, -3.0, 5.0)
        f["L_all50"] = _logit((count + 50 * p0) / (seen[:, None] + 50))
        f["L_all500"] = _logit((count + 500 * p0) / (seen[:, None] + 500))
        f["L_all100"] = _logit((count + 100 * p0) / (seen[:, None] + 100))
        for h in self.HALF_LIVES:
            f[f"L_ew{h}"] = _logit((f[f"s{h}"] + 20 * p0) / (f[f"m{h}"][:, None] + 20))
        return f, new

    def run(self, st: dict, obs: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        x = np.asarray(obs["X"], dtype=bool)
        t, n, k = len(x), self.spec.n, self.spec.k
        f, new = self._features(st, x)
        lp0 = float(_logit(self.p0))
        phi = np.stack([f["L_all100"] - lp0, f["L_ew10"] - lp0, f["L_ew100"] - lp0, f["gap_z"], f["prev"], f["prev2"]], axis=-1)  # (T+1, n, F)
        logistic = self._online_logistic(new, phi, x)
        z = np.stack([
            np.zeros((t + 1, n)),
            f["L_all50"], f["L_all500"], -f["L_all50"], -f["L_all500"],
            f["L_ew10"], f["L_ew100"], -f["L_ew10"], -f["L_ew100"],
            0.15 * f["gap_z"], -0.15 * f["gap_z"], 0.2 * f["prev"], -0.2 * f["prev"],
            logistic,
        ], axis=1)  # (T+1, J, n) log-odds
        z = np.clip(z - z.mean(axis=-1, keepdims=True), -5.0, 5.0)
        w = np.exp(z)
        pred = w / w.sum(axis=-1, keepdims=True)
        ll = set_loglik(pred[:t], x, obs.get("bonus"), k, self.spec.bonus_same_drum) if t else np.zeros((0, len(self.names)))
        return ll, pred, new

    def _online_logistic(self, st: dict, phi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Mini-batch AdaGrad on the per-number Bernoulli loss; batches are aligned to the
        global draw count so chunking never changes the result. Predictions for a draw only
        use parameters fitted on earlier batches."""
        t, n = len(x), self.spec.n
        out = np.empty(phi.shape[:2])
        theta, c = st["theta"].copy(), float(st["c"])
        big_g, acc_g, acc_n = st["G"].copy(), st["acc_g"].copy(), float(st["acc_n"])
        i = 0
        while i < t:
            end = min(t, i + self.batch - int(acc_n))
            seg = slice(i, end)
            out[seg] = phi[seg] @ theta
            err = 1.0 / (1.0 + np.exp(-(c + out[seg]))) - x[seg].astype(np.float64)
            acc_g[:-1] += np.einsum("tn,tnf->f", err, phi[seg])
            acc_g[-1] += err.sum()
            acc_n += end - i
            if acc_n >= self.batch:
                grad = acc_g / (self.batch * n) + self.l2 * np.append(theta, 0.0)
                big_g += grad**2
                step = self.lr * grad / (np.sqrt(big_g) + 1e-8)
                theta -= step[:-1]
                c -= step[-1]
                acc_g[:] = 0.0
                acc_n = 0.0
            i = end
        out[t] = phi[t] @ theta
        st.update({"theta": theta, "c": c, "G": big_g, "acc_g": acc_g, "acc_n": acc_n})
        return out


def set_loglik(pred: np.ndarray, x: np.ndarray, bonus: np.ndarray | None, k: int, same_drum: bool) -> np.ndarray:
    """log P(draw) under CB weights ``pred`` (T, J, n) for incidence ``x`` (T, n)."""
    logw = np.log(pred)
    ek = elementary_symmetric(pred, k)[..., k]
    ll = np.einsum("tjn,tn->tj", logw, x.astype(np.float64)) - np.log(ek)
    if same_drum and bonus is not None:
        b = np.asarray(bonus, dtype=np.int64) - 1
        rows = np.arange(len(x))
        ll += logw[rows, :, b] - np.log(np.einsum("tjn,tn->tj", pred, (~x).astype(np.float64)))
    return ll


class SetTilts:
    """Fixed sparse hypotheses "number i comes up (1+ε) times as often": n·|ε| experts."""

    def __init__(self, spec: SetSpec, eps: tuple[float, ...] = TILT_EPS) -> None:
        self.spec, self.eps = spec, np.asarray(eps, dtype=np.float64)
        n = spec.n
        self.names = [f"tilt[{i + 1}]{e:+.2f}" for e in eps for i in range(n)]
        w = np.ones((len(eps), n, n))
        w[:, np.arange(n), np.arange(n)] += self.eps[:, None]
        w = w.reshape(len(eps) * n, n)
        self.constant_pred = w / w.sum(axis=1, keepdims=True)

    def init_state(self) -> dict:
        return {}

    def run(self, st: dict, obs: dict) -> tuple[np.ndarray, None, dict]:
        x = np.asarray(obs["X"], dtype=bool)
        n, k = self.spec.n, self.spec.k
        le = np.log1p(self.eps)  # (E,)
        # e_k(w) = C(n−1,k) + (1+ε)·C(n−1,k−1) = C(n−1,k−1)·((n−k)/k + 1 + ε)
        logden = log_comb(n - 1, k - 1) + np.log((n - k) / k + 1 + self.eps)
        ll = x[:, None, :] * le[None, :, None] - logden[None, :, None]  # (T, E, n)
        if self.spec.bonus_same_drum and obs.get("bonus") is not None:
            b = np.asarray(obs["bonus"], dtype=np.int64) - 1
            hit = np.zeros_like(x)
            hit[np.arange(len(x)), b] = True
            ll = ll + hit[:, None, :] * le[None, :, None] - np.log((n - k) + self.eps[None, :, None] * (~x)[:, None, :])
        return ll.reshape(len(x), -1), None, st


# ====================================================================== digit games
class DigitExperts:
    """Adaptive experts for m numbers × L independent positions over A symbols."""

    def __init__(self, spec: DigitSpec) -> None:
        self.spec = spec
        self.names = [
            "uniform", "dirichlet_pos_a100", "dirichlet_pos_a1000", "dirichlet_pooled_a100", "dirichlet_pooled_a1000",
            "ewma20", "ewma200", "cold_a100", "repeat", "anti_repeat",
        ]
        self.constant_pred = None

    def init_state(self) -> dict:
        la = (self.spec.positions, self.spec.alphabet)
        return {"seen": 0.0, "count": np.zeros(la), "s20": np.zeros(la), "s200": np.zeros(la), "prev": np.zeros(la)}

    def run(self, st: dict, obs: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        c = np.asarray(obs["C"], dtype=np.float64)  # (T, L, A)
        t = len(c)
        big_a, m = self.spec.alphabet, self.spec.per_draw
        count = _running_sum(c, st["count"])
        s20 = _running_ewma(c, st["s20"], _lam(20))
        s200 = _running_ewma(c, st["s200"], _lam(200))
        prev = np.concatenate([st["prev"][None], c], axis=0)
        new = {"seen": st["seen"] + t, "count": count[-1], "s20": s20[-1], "s200": s200[-1], "prev": prev[-1]}

        def dirichlet(cnt: np.ndarray, a: float) -> np.ndarray:
            return (cnt + a / big_a) / (cnt.sum(axis=-1, keepdims=True) + a)

        pool = count.sum(axis=1, keepdims=True)  # (T+1, 1, A)
        mu = m / big_a
        zrep = (prev - mu) / np.sqrt(mu * (1 - 1 / big_a))
        rep = np.exp(0.15 * zrep)
        anti = np.exp(-0.15 * zrep)
        cold = 1.0 / dirichlet(count, 100.0)
        q = np.stack([
            np.full_like(count, 1.0 / big_a),
            dirichlet(count, 100.0), dirichlet(count, 1000.0),
            np.broadcast_to(dirichlet(pool, 100.0), count.shape), np.broadcast_to(dirichlet(pool, 1000.0), count.shape),
            dirichlet(s20, 50.0), dirichlet(s200, 50.0),
            cold / cold.sum(axis=-1, keepdims=True), rep / rep.sum(axis=-1, keepdims=True), anti / anti.sum(axis=-1, keepdims=True),
        ], axis=1)  # (T+1, J, L, A)
        ll = np.einsum("tjla,tla->tj", np.log(q[:t]), c) if t else np.zeros((0, len(self.names)))
        return ll, q.reshape(t + 1, len(self.names), -1), new


class DigitTilts:
    """Fixed sparse hypotheses "symbol a at position l is (1+ε) times as likely"."""

    def __init__(self, spec: DigitSpec, eps: tuple[float, ...] = TILT_EPS) -> None:
        self.spec, self.eps = spec, np.asarray(eps, dtype=np.float64)
        big_l, big_a = spec.positions, spec.alphabet
        self.names = [f"tilt[{spec.position_labels[pos]}={a + spec.symbol_offset}]{e:+.2f}" for e in eps for pos in range(big_l) for a in range(big_a)]
        q = np.full((len(eps), big_l, big_a, big_l, big_a), 1.0 / big_a)
        for ei, e in enumerate(eps):
            for pos in range(big_l):
                for a in range(big_a):
                    q[ei, pos, a, pos, :] = 1.0 / (big_a + e)
                    q[ei, pos, a, pos, a] = (1.0 + e) / (big_a + e)
        self.constant_pred = q.reshape(len(eps) * big_l * big_a, big_l * big_a)

    def init_state(self) -> dict:
        return {}

    def run(self, st: dict, obs: dict) -> tuple[np.ndarray, None, dict]:
        c = np.asarray(obs["C"], dtype=np.float64)  # (T, L, A)
        big_l, big_a, m = self.spec.positions, self.spec.alphabet, self.spec.per_draw
        null = -m * big_l * log(big_a)
        ll = null + c[:, None, :, :] * np.log1p(self.eps)[None, :, None, None] - m * np.log((big_a + self.eps) / big_a)[None, :, None, None]
        return ll.reshape(len(c), -1), None, st
