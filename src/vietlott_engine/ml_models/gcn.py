"""Graph Convolutional Network over the number co-occurrence graph (pure NumPy).

Architecture (Kipf & Welling 2017), per snapshot t:
    H = ReLU(Â_t X_t W₁ + b₁)          (n × hidden)
    z = Â_t H w₂ + b₂                  (n,)
    P(number i in draw t) = σ(z_i)
trained with binary cross-entropy + L2, Adam, mini-batches of snapshots and early
stopping on a chronologically later validation block. Gradients are analytic and
verified against finite differences in the test suite.

A dependency-free NumPy implementation keeps the container small (no PyTorch /
PyG) and is plenty for a 45/55-node graph. The point of the model is to *test*
whether co-occurrence structure carries predictive information, measured by
out-of-sample log-loss against the k/n baseline — not to assume that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from vietlott_engine.core.logging import get_logger
from vietlott_engine.inference.predictive import calibration, diebold_mariano
from vietlott_engine.inference.sequential import predictive_eprocess
from vietlott_engine.ml_models.features import Snapshots

log = get_logger(__name__)


@dataclass
class GCNConfig:
    hidden: int = 16
    learning_rate: float = 0.01
    l2: float = 1e-3
    epochs: int = 60
    batch_size: int = 32
    patience: int = 8
    validation_fraction: float = 0.2
    seed: int = 0


@dataclass
class GCNParams:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def as_list(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def copy(self) -> "GCNParams":
        return GCNParams(*(p.copy() for p in self.as_list()))


class GCNPredictor:
    def __init__(self, n_features: int, config: GCNConfig | None = None, base_rate: float = 0.13) -> None:
        self.cfg = config or GCNConfig()
        rng = np.random.default_rng(self.cfg.seed)
        f, hdim = n_features, self.cfg.hidden
        self.params = GCNParams(
            w1=rng.normal(0, np.sqrt(2 / f), (f, hdim)),
            b1=np.zeros(hdim),
            w2=rng.normal(0, np.sqrt(1 / hdim), (hdim,)) * 0.1,
            b2=np.array([np.log(base_rate / (1 - base_rate))]),
        )
        self.history: list[dict] = []

    # ------------------------------------------------------------ forward
    @staticmethod
    def _forward(p: GCNParams, a: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
        u = a @ x  # (B, n, f)
        z1 = u @ p.w1 + p.b1  # (B, n, h)
        h1 = np.maximum(z1, 0.0)
        v = a @ h1  # (B, n, h)
        z = v @ p.w2 + p.b2[0]  # (B, n)
        return z, {"u": u, "z1": z1, "h1": h1, "v": v}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def loss_and_grads(self, p: GCNParams, a: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, list[np.ndarray]]:
        z, c = self._forward(p, a, x)
        prob = self._sigmoid(z)
        eps = 1e-9
        m = y.size
        loss = -np.sum(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)) / m
        loss += 0.5 * self.cfg.l2 * (np.sum(p.w1**2) + np.sum(p.w2**2))
        dz = (prob - y) / m  # (B, n)
        dw2 = np.einsum("bnh,bn->h", c["v"], dz) + self.cfg.l2 * p.w2
        db2 = np.array([dz.sum()])
        dv = dz[..., None] * p.w2[None, None, :]  # (B, n, h)
        dh1 = np.transpose(a, (0, 2, 1)) @ dv
        dz1 = dh1 * (c["z1"] > 0)
        dw1 = np.einsum("bnf,bnh->fh", c["u"], dz1) + self.cfg.l2 * p.w1
        db1 = dz1.sum(axis=(0, 1))
        return float(loss), [dw1, db1, dw2, db2]

    # --------------------------------------------------------------- train
    def fit(self, snaps: Snapshots, t_start: int, t_end: int) -> "GCNPredictor":
        """Train on snapshots t ∈ [t_start, t_end) — callers guarantee t_end ≤ first test index."""
        idx = np.arange(t_start, t_end)
        n_val = max(1, int(len(idx) * self.cfg.validation_fraction))
        train_idx, val_idx = idx[:-n_val], idx[-n_val:]
        rng = np.random.default_rng(self.cfg.seed)
        a_all, x_all, y_all = snaps.a_hat.astype(np.float64), snaps.x.astype(np.float64), snaps.y.astype(np.float64)
        m = [np.zeros_like(q) for q in self.params.as_list()]
        v = [np.zeros_like(q) for q in self.params.as_list()]
        b1, b2, step = 0.9, 0.999, 0
        best, best_val, bad = self.params.copy(), np.inf, 0
        for epoch in range(self.cfg.epochs):
            rng.shuffle(train_idx)
            for s in range(0, len(train_idx), self.cfg.batch_size):
                bi = train_idx[s : s + self.cfg.batch_size]
                _, grads = self.loss_and_grads(self.params, a_all[bi], x_all[bi], y_all[bi])
                step += 1
                for j, (prm, g) in enumerate(zip(self.params.as_list(), grads)):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mh, vh = m[j] / (1 - b1**step), v[j] / (1 - b2**step)
                    prm -= self.cfg.learning_rate * mh / (np.sqrt(vh) + 1e-8)
            val_loss = self.log_loss(snaps, val_idx)
            self.history.append({"epoch": epoch, "val_log_loss": val_loss})
            if val_loss < best_val - 1e-6:
                best, best_val, bad = self.params.copy(), val_loss, 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        self.params = best
        return self

    # ----------------------------------------------------------- predict
    def predict_proba(self, snaps: Snapshots, t: int | np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t)
        z, _ = self._forward(self.params, snaps.a_hat[t].astype(np.float64), snaps.x[t].astype(np.float64))
        return self._sigmoid(z)

    def log_loss(self, snaps: Snapshots, t: np.ndarray) -> float:
        p = self.predict_proba(snaps, t)
        y = snaps.y[t].astype(np.float64)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


class SkillReport(BaseModel):
    model: str
    test_draws: int
    model_log_loss: float
    baseline_log_loss: float
    skill_per_draw_nats: float  # positive = better than k/n baseline (Bernoulli log score)
    standard_error: float  # HAC (Newey–West / Andrews)
    t_statistic: float  # Diebold–Mariano with HLN correction
    dm_p_value_greater: float
    mean_hits_top_k: float  # hits among the model's top-k numbers per draw
    expected_hits_random: float
    calibration_z: float  # Spiegelhalter; |z| > 2 ⇒ miscalibrated probabilities
    calibration_p_value: float
    eprocess_max_log10_wealth: float  # betting on the forecasts with exact subset probabilities
    eprocess_anytime_p_value: float


def walk_forward_skill(snaps: Snapshots, k: int, first_test: int, refit_every: int = 100, config: GCNConfig | None = None, model: str = "gcn") -> SkillReport:
    """Rolling-origin evaluation: refit on [warmup, T) and score (T, T+refit_every]."""
    d, n = snaps.y.shape
    base = k / n
    warmup = max(30, first_test // 5)
    per_draw_gain, model_ll, hits, all_probs = [], [], [], []
    t = first_test
    while t < d:
        end = min(d, t + refit_every)
        if model == "gcn":
            est = GCNPredictor(snaps.x.shape[2], config, base).fit(snaps, warmup, t)
            prob = est.predict_proba(snaps, np.arange(t, end))
        elif model == "logistic":
            prob = _logistic_predict(snaps, warmup, t, end)
        else:
            raise ValueError(model)
        y = snaps.y[t:end].astype(np.float64)
        ll_m = (y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9)).sum(axis=1)
        ll_b = (y * np.log(base) + (1 - y) * np.log(1 - base)).sum(axis=1)
        per_draw_gain.extend(ll_m - ll_b)
        model_ll.extend(ll_m)
        all_probs.append(prob)
        top = np.argsort(-prob, axis=1)[:, :k]
        hits.extend(np.take_along_axis(y, top, axis=1).sum(axis=1))
        log.info("%s walk-forward block [%d, %d) done", model, t, end)
        t = end
    g = np.asarray(per_draw_gain)
    dm = diebold_mariano(g)
    probs = np.vstack(all_probs)
    y_test = snaps.y[first_test:].astype(np.float64)
    cal = calibration(probs, y_test)
    draws = np.sort(np.argsort(-y_test, axis=1)[:, :k] + 1, axis=1)
    ep = predictive_eprocess(model, probs, draws, n)
    return SkillReport(
        model=model,
        test_draws=int(g.size),
        model_log_loss=float(-np.mean(model_ll) / n),
        baseline_log_loss=float(-(k * np.log(base) + (n - k) * np.log(1 - base)) / n),
        skill_per_draw_nats=float(g.mean()),
        standard_error=dm.hac_standard_error,
        t_statistic=dm.statistic,
        dm_p_value_greater=dm.p_value_greater,
        mean_hits_top_k=float(np.mean(hits)),
        expected_hits_random=k * k / n,
        calibration_z=cal.spiegelhalter_z,
        calibration_p_value=cal.p_value,
        eprocess_max_log10_wealth=ep.max_log10_wealth,
        eprocess_anytime_p_value=ep.anytime_p_value,
    )


def _logistic_predict(snaps: Snapshots, t0: int, t1: int, t2: int) -> np.ndarray:
    """Non-graph baseline: L2 logistic regression on the same node features."""
    from sklearn.linear_model import LogisticRegression

    n, f = snaps.x.shape[1], snaps.x.shape[2]
    x_train = snaps.x[t0:t1].reshape(-1, f)
    y_train = snaps.y[t0:t1].reshape(-1)
    clf = LogisticRegression(C=0.1, max_iter=500)
    clf.fit(x_train, y_train)
    return clf.predict_proba(snaps.x[t1:t2].reshape(-1, f))[:, 1].reshape(t2 - t1, n)
