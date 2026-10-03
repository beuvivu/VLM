"""Trainable bounded experts. All checkpoints are data, never executable pickle."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
from scipy.special import expit, logit


class OnlineLogistic:
    def __init__(self, features: int, base: float, l2: float = .001):
        self.features, self.base, self.l2 = features, base, l2
        self.theta = np.zeros(features + 1)
        self.acc = np.zeros_like(self.theta)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return expit(x @ self.theta[:-1] + self.theta[-1] + logit(self.base))

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        error = self.predict(x) - y
        gradient = np.append(x.T @ error / len(y), error.mean())
        gradient[:-1] += self.l2 * self.theta[:-1]
        gradient = np.clip(gradient, -1, 1)
        self.acc += gradient ** 2
        self.theta -= .08 * gradient / (np.sqrt(self.acc) + 1e-8)

    def to_dict(self) -> dict:
        return {'features':self.features, 'base':self.base, 'l2':self.l2,
                'theta':self.theta.tolist(), 'acc':self.acc.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> OnlineLogistic:
        out = cls(data['features'], data['base'], data['l2'])
        for key in ('theta', 'acc'):
            value = np.array(data[key], dtype=float)
            if value.shape != out.theta.shape or not np.isfinite(value).all():
                raise ValueError('Invalid logistic checkpoint')
            setattr(out, key, value)
        if (out.acc < 0).any():
            raise ValueError('Invalid AdaGrad state')
        return out


class GRU:
    """Shared node GRU, reset-before projection, eight-step truncated BPTT + Adam.

    Hidden state is rebuilt from the causal window, not carried across optimizer
    updates. This makes reload/chunk behavior deterministic and bounds memory.
    """
    def __init__(self, features: int, hidden: int = 8, base: float = .5,
                 seed: int = 20261003, l2: float = .001, lr: float = .006):
        self.features, self.hidden, self.base = features, hidden, base
        self.seed, self.l2, self.lr = seed, l2, lr
        rng = np.random.default_rng(seed)
        self.params = {}
        for gate in ('r', 'z', 'n'):
            self.params['W' + gate] = rng.normal(0, .1 / np.sqrt(features), (features, hidden))
            self.params['U' + gate] = rng.normal(0, .1 / np.sqrt(hidden), (hidden, hidden))
            self.params['b' + gate] = np.zeros(hidden)
        self.params['Wo'] = rng.normal(0, .1, hidden)
        self.params['bo'] = np.zeros(1)
        self.first = {k:np.zeros_like(v) for k, v in self.params.items()}
        self.second = {k:np.zeros_like(v) for k, v in self.params.items()}
        self.steps = 0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or x.shape[-1] != self.features or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError('Invalid GRU input')
        p = self.params
        h = np.zeros((x.shape[1], self.hidden))
        cache = []
        for frame in x:
            previous = h
            r = expit(frame @ p['Wr'] + previous @ p['Ur'] + p['br'])
            z = expit(frame @ p['Wz'] + previous @ p['Uz'] + p['bz'])
            candidate = np.tanh(frame @ p['Wn'] + (r * previous) @ p['Un'] + p['bn'])
            h = (1 - z) * previous + z * candidate
            cache.append((frame, previous, r, z, candidate, h))
        logits = h @ p['Wo'] + p['bo'][0] + logit(self.base)
        return logits, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x)
        return expit(logits)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self._forward(x)
        y = np.asarray(y, dtype=float)
        if y.shape != logits.shape or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError('Invalid GRU target')
        p = self.params
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
        g = {k:np.zeros_like(v) for k, v in p.items()}
        dl = (expit(logits) - y) / len(y)
        g['Wo'] = cache[-1][-1].T @ dl
        g['bo'][0] = dl.sum()
        dh = dl[:, None] * p['Wo'][None, :]
        for frame, previous, r, z, candidate, _ in reversed(cache):
            dn = dh * z * (1 - candidate**2)
            dz = dh * (candidate - previous) * z * (1 - z)
            drh = dn @ p['Un'].T
            dr = drh * previous * r * (1 - r)
            for gate, gradient, recurrent in [('n', dn, r * previous), ('z', dz, previous), ('r', dr, previous)]:
                g['W' + gate] += frame.T @ gradient
                g['U' + gate] += recurrent.T @ gradient
                g['b' + gate] += gradient.sum(axis=0)
            dh = dh * (1 - z) + drh * r + dz @ p['Uz'].T + dr @ p['Ur'].T
        for key in p:
            if not key.startswith('b'):
                loss += .5 * self.l2 * float(np.sum(p[key] ** 2))
                g[key] += self.l2 * p[key]
        return loss, g

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        _, gradients = self.loss_and_gradients(x, y)
        norm = np.sqrt(sum(float(np.sum(g**2)) for g in gradients.values()))
        scale = min(1., 1. / max(norm, 1e-12))
        self.steps += 1
        for key, gradient in gradients.items():
            gradient = gradient * scale
            self.first[key] = .9 * self.first[key] + .1 * gradient
            self.second[key] = .999 * self.second[key] + .001 * gradient**2
            m = self.first[key] / (1 - .9 ** self.steps)
            v = self.second[key] / (1 - .999 ** self.steps)
            self.params[key] -= self.lr * m / (np.sqrt(v) + 1e-8)

    def to_dict(self) -> dict:
        return {'features':self.features, 'hidden':self.hidden, 'base':self.base,
                'seed':self.seed, 'l2':self.l2, 'lr':self.lr, 'steps':self.steps,
                **{name:{k:v.tolist() for k, v in getattr(self, name).items()}
                   for name in ('params', 'first', 'second')}}

    @classmethod
    def from_dict(cls, data: dict) -> GRU:
        out = cls(**{k:data[k] for k in ('features', 'hidden', 'base', 'seed', 'l2', 'lr')})
        for name in ('params', 'first', 'second'):
            current = getattr(out, name)
            if set(data[name]) != set(current):
                raise ValueError('Invalid GRU checkpoint')
            for key in current:
                value = np.array(data[name][key], dtype=float)
                if value.shape != current[key].shape or not np.isfinite(value).all():
                    raise ValueError('Invalid GRU parameter')
                current[key] = value
        out.steps = int(data['steps'])
        if out.steps < 0 or any((v < 0).any() for v in out.second.values()):
            raise ValueError('Invalid Adam checkpoint')
        return out


class TreeExpert:
    """Periodic bounded tree fit; RF prediction is persisted as JSON tree arrays."""
    def __init__(self, backend: str, base: float, seed: int = 20261003, depth: int = 4):
        if backend not in ('rf', 'xgb', 'lgb'):
            raise ValueError('Unknown tree backend')
        module = {'rf':'sklearn', 'xgb':'xgboost', 'lgb':'lightgbm'}[backend]
        if importlib.util.find_spec(module) is None:
            raise ImportError(f'{module} is required; install VLM[ml] for optional boosters')
        self.backend, self.base, self.seed, self.depth = backend, base, seed, depth
        self.trees: list[dict] = []
        self.booster = None
        self.constant: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError('Invalid tree training data')
        self.trees, self.booster, self.constant = [], None, None
        if np.ptp(y) < 1e-12:
            self.constant = float(np.clip(y[0], .001, .999))
            return
        if self.backend == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            forest = RandomForestRegressor(n_estimators=24, max_depth=self.depth,
                min_samples_leaf=max(4, min(64, len(y)//20)), max_features=.7,
                random_state=self.seed, n_jobs=1).fit(x, y)
            self.trees = [{'left':e.tree_.children_left.tolist(), 'right':e.tree_.children_right.tolist(),
                           'feature':e.tree_.feature.tolist(), 'threshold':e.tree_.threshold.tolist(),
                           'value':e.tree_.value[:, 0, 0].tolist()} for e in forest.estimators_]
        elif self.backend == 'xgb':
            import xgboost as xgb
            self.booster = xgb.train({'objective':'binary:logistic', 'max_depth':self.depth,
                'eta':.05, 'lambda':10, 'min_child_weight':10, 'subsample':.8,
                'colsample_bytree':.7, 'seed':self.seed, 'nthread':1,
                'base_score':self.base, 'tree_method':'hist'}, xgb.DMatrix(x, label=y), num_boost_round=24)
        else:
            import lightgbm as lgb
            self.booster = lgb.train({'objective':'cross_entropy', 'max_depth':self.depth,
                'num_leaves':2 ** self.depth, 'learning_rate':.05, 'lambda_l2':10,
                'min_data_in_leaf':max(4, min(64, len(y)//20)), 'feature_fraction':.7,
                'seed':self.seed, 'num_threads':1, 'verbosity':-1}, lgb.Dataset(x, label=y), num_boost_round=24)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.constant is not None:
            return np.full(len(x), self.constant)
        if self.trees:
            result = np.zeros(len(x))
            for tree in self.trees:
                left, right = np.array(tree['left']), np.array(tree['right'])
                feature, threshold, value = np.array(tree['feature']), np.array(tree['threshold']), np.array(tree['value'])
                node = np.zeros(len(x), dtype=int)
                for _ in range(self.depth + 1):
                    active = left[node] != -1
                    if not active.any():
                        break
                    indices = np.flatnonzero(active)
                    nd = node[active]
                    go_left = x[indices, feature[nd]] <= threshold[nd]
                    node[indices] = np.where(go_left, left[nd], right[nd])
                result += value[node]
            return np.clip(result / len(self.trees), .001, .999)
        if self.booster is not None:
            if self.backend == 'xgb':
                import xgboost as xgb
                return np.clip(self.booster.predict(xgb.DMatrix(x)), .001, .999)
            return np.clip(self.booster.predict(x), .001, .999)
        return np.full(len(x), self.base)

    def to_dict(self) -> dict:
        raw = None
        if self.booster is not None:
            raw = self.booster.save_raw(raw_format='json').decode() if self.backend == 'xgb' else self.booster.model_to_string()
        return {'backend':self.backend, 'base':self.base, 'seed':self.seed, 'depth':self.depth,
                'trees':self.trees, 'constant':self.constant, 'booster':raw}

    @classmethod
    def from_dict(cls, data: dict, *, features: int | None = None) -> TreeExpert:
        out = cls(**{k:data[k] for k in ('backend', 'base', 'seed', 'depth')})
        out.trees, out.constant = data['trees'], data['constant']
        if (type(out.depth) is not int or not 1 <= out.depth <= 10 or len(out.trees) > 24
            or out.constant is not None and not 0 <= out.constant <= 1
            or out.trees and (out.backend != 'rf' or data['booster'] is not None)):
            raise ValueError('Invalid tree checkpoint')
        for tree in out.trees:
            length = len(tree['left'])
            if length < 1 or any(len(tree[k]) != length for k in ('right', 'feature', 'threshold', 'value')):
                raise ValueError('Invalid RF tree')
            if any(not -1 <= int(c) < length for k in ('left', 'right') for c in tree[k]):
                raise ValueError('Invalid RF child')
            if any(type(v) is not int for k in ('left', 'right', 'feature') for v in tree[k]):
                raise ValueError('Invalid RF index')
            if not np.isfinite(tree['threshold']).all() or not np.isfinite(tree['value']).all() or any(not 0 <= v <= 1 for v in tree['value']):
                raise ValueError('Invalid RF value')
            visited, stack = set(), [(0, 0)]
            while stack:
                node, depth = stack.pop()
                if node in visited or depth > out.depth:
                    raise ValueError('Invalid RF topology')
                visited.add(node)
                left, right = tree['left'][node], tree['right'][node]
                if left == -1:
                    if right != -1:
                        raise ValueError('Invalid RF leaf')
                else:
                    feature = tree['feature'][node]
                    if right == -1 or feature < 0 or features is not None and feature >= features:
                        raise ValueError('Invalid RF split')
                    stack.extend([(left, depth + 1), (right, depth + 1)])
            if len(visited) != length:
                raise ValueError('Unreachable RF nodes')
        if data['booster'] is not None:
            if out.backend == 'xgb':
                import xgboost as xgb
                out.booster = xgb.Booster()
                out.booster.load_model(bytearray(data['booster'], 'utf-8'))
            else:
                import lightgbm as lgb
                out.booster = lgb.Booster(model_str=data['booster'])
        # Reject NaN/Infinity even in unused tree fields.
        json.dumps(data, allow_nan=False)
        return out
