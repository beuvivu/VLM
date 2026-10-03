from __future__ import annotations

import numpy as np
import pytest

from vietlott_engine.core import config as _config
from vietlott_engine.core.games import MEGA_645, POWER_655, GameSpec
from vietlott_engine.core.history import DrawHistory


@pytest.fixture(autouse=True, scope="session")
def _ignore_local_dotenv():  # type: ignore[no-untyped-def]
    """The installers write a .env for the user; tests must not depend on it.

    Session scope: module-scoped fixtures (the API client in test_api.py) build their Settings
    before any function-scoped fixture runs, so a function-scoped guard came too late.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(_config.Settings.model_config, "env_file", None)
        _config.get_settings.cache_clear()
        yield
    _config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_settings():  # type: ignore[no-untyped-def]
    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()


def make_history(spec: GameSpec, draws: int, seed: int = 0, weights: np.ndarray | None = None) -> DrawHistory:
    """Synthetic history; ``weights`` (n,) plants a per-number bias (Plackett–Luce sampling)."""
    rng = np.random.default_rng(seed)
    n, k = spec.pool_size, spec.pick
    if weights is None:
        nums = np.argpartition(rng.random((draws, n)), k, axis=1)[:, :k] + 1
    else:
        g = np.log(weights)[None, :] + rng.gumbel(size=(draws, n))
        nums = np.argpartition(-g, k, axis=1)[:, :k] + 1
    bonus = None
    if spec.separate_special:
        bonus = rng.integers(1, int(spec.bonus_pool_size or 1) + 1, draws)
    elif spec.has_bonus:
        bonus = np.array([rng.choice(np.setdiff1d(np.arange(1, n + 1), row)) for row in nums])
    return DrawHistory.from_arrays(spec, nums, bonus=bonus)


@pytest.fixture
def mega_history() -> DrawHistory:
    return make_history(MEGA_645, 800, seed=1)


@pytest.fixture
def power_history() -> DrawHistory:
    return make_history(POWER_655, 800, seed=2)
