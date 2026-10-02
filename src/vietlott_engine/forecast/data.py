"""Product histories → forecasting series (one or two *components* per product).

* **set** component — k distinct numbers from 1..n (Mega, Power, Lotto main numbers, Keno):
  observation = incidence row ``X`` (n,) plus, for Power, the bonus ball drawn from the same
  drum (``bonus`` 1..n).
* **digit** component — m numbers per draw, each with L independent positions over an
  alphabet of A symbols (Max 3D / Pro: m=20, L=3, A=10 · Max 4D: m=6, L=4 · Bingo18: m=1,
  L=3 dice, A=6 · Lotto 5/35 special number: m=1, L=1, A=12): observation = counts
  ``C`` (L, A) of each symbol at each position.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vietlott_engine.core.games import GameCode, get_game
from vietlott_engine.core.history import DrawHistory
from vietlott_engine.core.products import DIGIT_PRODUCTS, ProductCode, ProductHistory, get_product


@dataclass(frozen=True)
class SetSpec:
    n: int
    k: int
    bonus_same_drum: bool = False
    kind: str = "set"


@dataclass(frozen=True)
class DigitSpec:
    alphabet: int
    positions: int
    per_draw: int
    position_labels: tuple[str, ...]
    symbol_offset: int = 0  # displayed symbol = index + offset (Bingo18 faces 1..6, Lotto special 1..12)
    kind: str = "digit"


@dataclass(frozen=True)
class ProductConfig:
    product: ProductCode
    components: dict[str, SetSpec | DigitSpec]
    draws_per_year: int
    logistic_batch: int = 20


CONFIGS: dict[ProductCode, ProductConfig] = {
    ProductCode.MEGA_645: ProductConfig(ProductCode.MEGA_645, {"main": SetSpec(45, 6)}, 156),
    ProductCode.POWER_655: ProductConfig(ProductCode.POWER_655, {"main": SetSpec(55, 6, bonus_same_drum=True)}, 156),
    ProductCode.LOTTO_535: ProductConfig(
        ProductCode.LOTTO_535,
        {"main": SetSpec(35, 5), "special": DigitSpec(12, 1, 1, ("số đặc biệt",), symbol_offset=1)},
        730,
    ),
    ProductCode.KENO: ProductConfig(ProductCode.KENO, {"main": SetSpec(80, 20)}, 41_800, logistic_batch=500),
    ProductCode.BINGO18: ProductConfig(ProductCode.BINGO18, {"dice": DigitSpec(6, 3, 1, ("bóng 1", "bóng 2", "bóng 3"), symbol_offset=1)}, 57_700, logistic_batch=500),
    ProductCode.MAX3D: ProductConfig(ProductCode.MAX3D, {"digits": DigitSpec(10, 3, 20, ("trăm", "chục", "đơn vị"))}, 156),
    ProductCode.MAX3D_PRO: ProductConfig(ProductCode.MAX3D_PRO, {"digits": DigitSpec(10, 3, 20, ("trăm", "chục", "đơn vị"))}, 156),
    ProductCode.MAX4D: ProductConfig(ProductCode.MAX4D, {"digits": DigitSpec(10, 4, 6, ("nghìn", "trăm", "chục", "đơn vị"))}, 151),
}


@dataclass
class Series:
    """Draws in id order; ``obs[component]`` holds that component's observation arrays."""

    product: ProductCode
    draw_ids: np.ndarray
    dates: np.ndarray
    obs: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.draw_ids.size)

    def after(self, draw_id: int | None) -> "Series":
        """Draws with id > ``draw_id`` (all when None)."""
        keep = np.ones(len(self), bool) if draw_id is None else self.draw_ids > draw_id
        return Series(self.product, self.draw_ids[keep], self.dates[keep], {c: {k: v[keep] for k, v in o.items()} for c, o in self.obs.items()})

    def head(self, t: int) -> "Series":
        return Series(self.product, self.draw_ids[:t], self.dates[:t], {c: {k: v[:t] for k, v in o.items()} for c, o in self.obs.items()})

    def tail(self, m: int) -> "Series":
        start = max(0, len(self) - m)
        return Series(self.product, self.draw_ids[start:], self.dates[start:], {c: {k: v[start:] for k, v in o.items()} for c, o in self.obs.items()})


def digit_counts(digits: np.ndarray, alphabet: int) -> np.ndarray:
    """(T, m, L) symbols 0..A−1 → (T, L, A) counts per position."""
    t, _, ll = digits.shape
    out = np.zeros((t, ll, alphabet), dtype=np.int32)
    for pos in range(ll):
        np.add.at(out[:, pos, :], (np.repeat(np.arange(t), digits.shape[1]), digits[:, :, pos].ravel()), 1)
    return out


def to_digits(values: np.ndarray, width: int) -> np.ndarray:
    """(T, m) integers → (T, m, width) decimal digits, most significant first."""
    v = np.asarray(values, dtype=np.int64)
    powers = 10 ** np.arange(width - 1, -1, -1)
    return (v[..., None] // powers) % 10


def matrix_series(h: DrawHistory) -> Series:
    code = get_product(h.spec.code.value)
    order = np.argsort(h.draw_ids, kind="stable")
    obs: dict[str, dict[str, np.ndarray]] = {"main": {"X": h.incidence[order].copy(), "bonus": h.bonus[order].astype(np.int16)}}
    if code == ProductCode.LOTTO_535:
        sp = h.bonus[order].astype(np.int64) - 1
        if (sp < 0).any():
            raise ValueError("Lotto 5/35 draws without a special number")
        obs["special"] = {"C": digit_counts(sp.reshape(-1, 1, 1), 12)}
    return Series(code, h.draw_ids[order].copy(), h.dates[order].copy(), obs)


def product_series(h: ProductHistory) -> Series:
    code = h.product
    order = np.argsort(h.draw_ids, kind="stable")
    vals = h.values[order]
    if code == ProductCode.KENO:
        x = np.zeros((len(vals), 80), dtype=bool)
        x[np.repeat(np.arange(len(vals)), vals.shape[1]), vals.ravel().astype(int) - 1] = True
        obs = {"main": {"X": x, "bonus": np.zeros(len(vals), dtype=np.int16)}}
    elif code == ProductCode.BINGO18:
        obs = {"dice": {"C": digit_counts((vals.astype(np.int64) - 1).reshape(-1, 1, 3), 6)}}
    else:
        width = DIGIT_PRODUCTS[code]
        obs = {"digits": {"C": digit_counts(to_digits(vals, width), 10)}}
    return Series(code, h.draw_ids[order].copy(), h.dates[order].copy(), obs)


def load_series(product: str | ProductCode, repository=None, store=None, seed_dir: Path | None = None) -> Series:  # type: ignore[no-untyped-def]
    """The engine's current data for ``product``: the draw repository for Mega/Power/Lotto
    (falling back to the bundled seed when it is empty), the product store for the rest."""
    code = get_product(product)
    if code in (ProductCode.MEGA_645, ProductCode.POWER_655, ProductCode.LOTTO_535):
        spec = get_game(GameCode(code.value))
        h = repository.load_history(spec.code) if repository is not None else None
        if h is None or len(h) == 0:
            from vietlott_engine.crawler.sources.mirror import JsonlFileSource

            res = asyncio.run(JsonlFileSource(Path(seed_dir or "data/seed")).fetch(spec))
            h = DrawHistory.from_draws(res.draws, spec)
        return matrix_series(h)
    if store is None:
        from vietlott_engine.crawler.product_store import ProductStore

        store = ProductStore(Path("data"), Path(seed_dir or "data/seed"))
    return product_series(store.load(code))
