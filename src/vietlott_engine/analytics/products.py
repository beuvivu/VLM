"""Randomness tests for Keno, Bingo18, Max 3D, Max 3D Pro and Max 4D histories.

Each test compares the history with the exact law of a fair draw; p-values are adjusted
together with Benjamini–Hochberg. With tens of thousands of draws these tests detect
very small biases (e.g. Keno: a number drawn 1 % more often than 1/4 of the time).
"""

from __future__ import annotations

from math import comb

import numpy as np
from pydantic import BaseModel
from scipy import stats

from vietlott_engine.core.products import ProductCode, ProductHistory
from vietlott_engine.game_theory.fastgames import bingo18_sum_distribution


class ProductTest(BaseModel):
    name: str
    statistic: float
    df: int | None
    p_value: float
    q_value: float = 1.0
    detail: str = ""


class ProductRandomnessReport(BaseModel):
    product: str
    draws: int
    first_date: str
    last_date: str
    consecutive_pairs: int
    tests: list[ProductTest]
    min_q_value: float
    verdict: str


def _g_test(observed: np.ndarray, expected_p: np.ndarray, min_expected: float = 5.0) -> tuple[float, int, float]:
    """G-test with tail cells pooled until every expected count ≥ ``min_expected``."""
    n = observed.sum()
    exp = expected_p * n
    obs = observed.astype(float)
    # pool from both ends towards the centre
    o_c, e_c = [], []
    acc_o = acc_e = 0.0
    for o, e in zip(obs, exp):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            o_c.append(acc_o)
            e_c.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0 and e_c:
        o_c[-1] += acc_o
        e_c[-1] += acc_e
    o_a, e_a = np.array(o_c), np.array(e_c)
    mask = o_a > 0
    g = 2.0 * float(np.sum(o_a[mask] * np.log(o_a[mask] / e_a[mask])))
    df = max(len(o_a) - 1, 1)
    return g, df, float(stats.chi2.sf(g, df))


def _consecutive(h: ProductHistory) -> np.ndarray:
    """Indices t ≥ 1 whose previous row is the previous draw (id difference 1)."""
    return np.flatnonzero(np.diff(h.draw_ids) == 1) + 1


def _bh(tests: list[ProductTest]) -> None:
    p = np.array([t.p_value for t in tests])
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    for t, qq in zip(tests, q):
        t.q_value = float(min(qq, 1.0))


def _chi2_uniform(counts: np.ndarray, factor: float = 1.0) -> tuple[float, int, float]:
    e = counts.sum() / counts.size
    chi = float(np.sum((counts - e) ** 2 / e) * factor)
    df = counts.size - 1
    return chi, df, float(stats.chi2.sf(chi, df))


# ------------------------------------------------------------------ Keno
def keno_tests(h: ProductHistory) -> list[ProductTest]:
    v = h.values.astype(int)
    d = len(v)
    tests = []
    counts = np.bincount(v.ravel(), minlength=81)[1:]
    chi, df, p = _chi2_uniform(counts, factor=(80 - 1) / (80 - 20))  # without-replacement correction
    hot, cold = int(np.argmax(counts)) + 1, int(np.argmin(counts)) + 1
    tests.append(ProductTest(name="Tần suất 80 số (χ² hiệu chỉnh)", statistic=chi, df=df, p_value=p, detail=f"nhiều nhất {hot} ({counts.max() / d:.4f}/kỳ), ít nhất {cold} ({counts.min() / d:.4f}/kỳ), kỳ vọng 0.2500"))
    law = np.array([comb(40, x) * comb(40, 20 - x) / comb(80, 20) for x in range(21)])
    large = (v > 40).sum(axis=1)
    g, df, p = _g_test(np.bincount(large, minlength=21), law)
    tests.append(ProductTest(name="Số lượng số Lớn (41–80) mỗi kỳ", statistic=g, df=df, p_value=p, detail=f"P(≥13) thực tế {np.mean(large >= 13):.4f} vs 0.0980; P(=10) {np.mean(large == 10):.4f} vs 0.2032"))
    even = (v % 2 == 0).sum(axis=1)
    g, df, p = _g_test(np.bincount(even, minlength=21), law)
    tests.append(ProductTest(name="Số lượng số Chẵn mỗi kỳ", statistic=g, df=df, p_value=p, detail=f"P(≥13) thực tế {np.mean(even >= 13):.4f} vs 0.0980"))
    cons = _consecutive(h)
    if len(cons) > 100:
        masks = np.zeros((d, 81), dtype=bool)
        masks[np.arange(d)[:, None], v] = True
        overlap = (masks[cons] & masks[cons - 1]).sum(axis=1)
        law_o = np.array([comb(20, x) * comb(60, 20 - x) / comb(80, 20) for x in range(21)])
        g, df, p = _g_test(np.bincount(overlap, minlength=21), law_o)
        tests.append(ProductTest(name="Số trùng với kỳ liền trước", statistic=g, df=df, p_value=p, detail=f"trung bình {overlap.mean():.3f} vs 5.000 ({len(cons):,} cặp kỳ liền nhau)"))
        r = float(np.corrcoef(large[cons], large[cons - 1])[0, 1])
        z = r * np.sqrt(len(cons))
        tests.append(ProductTest(name="Tương quan số Lớn giữa hai kỳ liền nhau", statistic=z, df=None, p_value=float(2 * stats.norm.sf(abs(z))), detail=f"r = {r:+.4f}"))
    return tests


# ------------------------------------------------------------------ Bingo18
def bingo18_tests(h: ProductHistory) -> list[ProductTest]:
    v = h.values.astype(int)
    d = len(v)
    tests = []
    chi, df, p = _chi2_uniform(np.bincount(v.ravel(), minlength=7)[1:])
    tests.append(ProductTest(name="Tần suất mặt 1–6 (gộp 3 vị trí)", statistic=chi, df=df, p_value=p, detail=", ".join(f"{k}: {c / (3 * d):.4f}" for k, c in enumerate(np.bincount(v.ravel(), minlength=7)[1:], start=1))))
    for pos in range(3):
        chi, df, p = _chi2_uniform(np.bincount(v[:, pos], minlength=7)[1:])
        tests.append(ProductTest(name=f"Tần suất mặt ở vị trí {pos + 1}", statistic=chi, df=df, p_value=p))
    sums = v.sum(axis=1)
    law = bingo18_sum_distribution()
    g, df, p = _g_test(np.bincount(sums, minlength=19)[3:], np.array([law[s] for s in range(3, 19)]))
    tests.append(ProductTest(name="Phân phối tổng 3–18", statistic=g, df=df, p_value=p, detail=f"P(Nhỏ 3–9) {np.mean(sums <= 9):.4f} vs 0.3750; P(Hòa) {np.mean((sums >= 10) & (sums <= 11)):.4f} vs 0.2500"))
    triples = int(np.sum((v[:, 0] == v[:, 1]) & (v[:, 1] == v[:, 2])))
    p = float(stats.binomtest(triples, d, 6 / 216).pvalue)
    tests.append(ProductTest(name="Số kỳ ra bộ ba", statistic=triples, df=None, p_value=p, detail=f"{triples / d:.4f} vs {6 / 216:.4f}"))
    table = np.zeros((6, 6))
    np.add.at(table, (v[:, 0] - 1, v[:, 1] - 1), 1)
    chi, p, df, _ = stats.chi2_contingency(table, correction=False)
    tests.append(ProductTest(name="Độc lập giữa vị trí 1 và 2", statistic=float(chi), df=int(df), p_value=float(p)))
    cons = _consecutive(h)
    if len(cons) > 100:
        t2 = np.zeros((6, 6))
        np.add.at(t2, (v[cons - 1, 0] - 1, v[cons, 0] - 1), 1)
        chi, p, df, _ = stats.chi2_contingency(t2, correction=False)
        tests.append(ProductTest(name="Phụ thuộc giữa hai kỳ liền nhau (vị trí 1)", statistic=float(chi), df=int(df), p_value=float(p), detail=f"{len(cons):,} cặp kỳ"))
        r = float(np.corrcoef(sums[cons], sums[cons - 1])[0, 1])
        z = r * np.sqrt(len(cons))
        tests.append(ProductTest(name="Tương quan tổng giữa hai kỳ liền nhau", statistic=z, df=None, p_value=float(2 * stats.norm.sf(abs(z))), detail=f"r = {r:+.4f}"))
    return tests


# ------------------------------------------------------------------ Max 3D / Max 4D
DIGIT_NAMES = {3: ("trăm", "chục", "đơn vị"), 4: ("nghìn", "trăm", "chục", "đơn vị")}


def digit_columns(v: np.ndarray, n_digits: int) -> list[np.ndarray]:
    """Digit arrays from the most significant position to the units."""
    return [(v // 10 ** (n_digits - 1 - k)) % 10 for k in range(n_digits)]


def digit_tests(h: ProductHistory, n_digits: int = 3) -> list[ProductTest]:
    v = h.values.astype(int)
    d, w = v.shape
    space = 10**n_digits
    tests = []
    for name, digit in zip(DIGIT_NAMES[n_digits], digit_columns(v, n_digits)):
        c = np.bincount(digit.ravel(), minlength=10)
        chi, df, p = _chi2_uniform(c)
        share = c / c.sum()
        tests.append(ProductTest(name=f"Chữ số hàng {name}", statistic=chi, df=df, p_value=p, detail=f"nhiều nhất {int(share.argmax())} ({share.max():.4f}), ít nhất {int(share.argmin())} ({share.min():.4f}), kỳ vọng 0.1000"))
    if w * d / space >= 5:
        chi, df, p = _chi2_uniform(np.bincount(v.ravel(), minlength=space))
        tests.append(ProductTest(name=f"Tần suất {space} số", statistic=chi, df=df, p_value=p, detail=f"{w * d:,} lần quay, kỳ vọng {w * d / space:.1f} lần/số"))
    if n_digits == 3:
        chi, df, p = _chi2_uniform(np.bincount(v[:, :2].ravel(), minlength=1000) if d * 2 >= 5000 else np.bincount(v[:, :2].ravel() // 100, minlength=10))
        tests.append(ProductTest(name="Hai số giải Đặc biệt (chữ số hàng trăm)" if d * 2 < 5000 else "Hai số giải Đặc biệt", statistic=chi, df=df, p_value=p))
    dup = 0
    for row in v:
        _, c = np.unique(row, return_counts=True)
        dup += int(np.sum(c * (c - 1) // 2))
    lam = d * comb(w, 2) / space
    p = float(2 * min(stats.poisson.cdf(dup, lam), stats.poisson.sf(dup - 1, lam)))
    tests.append(ProductTest(name="Số trùng nhau trong cùng kỳ", statistic=dup, df=None, p_value=min(p, 1.0), detail=f"{dup} cặp vs kỳ vọng {lam:.1f}"))
    cons = _consecutive(h)
    if len(cons) > 50:
        ov = sum(int(np.sum(np.isin(v[t], v[t - 1]))) for t in cons)
        lam = len(cons) * w * (1 - (1 - 1 / space) ** w)
        p = float(2 * min(stats.poisson.cdf(ov, lam), stats.poisson.sf(ov - 1, lam)))
        tests.append(ProductTest(name="Số lặp lại từ kỳ trước", statistic=ov, df=None, p_value=min(p, 1.0), detail=f"{ov} vs kỳ vọng {lam:.1f}"))
    return tests


def three_digit_tests(h: ProductHistory) -> list[ProductTest]:
    return digit_tests(h, 3)


class CellTest(BaseModel):
    """One pre-specified (position, digit) hypothesis tested on a new sample."""

    product: str
    position: str
    digit: int
    alternative: str
    hits: int
    trials: int
    share: float
    ci95: tuple[float, float]
    p_value: float
    note: str


def digit_cell_test(h: ProductHistory, n_digits: int, position_from_right: int, digit: int, alternative: str = "greater") -> CellTest:
    """Binomial test of a cell chosen *before* looking at ``h`` (e.g. the Max 3D units-digit 6)."""
    col = digit_columns(h.values.astype(int), n_digits)[n_digits - 1 - position_from_right].ravel()
    hits, n = int(np.sum(col == digit)), int(col.size)
    share = hits / n
    se = np.sqrt(share * (1 - share) / n)
    p = float(stats.binomtest(hits, n, 0.1, alternative=alternative).pvalue)
    return CellTest(
        product=h.product.value,
        position=DIGIT_NAMES[n_digits][n_digits - 1 - position_from_right],
        digit=digit,
        alternative=alternative,
        hits=hits,
        trials=n,
        share=share,
        ci95=(float(share - 1.96 * se), float(share + 1.96 * se)),
        p_value=p,
        note="Giả thuyết đặt trước khi xem dữ liệu này: một kiểm định, không hiệu chỉnh đa kiểm định.",
    )


POSITIONS = ("trăm", "chục", "đơn vị")


def _digits(v: np.ndarray) -> list[np.ndarray]:
    return [v // 100, (v // 10) % 10, v % 10]


class DigitCell(BaseModel):
    position: str
    digit: int
    discovery_share: float
    discovery_draws: int
    confirm_share: float
    confirm_hits: int
    confirm_trials: int
    confirm_p_value: float  # one-sided, in the direction seen in the discovery sample
    confirm_z: float


class DigitReplication(BaseModel):
    """Pick the most deviant (position, digit) cell in a discovery sample, then test only
    that cell — a single pre-specified hypothesis — on an independent sample."""

    discovery: str
    confirmation: str
    cell: DigitCell
    pooled_share: float
    pooled_ci95: tuple[float, float]
    per_number_multiplier_max: float  # largest P(number)/0.001 implied by pooled digit shares
    best_number: str
    note: str


def _most_deviant(v: np.ndarray) -> tuple[int, int, float]:
    best = (0, 0, 0.0)
    for pos, d in enumerate(_digits(v)):
        c = np.bincount(d.ravel(), minlength=10)
        z = (c - c.sum() / 10) / np.sqrt(c.sum() * 0.09)
        k = int(np.argmax(np.abs(z)))
        if abs(z[k]) > abs(best[2]):
            best = (pos, k, float(z[k]))
    return best


def digit_replication(discovery: ProductHistory, confirmation: ProductHistory, discovery_label: str | None = None, confirmation_label: str | None = None) -> DigitReplication:
    """Out-of-sample check of the strongest digit bias (Max 3D ↔ Max 3D Pro, or first half ↔
    second half). Selection happens on ``discovery`` only, so the confirmation p-value is
    honest; the pooled share and per-number multiplier are descriptive."""
    vd, vc = discovery.values.astype(int), confirmation.values.astype(int)
    pos, k, z = _most_deviant(vd)
    dd, dc = _digits(vd)[pos], _digits(vc)[pos]
    hits, n = int(np.sum(dc == k)), int(dc.size)
    alt = "greater" if z > 0 else "less"
    p = float(stats.binomtest(hits, n, 0.1, alternative=alt).pvalue)
    zc = (hits - 0.1 * n) / np.sqrt(n * 0.09)
    pooled = (int(np.sum(dd == k)) + hits) / (dd.size + n)
    se = np.sqrt(pooled * (1 - pooled) / (dd.size + n))
    both = np.vstack([vd, vc])
    shares = [np.bincount(d.ravel(), minlength=10) / d.size for d in _digits(both)]
    mult = np.einsum("i,j,k->ijk", *shares) * 1000
    best = np.unravel_index(int(np.argmax(mult)), mult.shape)
    return DigitReplication(
        discovery=discovery_label or discovery.product.value,
        confirmation=confirmation_label or confirmation.product.value,
        cell=DigitCell(
            position=POSITIONS[pos],
            digit=k,
            discovery_share=float(np.mean(dd == k)),
            discovery_draws=len(vd),
            confirm_share=hits / n,
            confirm_hits=hits,
            confirm_trials=n,
            confirm_p_value=p,
            confirm_z=float(zc),
        ),
        pooled_share=float(pooled),
        pooled_ci95=(float(pooled - 1.96 * se), float(pooled + 1.96 * se)),
        per_number_multiplier_max=float(mult.max()),
        best_number="".join(str(int(x)) for x in best),
        note=(
            "Lệch được xác nhận trên mẫu độc lập." if p < 0.01 else "Không xác nhận được trên mẫu độc lập — nhiều khả năng là ngẫu nhiên."
        )
        + f" Hệ số ×{mult.max():.3f} cho số lợi nhất là ước lượng trong mẫu (thiên lệch lên do chọn lọc); RTP Max 3D ≈ 0,545 nên vé chỉ hoà vốn khi xác suất tăng ≈ ×1,83 — lệch này chỉ đưa RTP số lợi nhất lên ≈ {0.545 * mult.max():.2f}.",
    )


def split_history(h: ProductHistory, fraction: float = 0.5) -> tuple[ProductHistory, ProductHistory]:
    cut = int(len(h) * fraction)
    part = lambda s: ProductHistory(h.product, h.draw_ids[s], h.dates[s], h.values[s], h.source)  # noqa: E731
    return part(slice(0, cut)), part(slice(cut, None))


def product_randomness(h: ProductHistory) -> ProductRandomnessReport:
    if h.product == ProductCode.KENO:
        tests = keno_tests(h)
    elif h.product == ProductCode.BINGO18:
        tests = bingo18_tests(h)
    elif h.product in (ProductCode.MAX3D, ProductCode.MAX3D_PRO):
        tests = digit_tests(h, 3)
    elif h.product == ProductCode.MAX4D:
        tests = digit_tests(h, 4)
    else:
        raise ValueError(f"use analytics.randomness for {h.product.value}")
    _bh(tests)
    min_q = min(t.q_value for t in tests)
    verdict = (
        "Không có kiểm định nào bác bỏ giả thuyết quay công bằng (FDR 5%)."
        if min_q > 0.05
        else "Có kiểm định bị bác bỏ ở FDR 5% — xem chi tiết, kiểm tra dữ liệu thiếu/trùng trước khi kết luận."
    )
    return ProductRandomnessReport(
        product=h.product.value,
        draws=len(h),
        first_date=str(h.dates.min()),
        last_date=str(h.dates.max()),
        consecutive_pairs=int(len(_consecutive(h))),
        tests=tests,
        min_q_value=float(min_q),
        verdict=verdict,
    )
