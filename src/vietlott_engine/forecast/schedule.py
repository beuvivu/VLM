"""When may a forecast be written to the ledger? Only before the draw it targets.

A forecast targets the draw right after the last one in the data. It counts as *pre-registered*
only if, by Vietlott's published schedule (Vietnam time), that draw has not started yet:

* Mega 6/45 — Wed, Fri, Sun 18:00 · Power 6/55 — Tue, Thu, Sat 18:00
* Max 3D / 3D+ — Mon, Wed, Fri 18:00 · Max 3D Pro — Tue, Thu, Sat 18:00
* Lotto 5/35 — every day 13:00 and 21:00
* Keno / Bingo18 — every few minutes from 06:00 to about 22:00: draw times are not in the data,
  so a forecast is recorded only in the overnight gap (after 22:15, before 05:55) and only when
  the day in the data looks complete (at least 98 % of the usual number of draws per day).
* Max 4D — discontinued in 2021: never recorded.

``VQE_FORECAST_NOW`` (ISO 8601, e.g. ``2026-09-30T20:00:00+07:00``) fixes "now" for tests and replays.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from vietlott_engine.core.products import ProductCode

VN = timezone(timedelta(hours=7))
WEEKLY = {  # Monday = 0, draw at 18:00
    ProductCode.MEGA_645: (2, 4, 6),
    ProductCode.POWER_655: (1, 3, 5),
    ProductCode.MAX3D: (0, 2, 4),
    ProductCode.MAX3D_PRO: (1, 3, 5),
}
FAST = (ProductCode.KENO, ProductCode.BINGO18)
MARGIN = timedelta(minutes=5)


def now_vn() -> datetime:
    fixed = os.environ.get("VQE_FORECAST_NOW")
    if fixed:
        t = datetime.fromisoformat(fixed)
        return (t if t.tzinfo else t.replace(tzinfo=VN)).astimezone(VN)
    return datetime.now(VN)


def target_draw_time(code: ProductCode, last_date: date, draws_on_last_date: int) -> datetime | None:
    """Scheduled start of the draw after the last one in the data (None for Max 4D)."""
    if code in WEEKLY:
        d = last_date + timedelta(days=1)
        while d.weekday() not in WEEKLY[code]:
            d += timedelta(days=1)
        return datetime.combine(d, time(18, 0), VN)
    if code == ProductCode.LOTTO_535:
        if draws_on_last_date >= 2:
            return datetime.combine(last_date + timedelta(days=1), time(13, 0), VN)
        return datetime.combine(last_date, time(21, 0), VN)
    if code in FAST:
        return datetime.combine(last_date + timedelta(days=1), time(6, 0), VN)
    return None


def record_window(code: ProductCode, dates: np.ndarray, now: datetime | None = None) -> tuple[bool, str, datetime | None]:
    """(may record, reason in Vietnamese, target draw time) for a history whose draw dates are ``dates``."""
    if len(dates) == 0:
        return False, "chưa có dữ liệu", None
    now = now or now_vn()
    last = np.datetime64(dates[-1], "D").astype(date)
    on_last = int((np.asarray(dates, dtype="datetime64[D]") == np.datetime64(last)).sum())
    target = target_draw_time(code, last, on_last)
    if target is None:
        return False, "sản phẩm đã ngừng phát hành: không có kỳ tới để dự báo", None
    if code in FAST:
        day_end = datetime.combine(last, time(22, 15), VN)
        if now < day_end:
            return False, f"Keno/Bingo18 chỉ ghi dự báo trong khoảng 22:15–05:55, khi các kỳ của ngày {last} đã quay xong", target
        days = np.asarray(dates, dtype="datetime64[D]")
        window = days[(days < np.datetime64(last)) & (days >= np.datetime64(last) - np.timedelta64(14, "D"))]
        if window.size:
            _, counts = np.unique(window, return_counts=True)
            usual = float(np.median(counts))
            if on_last < 0.98 * usual:
                return False, f"dữ liệu ngày {last} mới có {on_last} kỳ (thường {usual:.0f}): còn thiếu kỳ, đồng bộ lại trước khi dự báo", target
    if now >= target - MARGIN:
        return False, f"kỳ được dự báo (lịch quay {target:%d/%m/%Y %H:%M}) đã hoặc sắp quay trong khi dữ liệu dừng ở {last}: đồng bộ trước rồi dự báo lại", target
    return True, f"ghi trước kỳ quay {target:%d/%m/%Y %H:%M} (giờ Việt Nam)", target
