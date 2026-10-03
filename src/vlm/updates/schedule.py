"""Polling windows and conservative freshness, independent of machine timezone.

18:00 weekly draws finish by 18:30. Lotto uses a ten-minute publication grace.
Fast games are polled every two minutes; no draw ID is inferred from a clock.
"""
from datetime import date, datetime, time, timedelta

from vietlott_engine.core.products import ProductCode, get_product
from vietlott_engine.forecast.schedule import VN, WEEKLY

LIVE_PRODUCTS = ('mega645', 'power655', 'lotto535', 'max3d', 'max3dpro', 'keno', 'bingo18')
FAST = (ProductCode.KENO, ProductCode.BINGO18)


def vietnam_time(now: datetime) -> datetime:
    return (now if now.tzinfo else now.replace(tzinfo=VN)).astimezone(VN)


def latest_completed_slot(product: str, now: datetime) -> datetime | None:
    code, now = get_product(product), vietnam_time(now)
    if code in FAST or code == ProductCode.MAX4D:
        return None
    for back in range(8):
        day = now.date() - timedelta(days=back)
        if code in WEEKLY and day.weekday() not in WEEKLY[code]:
            continue
        times = (time(21, 10), time(13, 10)) if code == ProductCode.LOTTO_535 else (time(18, 30),)
        for clock in times:
            slot = datetime.combine(day, clock, VN)
            if slot <= now:
                return slot
    raise ValueError(f'No scheduled slot for {product}')


def poll_interval(product: str, now: datetime) -> int:
    code, now = get_product(product), vietnam_time(now)
    minute = now.hour * 60 + now.minute
    if code in FAST:
        live = 360 <= minute < 1335  # 06:00–22:15 includes final publication/recovery
    elif code == ProductCode.LOTTO_535:
        live = 780 <= minute < 840 or 1260 <= minute < 1320
    else:
        live = code in WEEKLY and now.weekday() in WEEKLY[code] and 1080 <= minute < 1140
    return 120 if live else 3600


def slot_key(product: str, now: datetime) -> str:
    now = vietnam_time(now)
    slot = latest_completed_slot(product, now)
    return f'{slot.isoformat() if slot else now.date()}:{poll_interval(product, now)}'


def freshness(product: str, now: datetime, last_date: str | None, draws_on_last_date: int = 0) -> dict:
    now = vietnam_time(now)
    code = get_product(product)
    slot = latest_completed_slot(product, now)
    expected = slot.date() if slot else now.date() if now.hour >= 6 else now.date() - timedelta(days=1)
    observed = date.fromisoformat(last_date[:10]) if last_date else None
    if observed and code in WEEKLY and observed.weekday() not in WEEKLY[code]:
        return {'status':'invalid_schedule', 'verified':False, 'expected_through':slot.isoformat(),
                'note':'Stored result date is outside this product draw schedule'}
    required_count = 2 if code == ProductCode.LOTTO_535 and slot.hour == 21 else 1
    behind = observed is None or observed < expected or (observed == expected and draws_on_last_date < required_count)
    if observed and observed > now.date():
        behind = True
    if observed and code not in FAST and (observed > expected or
        (code == ProductCode.LOTTO_535 and observed == expected and required_count == 1 and draws_on_last_date > 1)):
        return {'status':'awaiting_publication', 'verified':False, 'expected_through':slot.isoformat(), 'note':None}
    verified = not behind and code not in FAST and code != ProductCode.MAX4D
    return {'status': 'behind_schedule' if behind else 'date_only' if code in FAST else 'caught_up',
            'verified': verified, 'expected_through': slot.isoformat() if slot else expected.isoformat(),
            'note': 'Intraday draw time/latest ID not certified from date-only rows' if code in FAST else None}
