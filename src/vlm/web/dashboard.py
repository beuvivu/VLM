"""Join immutable pre-draw predictions to exact, validated results for the static site.

No fitting, network calls, ledger edits, or fabricated financial values occur here.
"""
from __future__ import annotations

import gzip
import heapq
import json
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterator

import numpy as np

from vietlott_engine.core.games import GAMES, get_game
from vietlott_engine.core.prizes import PrizeRecord
from vietlott_engine.core.products import PRODUCT_INFO, SEED_FILES, ProductCode, load_exclusions
from vietlott_engine.forecast.data import CONFIGS, SetSpec
from vietlott_engine.forecast.engine import Forecaster, state_path
from vietlott_engine.forecast.schedule import FAST, MARGIN, VN, WEEKLY, target_draw_time
from vietlott_engine.game_theory.fastgames import KENO_TABLE, KENO_SIDE_BETS, bingo18_odds
from vietlott_engine.game_theory.max3d import PRODUCTS as MAX_PRODUCTS
from vlm.database.schema import DrawRecord
from vlm.forecast.distribution import Law
from vlm.forecast.pipeline import MLForecaster

ACTIVE = tuple(c for c in ProductCode if c != ProductCode.MAX4D)
MATRIX_FILES = {'mega645': 'power645.jsonl', 'power655': 'power655.jsonl', 'lotto535': 'power535.jsonl'}
TIER_LABELS = {'jackpot1': 'Jackpot', 'jackpot2': 'Jackpot 2', 'first': 'Giải Nhất',
               'second': 'Giải Nhì', 'third': 'Giải Ba', 'fourth': 'Giải Tư', 'fifth': 'Giải Năm',
               'consolation': 'Khuyến khích'}
RECENT_LIMIT = 40


def _rank(source: str) -> int:
    return 3 if source.startswith('vietlott') else 2 if source.startswith('canonical') else 1


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings and len(warnings) < 20:
        warnings.append(message)


def _reconcile(old: DrawRecord, new: DrawRecord, warnings: list[str]) -> DrawRecord:
    key = f'{new.game_type.value} #{new.draw_id}'
    if (old.draw_date.date(), old.winning_numbers, old.bonus_number) != (new.draw_date.date(), new.winning_numbers, new.bonus_number):
        _warn(warnings, f'result_conflict: {key}; nguồn {old.source} / {new.source}')
        return new if _rank(new.source) >= _rank(old.source) else old
    primary, secondary = (new, old) if _rank(new.source) >= _rank(old.source) else (old, new)
    values = primary.model_dump()
    sub = {**(secondary.sub_prizes_json or {}), **(primary.sub_prizes_json or {})}
    sources = {**(secondary.sub_prizes_json or {}).get('field_sources', {}), **(primary.sub_prizes_json or {}).get('field_sources', {})}
    for field in ('jackpot1_value', 'jackpot2_value', 'jackpot1_winners', 'jackpot2_winners'):
        a, b = getattr(primary, field), getattr(secondary, field)
        if a is not None and b is not None and a != b:
            _warn(warnings, f'finance_conflict: {key} {field}')
        chosen = primary if a is not None else secondary
        values[field] = getattr(chosen, field)
        if values[field] is not None:
            sources[field] = (chosen.sub_prizes_json or {}).get('field_sources', {}).get(field, chosen.source)
    winners = dict((secondary.sub_prizes_json or {}).get('winners', {}))
    for tier, count in (primary.sub_prizes_json or {}).get('winners', {}).items():
        if count is not None:
            if winners.get(tier) is not None and winners[tier] != count:
                _warn(warnings, f'finance_conflict: {key} {tier} winners')
            winners[tier] = count
    if winners:
        sub['winners'] = winners
    if sources:
        sub['field_sources'] = sources
    values['sub_prizes_json'] = sub or None
    if primary.time_precision == 'day' and secondary.time_precision == 'second':
        values['draw_date'], values['time_precision'] = secondary.draw_date, 'second'
    return DrawRecord.model_validate(values)


def _rows(path: Path, warnings: list[str]) -> Iterator[dict]:
    if not path.exists():
        return
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('row is not an object')
                yield row
            except (ValueError, UnicodeError):
                if len(warnings) < 20:
                    warnings.append(f'{path.name}:{line_no}: bản ghi không hợp lệ, đã bỏ qua')


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError('forecast time is missing')
    when = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if when.tzinfo is None:
        raise ValueError('forecast time requires timezone')
    return when.astimezone(VN)


def _valid_issue(code: ProductCode, entry: dict, now: datetime) -> bool:
    try:
        if code in (*FAST, ProductCode.MAX4D) or entry.get('pre_draw') is False:
            return False
        target, made = _time(entry['target_time']), _time(entry['made_at'])
        based = date.fromisoformat(entry['based_on_date'])
        count = entry.get('draws_on_last_date', 1 if target.date() == based else 2)
        if code in WEEKLY and based.weekday() not in WEEKLY[code]:
            return False
        if code == ProductCode.LOTTO_535 and count not in (1, 2):
            return False
        previous_hour = (13 if count == 1 else 21) if code == ProductCode.LOTTO_535 else 18
        previous = datetime.combine(based, time(previous_hour), VN)
        return (type(entry['target_id']) is int and type(entry['based_on_id']) is int
                and entry['target_id'] == entry['based_on_id'] + 1
                and target_draw_time(code, based, count) == target
                and made >= previous
                and made <= now and made < target - MARGIN
                and entry.get('target_date', target.date().isoformat()) == target.date().isoformat())
    except (ValueError, KeyError, TypeError):
        return False


def _from_issue(code: ProductCode, event: dict, now: datetime) -> dict | None:
    p = event.get('pending', {})
    if (event.get('target_id') != p.get('target_id') or event.get('history_sha256') != p.get('history_sha256')
            or not _valid_issue(code, p, now)):
        return None
    cfg = CONFIGS[code]
    if set(p.get('laws', {})) != set(cfg.components):
        return None
    components = []
    for name, spec in cfg.components.items():
        raw = p['laws'][name]
        law = Law(spec, np.array(raw['values']), np.array(raw['mixture']))
        tops, exact = law.top(5, max_nodes=event.get('config', {}).get('search_nodes', 1000))
        components.append({'name': name, 'kind': spec.kind, 'top': tops, 'ranking_exact': exact})
    return {k: p[k] for k in ('target_id', 'target_date', 'target_time', 'made_at', 'based_on_id')} | {
        'product': code.value, 'engine': 'ml', 'registered': True, 'components': components,
        'status': 'registered', 'note': 'Bộ số tái dựng từ phân phối đã lưu trước kỳ quay.'}


def _from_legacy(code: ProductCode, event: dict, now: datetime) -> dict | None:
    p = {**event, 'target_time': event.get('target_draw_time')}
    if not _valid_issue(code, p, now):
        return None
    components = []
    for name, spec in CONFIGS[code].components.items():
        if isinstance(spec, SetSpec):
            values = [int(x) for x in (p.get('ticket') or p.get('picks', {}).get(name, []))]
            if len(values) != spec.k or len(set(values)) != spec.k or not all(1 <= x <= spec.n for x in values):
                return None
            top = [{'numbers': values}]
        else:
            symbols = p.get('top', {}).get(name, [])
            top = [{'numbers': ([int(s)] if spec.positions == 1 else [int(x) for x in str(s)])} for s in symbols[:5]]
            for row in top:
                if len(row['numbers']) != spec.positions or not all(spec.symbol_offset <= x < spec.alphabet + spec.symbol_offset for x in row['numbers']):
                    return None
        if not top:
            return None
        components.append({'name': name, 'kind': spec.kind, 'top': top, 'ranking_exact': None})
    return {'product': code.value, 'engine': 'legacy', 'registered': True, 'status': 'registered',
            'target_id': p['target_id'], 'target_date': _time(p['target_time']).date().isoformat(),
            'target_time': p['target_time'], 'made_at': p['made_at'], 'based_on_id': p['based_on_id'],
            'components': components, 'note': 'Bộ số từ sổ dự báo đã ghi trước kỳ quay.'}


def _predictions(directory: Path, now: datetime, warnings: list[str]) -> dict[tuple[str, int], dict]:
    chosen: dict[tuple[str, int], dict] = {}
    for filename, engine in (('ml-ledger.jsonl', 'ml'), ('ledger.jsonl', 'legacy')):
        candidates = []
        for e in _rows(directory / filename, warnings):
            try:
                code = ProductCode(e['product'])
                if engine == 'ml' and e.get('event') != 'issue':
                    continue
                p = _from_issue(code, e, now) if engine == 'ml' else _from_legacy(code, e, now)
                if p:
                    candidates.append(p)
            except (ValueError, KeyError, TypeError):
                warnings.append(f'{filename}: dự báo không hợp lệ, đã bỏ qua')
        for p in sorted(candidates, key=lambda row: _time(row['made_at'])):
            chosen.setdefault((p['product'], p['target_id']), p)
    return chosen


def compare_prediction(prediction: dict, draw: DrawRecord) -> dict:
    """Compare the stored candidate numbers, never a newly fitted forecast."""
    code = draw.game_type.value
    if (prediction['product'], prediction['target_id']) != (code, draw.draw_id):
        raise ValueError('Prediction product/draw does not match this result')
    out = {'status': 'matched', 'product': code, 'target_id': draw.draw_id,
           'target_date': prediction['target_date'], 'made_at': prediction['made_at'],
           'engine': prediction['engine'], 'registered': prediction['registered'], 'tickets': []}
    if draw.draw_date.date().isoformat() != prediction['target_date']:
        return {**out, 'status': 'date_mismatch'}
    components = {c['name']: c for c in prediction['components']}
    if code in GAMES or code == 'keno':
        for row in components['main']['top']:
            nums = row['numbers']
            hits = sorted(set(nums) & set(draw.winning_numbers))
            special = components.get('special', {}).get('top', [{}])[0].get('numbers', [None])[0]
            bonus_hit = draw.bonus_number in nums if code == 'power655' else special == draw.bonus_number if code == 'lotto535' else False
            tier = get_game(code).classify(len(hits), bonus_hit) if code in GAMES else None
            out['tickets'].append({'numbers': nums, 'matched_numbers': hits, 'hits': len(hits),
                'bonus_hit': bonus_hit, 'special': special, 'tier': tier.name if tier else None})
    elif code.startswith('max'):
        prod = MAX_PRODUCTS[code]
        group_ranges = ((0, 2), (2, 6), (6, 12), (12, 20))
        for row in components['digits']['top']:
            symbol = ''.join(str(n) for n in row['numbers'])
            tiers = [label for (lo, hi), label in zip(group_ranges, prod.group_labels.values()) if symbol in draw.winning_numbers[lo:hi]]
            out['tickets'].append({'numbers': row['numbers'], 'symbol': symbol, 'tiers': tiers,
                                   'hits': draw.winning_numbers.count(symbol)})
    else:
        for row in components['dice']['top']:
            nums = row['numbers']
            positions = [a == b for a, b in zip(nums, draw.winning_numbers)]
            out['tickets'].append({'numbers': nums, 'position_matches': positions, 'position_hits': sum(positions),
                'multiset_hits': sum((Counter(nums) & Counter(draw.winning_numbers)).values()),
                'sum_match': sum(nums) == sum(draw.winning_numbers), 'exact': all(positions)})
    return out


def prize_catalogue(code: ProductCode) -> list[dict]:
    if code.value in GAMES:
        spec = get_game(code.value)
        return [{'label': ('Jackpot 1' if t.name == 'jackpot1' and code == ProductCode.POWER_655 else TIER_LABELS[t.name]),
                 'condition': (f'{t.main_matches}–{t.main_matches_max}' if t.main_matches_max is not None else str(t.main_matches)) + ' số chính' + (' + số đặc biệt/phụ' if t.bonus_required else ''),
                 'value_vnd': t.fixed_amount, 'code': t.name} for t in spec.tiers]
    if code in (ProductCode.MAX3D, ProductCode.MAX3D_PRO):
        products = ('max3d', 'max3dplus') if code == ProductCode.MAX3D else ('max3dpro',)
        return [{'product': MAX_PRODUCTS[p].display_name, 'label': {'nhat':'Nhất','nhi':'Nhì','ba':'Ba','tu':'Tư','nam':'Năm','sau':'Sáu','bay':'Bảy','khuyen_khich':'Khuyến khích','dac_biet':'Đặc biệt','phu_dac_biet':'Phụ đặc biệt'}.get(t.name,t.name),
                 'condition': t.rule, 'value_vnd': t.value,
                 'note': 'Giải cao nhất chịu trần tổng trả thưởng của kỳ.' if i == 0 and MAX_PRODUCTS[p].top_prize_cap else None}
                for p in products for i, t in enumerate(MAX_PRODUCTS[p].tiers)]
    if code == ProductCode.KENO:
        rows = [{'label': f'Bậc {bac}', 'condition': f'Trùng {hits} số', 'value_vnd': value,
                 'note': 'Giá trị chưa thống nhất giữa các bảng nguồn (0 / 150.000 đ).' if (bac, hits) == (5, 4) else None}
                for bac, prizes in KENO_TABLE.items() for hits, value in sorted(prizes.items())]
        rows += [{'label': b.name, 'condition': f'{lo}–{hi} số; {b.condition}', 'value_vnd': pay}
                 for b in KENO_SIDE_BETS for lo, hi, pay in b.tiers]
        return rows
    return [{'label': b.bet, 'condition': b.condition, 'value_text': b.payout, 'value_vnd': None} for b in bingo18_odds()]


def _result(record: DrawRecord, finance: PrizeRecord | None, warnings: list[str] | None = None) -> dict:
    code = record.game_type.value
    if finance is not None and finance.draw_date != record.draw_date.date():
        finance = None
    prizes = []
    if code in GAMES:
        winners = dict((record.sub_prizes_json or {}).get('winners', {}))
        pots = {'jackpot1': record.jackpot1_value, 'jackpot2': record.jackpot2_value}
        field_sources = (record.sub_prizes_json or {}).get('field_sources', {})
        pot_sources = {tier: field_sources.get(tier + '_value', record.source) for tier in pots}
        if finance is not None:
            for incoming, known, sources in ((finance.winners, winners, {}), (finance.jackpot_pots or {}, pots, pot_sources)):
                for tier, value in incoming.items():
                    if value is None:
                        continue
                    if known.get(tier) is not None and known[tier] != value and warnings is not None:
                        _warn(warnings, f'finance_conflict: {code} #{record.draw_id} {tier}')
                    if known.get(tier) is None or _rank(finance.source) >= _rank(sources.get(tier, record.source)):
                        known[tier] = value
                        sources[tier] = finance.source
        for tier in prize_catalogue(ProductCode(code)):
            count = winners.get(tier['code'])
            if count is None and tier['code'] in ('jackpot1', 'jackpot2'):
                count = getattr(record, tier['code'] + '_winners')
            prizes.append({**tier, 'value_vnd': pots.get(tier['code']) if tier['value_vnd'] is None else tier['value_vnd'],
                           'winners': count, 'pool': tier['value_vnd'] is None, 'source': pot_sources.get(tier['code']) if tier['value_vnd'] is None else 'catalogue'})
    elif code.startswith('max'):
        prod = MAX_PRODUCTS[code]
        for (lo, hi), label in zip(((0, 2), (2, 6), (6, 12), (12, 20)), prod.group_labels.values()):
            prizes.append({'label': label, 'numbers': list(record.winning_numbers[lo:hi])})
    nums = list(record.winning_numbers)
    facts = {}
    if code == 'keno':
        facts = {'large': sum(n > 40 for n in nums), 'small': sum(n <= 40 for n in nums),
                 'even': sum(n % 2 == 0 for n in nums), 'odd': sum(n % 2 != 0 for n in nums)}
    if code == 'bingo18':
        facts = {'sum': sum(nums), 'multiplicity': {str(n): k for n, k in sorted(Counter(nums).items())},
                 'size': 'Nhỏ' if sum(nums) < 10 else 'Hòa' if sum(nums) <= 11 else 'Lớn'}
    info = PRODUCT_INFO[ProductCode(code)]
    return {'draw_id': record.draw_id, 'draw_date': record.draw_date.isoformat(), 'numbers': nums,
            'bonus': record.bonus_number, 'time_precision': record.time_precision,
            'source': record.source, 'source_url': record.source_url,
            'official_url': 'https://vietlott.vn' + info.results_path,
            'official_direct': record.source.startswith('vietlott.vn') or record.source == 'vietlott',
            'prizes': prizes, 'prize_source': finance.source if finance else None, 'facts': facts}


def _load_results(data_dir: Path, seed_dir: Path, journal: Path, tracked: set[tuple[str, int]], now: datetime, warnings: list[str]):
    records: dict[str, dict[int, DrawRecord]] = {c.value: {} for c in ACTIVE}
    heaps: dict[str, list[int]] = {c.value: [] for c in ACTIVE}
    excluded = load_exclusions(seed_dir, data_dir / 'products')

    def add(code: str, row: dict):
        try:
            did = int(str(row.get('draw_id', row.get('id'))).lstrip('#'))
            if did in excluded.get(code, ()) or row.get('draw_status', 'confirmed') != 'confirmed':
                return
            known, heap = records[code], heaps[code]
            if len(heap) >= RECENT_LIMIT and did < heap[0] and (code, did) not in tracked:
                return
            rec = DrawRecord.model_validate(row) if 'game_type' in row else DrawRecord.from_legacy(code, row)
            if rec.draw_date > now:
                return
            if did not in known and (code, did) not in tracked:
                heapq.heappush(heap, did)
            known[did] = _reconcile(known[did], rec, warnings) if did in known else rec
            if len(heap) > RECENT_LIMIT:
                known.pop(heapq.heappop(heap), None)
        except (ValueError, TypeError, KeyError):
            if len(warnings) < 20:
                warnings.append(f'{code}: kết quả không hợp lệ, đã bỏ qua')

    for code in ACTIVE:
        filename = MATRIX_FILES.get(code.value, SEED_FILES.get(code))
        for path in (seed_dir / filename, data_dir / 'products' / filename):
            for row in _rows(path, warnings):
                row.setdefault('source', row.get('src', 'seed' if path.parent == seed_dir else 'local-store'))
                add(code.value, row)
    database = data_dir / 'vietlott.duckdb'
    finance: dict[tuple[str, int], PrizeRecord] = {}
    if database.exists():
        import duckdb
        try:
            with duckdb.connect(str(database), read_only=True) as conn:
                for row in conn.execute('SELECT game,draw_id,draw_date,numbers,bonus,jackpot1_value,jackpot2_value,source FROM draws').fetchall():
                    add(row[0], {'id': row[1], 'date': str(row[2]), 'result': list(row[3]), 'bonus_number': row[4],
                        'jackpot1_value': row[5], 'jackpot2_value': row[6], 'source': row[7] or 'unknown'})
                for row in conn.execute('SELECT game,draw_id,draw_date,winners,jackpot_pots,source FROM prizes').fetchall():
                    if row[1] in records.get(row[0], {}):
                        finance[(row[0], row[1])] = PrizeRecord(game=row[0], draw_id=row[1], draw_date=row[2],
                            winners=json.loads(row[3]), jackpot_pots=json.loads(row[4]) if row[4] else None, source=row[5])
        except (duckdb.Error, ValueError):
            warnings.append('Cache DuckDB không đọc được; dùng seed và journal đã lưu.')
    for row in _rows(journal, warnings):
        if row.get('game_type') in records:
            add(row['game_type'], row)
    for code in MATRIX_FILES:
        for row in _rows(seed_dir / f'prizes_{code}.jsonl', warnings):
            if row.get('draw_id') in records[code]:
                try:
                    p = PrizeRecord.model_validate(row)
                    finance.setdefault((code, p.draw_id), p)
                except ValueError:
                    warnings.append(f'{code}: bảng giải không hợp lệ')
    return records, finance


def _fallback(code: ProductCode, latest: DrawRecord, recent: list[DrawRecord], directory: Path, now: datetime, warnings: list[str]) -> dict | None:
    try:
        path = directory / 'ml' / f'{code.value}.json.gz'
        components, engine, made_at = [], 'ml', now.isoformat()
        if path.exists():
            model = MLForecaster.load(path)
            if model.last_id != latest.draw_id or model.last_date != latest.draw_date.date().isoformat():
                return None
            rep = model.report()
            components = rep['components']
            made_at = rep['made_at']
        elif state_path(directory, code).exists():
            rep = Forecaster.load(state_path(directory, code)).forecast().model_dump(mode='json')
            if rep['last_id'] != latest.draw_id or rep['last_date'] != latest.draw_date.date().isoformat():
                return None
            engine, made_at = 'legacy', rep['made_at']
            for c in rep['components']:
                if c.get('set'):
                    top = [{'numbers': c['set']['ticket'], 'p_model': c['set']['p_ticket_model'], 'p_fair': c['set']['p_ticket_fair']}]
                else:
                    sp = CONFIGS[code].components[c['name']]
                    top = [{'numbers': [int(x['symbol'])] if sp.positions == 1 else [int(n) for n in x['symbol']]} for x in c['digit']['top_numbers'][:5]]
                components.append({'name': c['name'], 'kind': 'set' if c.get('set') else 'digit', 'top': top})
        else:
            return None
        on_day = sum(r.draw_date.date() == latest.draw_date.date() for r in recent)
        target = None if code in FAST else target_draw_time(code, latest.draw_date.date(), on_day)
        if target is not None and now >= target - MARGIN:
            return None
        return {'product': code.value, 'target_id': latest.draw_id + 1, 'target_date': target.date().isoformat() if target else None,
            'target_time': target.isoformat() if target else None, 'made_at': made_at, 'based_on_id': latest.draw_id,
            'engine': engine, 'registered': False, 'status': 'reference', 'components': components,
            'note': 'Dự báo tham khảo; chưa xác minh thời điểm từng kỳ.' if code in FAST else 'Dự báo tham khảo, chưa có bản đăng ký trước kỳ trong sổ.'}
    except (ValueError, KeyError, TypeError, OSError):
        warnings.append(f'{code.value}: checkpoint không đọc được; không dựng dự báo thay thế.')
        return None


def build_dashboard(data_dir: Path, seed_dir: Path, directory: Path, journal: Path | None = None, now: datetime | None = None) -> dict:
    now = (now or datetime.now(VN)).astimezone(VN)
    warnings: list[str] = []
    predictions = _predictions(directory, now, warnings)
    records, finance = _load_results(data_dir, seed_dir, journal or data_dir/'results'/'results.jsonl', set(predictions), now, warnings)
    products = []
    for code in ACTIVE:
        known = records[code.value]
        recent = sorted(known.values(), key=lambda r: r.draw_id, reverse=True)
        latest = recent[0] if recent else None
        draws = [_result(r, finance.get((code.value, r.draw_id)), warnings) for r in recent[:RECENT_LIMIT]]
        comparisons = []
        for (product, did), prediction in predictions.items():
            if product == code.value and did in known:
                result = compare_prediction(prediction, known[did])
                result['result'] = _result(known[did], finance.get((product, did)), warnings)
                comparisons.append(result)
            elif product == code.value and now >= _time(prediction['target_time']) - MARGIN:
                comparisons.append({**prediction, 'status': 'pending', 'tickets': []})
        nxt = predictions.get((code.value, latest.draw_id + 1)) if latest else None
        if nxt and now >= _time(nxt['target_time']) - MARGIN:
            nxt = None
        if latest and nxt is None:
            nxt = _fallback(code, latest, recent, directory, now, warnings)
        info = PRODUCT_INFO[code]
        products.append({'product': code.value, 'name': info.display_name, 'schedule': info.schedule,
            'official_url': 'https://vietlott.vn' + info.results_path,
            'latest': draws[0] if draws else None, 'draws': draws, 'prize_catalogue': prize_catalogue(code),
            'next_forecast': nxt, 'comparisons': sorted(comparisons, key=lambda x: x['target_id'], reverse=True)[:30]})
    return {'schema_version': 1, 'generated_at': now.isoformat(), 'products': products, 'warnings': warnings,
            'stats': {'products': len(ACTIVE), 'results': sum(p['latest'] is not None for p in products),
                      'registered_next': sum(bool(p['next_forecast'] and p['next_forecast']['registered']) for p in products),
                      'compared_draws': sum(c['status'] == 'matched' for p in products for c in p['comparisons'])}}
