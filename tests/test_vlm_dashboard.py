"""Dashboard tests exercise draw pairing, frozen forecasts and nullable prize data."""
import json
from datetime import datetime

import numpy as np
import pytest

from vlm.database.schema import DrawRecord, VN


def prediction(game='mega645', numbers=None, target_date='2026-10-04'):
    return {'product': game, 'target_id': 2, 'target_date': target_date,
            'target_time': target_date + 'T18:00:00+07:00', 'made_at': '2026-10-03T08:00:00+07:00',
            'registered': True, 'engine': 'ml', 'based_on_id': 1,
            'components': [{'name': 'main', 'kind': 'set', 'top': [{'numbers': numbers or [1, 2, 3, 4, 5, 6]}]}]}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_matrix_comparison_pairs_exact_game_and_id():
    from vlm.web.dashboard import compare_prediction
    draw = DrawRecord.from_legacy('mega645', {'id': 2, 'date': '2026-10-04', 'result': [1, 2, 3, 4, 5, 7]})
    result = compare_prediction(prediction(), draw)
    assert result['status'] == 'matched'
    assert result['tickets'][0]['hits'] == 5
    assert result['tickets'][0]['matched_numbers'] == [1, 2, 3, 4, 5]
    assert result['tickets'][0]['tier'] == 'first'
    with pytest.raises(ValueError, match='product|draw'):
        compare_prediction({**prediction(), 'target_id': 3}, draw)
    with pytest.raises(ValueError, match='product|draw'):
        compare_prediction({**prediction(), 'product': 'power655'}, draw)
    assert compare_prediction(prediction(target_date='2026-10-05'), draw)['status'] == 'date_mismatch'


def test_power_bonus_changes_tier_without_counting_it_as_main():
    from vlm.web.dashboard import compare_prediction
    draw = DrawRecord.from_legacy('power655', {'id': 2, 'date': '2026-10-04', 'result': [1, 2, 3, 4, 5, 7, 6]})
    ticket = compare_prediction(prediction('power655'), draw)['tickets'][0]
    assert (ticket['hits'], ticket['bonus_hit'], ticket['tier']) == (5, True, 'jackpot2')


def test_lotto_compares_special_number_from_the_frozen_forecast():
    from vlm.web.dashboard import compare_prediction
    pred = prediction('lotto535', [1, 2, 3, 4, 5])
    pred['components'].append({'name': 'special', 'kind': 'digit', 'top': [{'numbers': [9]}]})
    draw = DrawRecord.from_legacy('lotto535', {'id': 2, 'date': '2026-10-04', 'result': [1, 2, 3, 4, 7, 9]})
    ticket = compare_prediction(pred, draw)['tickets'][0]
    assert (ticket['hits'], ticket['bonus_hit'], ticket['tier']) == (4, True, 'second')


def test_max_digits_match_full_padded_number_and_every_tier():
    from vlm.web.dashboard import compare_prediction
    nums = ['016', '122', '016', '222', '333', '444'] + [f'{x:03d}' for x in range(100, 114)]
    pred = prediction('max3d')
    pred['components'] = [{'name': 'digits', 'kind': 'digit', 'top': [{'numbers': [0, 1, 6]}, {'numbers': [0, 2, 6]}]}]
    draw = DrawRecord.from_legacy('max3d', {'id': 2, 'date': '2026-10-04', 'result': nums})
    tickets = compare_prediction(pred, draw)['tickets']
    assert tickets[0]['symbol'] == '016'
    assert tickets[0]['hits'] == 2
    assert tickets[0]['tiers'] == ['Nhất', 'Nhì']
    assert tickets[1]['hits'] == 0


def test_bingo_repeated_dice_keep_order_and_multiplicity():
    from vlm.web.dashboard import compare_prediction
    pred = prediction('bingo18')
    pred['components'] = [{'name': 'dice', 'kind': 'digit', 'top': [{'numbers': [1, 1, 6]}]}]
    draw = DrawRecord.from_legacy('bingo18', {'id': 2, 'date': '2026-10-04', 'result': [1, 6, 1]})
    ticket = compare_prediction(pred, draw)['tickets'][0]
    assert ticket['position_hits'] == 1
    assert ticket['multiset_hits'] == 3
    assert ticket['exact'] is False
    assert ticket['sum_match'] is True


def test_keno_reference_is_not_settled_as_a_twenty_number_ticket():
    from vlm.web.dashboard import compare_prediction
    draw = DrawRecord.from_legacy('keno', {'id': 2, 'date': '2026-10-04', 'result': list(range(2, 22))})
    row = compare_prediction(prediction('keno', list(range(1, 21))), draw)['tickets'][0]
    assert row['hits'] == 19
    assert row['tier'] is None


def issue_entry(numbers, made='2026-10-03T08:00:00+07:00'):
    values = np.ones((1, 45)); values[0, np.array(numbers) - 1] = 2
    return {'event': 'issue', 'product': 'mega645', 'target_id': 2, 'history_sha256': 'abc',
            'config': {'search_nodes': 1000}, 'pending': {
                'target_id': 2, 'target_date': '2026-10-04', 'target_time': '2026-10-04T18:00:00+07:00',
                'made_at': made, 'based_on_id': 1, 'based_on_date': '2026-10-02', 'draws_on_last_date': 1,
                'history_sha256': 'abc', 'laws': {'main': {'values': values.tolist(), 'mixture': [1]}}}}


def test_snapshot_uses_first_pre_draw_issue_never_current_or_late_picks(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed, data, forecasts = tmp_path/'seed', tmp_path/'data', tmp_path/'forecast'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 7]},
                                     {'id': 2, 'date': '2026-10-04', 'result': [1, 2, 3, 4, 5, 8]}])
    write_rows(forecasts/'ml-ledger.jsonl', [issue_entry([8, 9, 10, 11, 12, 13], '2026-10-04T18:01:00+07:00'),
                                          issue_entry([1, 2, 3, 4, 5, 6]),
                                          issue_entry([8, 9, 10, 11, 12, 13], '2026-10-03T09:00:00+07:00')])
    snap = build_dashboard(data, seed, forecasts, now=datetime(2026, 10, 4, 19, tzinfo=VN))
    mega = next(p for p in snap['products'] if p['product'] == 'mega645')
    assert len(mega['comparisons']) == 1
    assert mega['comparisons'][0]['tickets'][0]['numbers'] == [1, 2, 3, 4, 5, 6]
    assert mega['comparisons'][0]['tickets'][0]['hits'] == 5
    assert mega['next_forecast'] is None
    assert len(snap['products']) == 7


def test_snapshot_exclusions_and_finance_do_not_carry_forward(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed, data, forecasts = tmp_path/'seed', tmp_path/'data', tmp_path/'forecast'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 7]},
                                     {'id': 2, 'date': '2026-10-04', 'result': [1, 2, 3, 4, 5, 8]}])
    write_rows(seed/'prizes_mega645.jsonl', [{'game': 'mega645', 'draw_id': 1, 'draw_date': '2026-10-02',
          'winners': {'jackpot1': 0, 'first': 3}, 'jackpot_pots': {'jackpot1': 123456789}, 'source': 'vietlott.vn'}])
    seed.joinpath('exclusions.json').write_text(json.dumps([{'product': 'mega645', 'draw_id': 2}]))
    snap = build_dashboard(data, seed, forecasts, now=datetime(2026, 10, 4, 19, tzinfo=VN))
    mega = next(p for p in snap['products'] if p['product'] == 'mega645')
    assert mega['latest']['draw_id'] == 1
    assert mega['latest']['prizes'][0]['winners'] == 0
    seed.joinpath('exclusions.json').write_text('[]')
    snap = build_dashboard(data, seed, forecasts, now=datetime(2026, 10, 4, 19, tzinfo=VN))
    latest = next(p for p in snap['products'] if p['product'] == 'mega645')['latest']
    assert latest['prizes'][0]['value_vnd'] is None
    assert latest['prizes'][0]['winners'] is None
    assert latest['prizes'][1]['value_vnd'] == 10000000


def test_journal_updates_results_and_source_without_mutating_inputs(tmp_path):
    from vlm.web.dashboard import build_dashboard
    record = DrawRecord.from_legacy('bingo18', {'id': 8, 'date': '2026-10-04', 'result': [6, 6, 6], 'source': 'nhanaz'})
    journal = tmp_path/'results.jsonl'; journal.write_text(record.model_dump_json()+'\n')
    snap = build_dashboard(tmp_path/'data', tmp_path/'seed', tmp_path/'forecast', journal, datetime(2026, 10, 4, 19, tzinfo=VN))
    bingo = next(p for p in snap['products'] if p['product'] == 'bingo18')['latest']
    assert bingo['numbers'] == [6, 6, 6]
    assert bingo['facts']['sum'] == 18
    assert bingo['facts']['multiplicity'] == {'6': 3}
    assert bingo['source'] == 'nhanaz'
    assert bingo['time_precision'] == 'day'
    assert bingo['official_direct'] is False
    assert journal.read_text() == record.model_dump_json()+'\n'


def test_issue_cannot_claim_pre_draw_when_its_input_result_had_not_occurred(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed = tmp_path/'seed'; directory = tmp_path/'forecasts'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 6]}])
    write_rows(directory/'ml-ledger.jsonl', [issue_entry([1, 2, 3, 4, 5, 6], '2026-10-02T08:00:00+07:00')])
    snap = build_dashboard(tmp_path/'data', seed, directory, now=datetime(2026, 10, 3, 19, tzinfo=VN))
    assert next(p for p in snap['products'] if p['product'] == 'mega645')['next_forecast'] is None


def test_site_builder_creates_home_and_preserves_forecast_outputs(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts'/'build_site.py'
    spec = importlib.util.spec_from_file_location('vlm_site_builder', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    seed, directory, out = tmp_path/'seed', tmp_path/'forecasts', tmp_path/'site'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 7]}])
    meta = module.build(out, directory, data_dir=tmp_path/'data', seed_dir=seed)
    assert (out/'index.html').exists() and (out/'forecast.html').exists()
    snap = json.loads((out/'data'/'dashboard.json').read_text())
    assert snap['products'][0]['latest']['numbers'] == [1, 2, 3, 4, 5, 7]
    assert meta['dashboard']['results'] == 1
    assert json.loads((out/'data'/'scoreboard.json').read_text()) == []
    assert (out/'data'/'summary.json').exists()


def test_homepage_embedded_json_cannot_be_terminated_by_source_text(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts'/'build_site.py'
    spec = importlib.util.spec_from_file_location('safe_site_builder', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    seed, out = tmp_path/'seed', tmp_path/'site'
    malicious = '</script><script>alert(1)</script>'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 7], 'source': malicious}])
    module.build(out, tmp_path/'forecasts', data_dir=tmp_path/'data', seed_dir=seed)
    page = (out/'index.html').read_text()
    assert malicious not in page
    embedded = page.split('id="initial-data">')[1].split('</script>')[0]
    assert json.loads(embedded)['products'][0]['latest']['source'] == malicious


def test_finance_is_not_attached_when_same_id_has_a_different_date(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed=tmp_path/'seed'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 7]}])
    write_rows(seed/'prizes_mega645.jsonl', [{'game': 'mega645', 'draw_id': 1, 'draw_date': '2026-09-30',
                'winners': {'jackpot1': 0}, 'jackpot_pots': {'jackpot1': 123456789}, 'source': 'vietlott.vn'}])
    snap=build_dashboard(tmp_path/'data',seed,tmp_path/'forecasts')
    prize=snap['products'][0]['latest']['prizes'][0]
    assert prize['value_vnd'] is None and prize['winners'] is None


def test_missing_target_stays_pending_when_a_later_draw_arrives(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed, directory = tmp_path/'seed', tmp_path/'forecasts'
    write_rows(seed/'power645.jsonl', [{'id': 1, 'date': '2026-10-02', 'result': [1, 2, 3, 4, 5, 6]},
                                      {'id': 3, 'date': '2026-10-07', 'result': [8, 9, 10, 11, 12, 13]}])
    write_rows(directory/'ml-ledger.jsonl',[issue_entry([1, 2, 3, 4, 5, 6])])
    snap=build_dashboard(tmp_path/'data',seed,directory,now=datetime(2026,10,8,19,tzinfo=VN))
    comparisons=snap['products'][0]['comparisons']
    assert len(comparisons)==1
    assert (comparisons[0]['target_id'],comparisons[0]['status'])==(2,'pending')


def test_older_mirror_journal_preserves_same_draw_official_finance(tmp_path):
    from vietlott_engine.core.models import Draw
    from vietlott_engine.crawler.storage import DuckDBRepository
    from vlm.web.dashboard import build_dashboard
    data=tmp_path/'data'; repo=DuckDBRepository(data/'vietlott.duckdb')
    repo.upsert([Draw(game='mega645',draw_id=1,draw_date='2026-10-02',numbers=(1,2,3,4,5,6),
                      jackpot1_value=123456789,source='vietlott.vn')]); repo.close()
    row=DrawRecord.from_legacy('mega645',{'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6],'source':'mirror'})
    journal=data/'results'/'results.jsonl';journal.parent.mkdir(parents=True);journal.write_text(row.model_dump_json()+'\n')
    result=build_dashboard(data,tmp_path/'seed',tmp_path/'forecasts')['products'][0]['latest']
    assert result['prizes'][0]['value_vnd']==123456789
    assert result['source']=='vietlott.vn'


def test_partial_prize_record_keeps_known_same_draw_jackpot(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed=tmp_path/'seed'
    write_rows(seed/'power645.jsonl',[{'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6],
                                    'jackpot1_value':123456789,'source':'vietlott.vn'}])
    write_rows(seed/'prizes_mega645.jsonl',[{'game':'mega645','draw_id':1,'draw_date':'2026-10-02',
                  'winners':{'jackpot1':0,'first':3},'jackpot_pots':None,'source':'vietlott.vn'}])
    result=build_dashboard(tmp_path/'data',seed,tmp_path/'forecasts')['products'][0]['latest']
    assert result['prizes'][0]['value_vnd']==123456789
    assert result['prizes'][0]['winners']==0


def test_conflicting_same_draw_observations_are_reported(tmp_path):
    from vlm.web.dashboard import build_dashboard
    seed,data=tmp_path/'seed',tmp_path/'data'
    write_rows(seed/'power645.jsonl',[{'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6],'source':'vietlott.vn'}])
    row=DrawRecord.from_legacy('mega645',{'id':1,'date':'2026-10-02','result':[7,8,9,10,11,12],'source':'mirror'})
    journal=data/'results'/'results.jsonl';journal.parent.mkdir(parents=True);journal.write_text(row.model_dump_json()+'\n')
    snap=build_dashboard(data,seed,tmp_path/'forecasts')
    assert snap['products'][0]['latest']['numbers']==[1,2,3,4,5,6]
    assert any('conflict' in message for message in snap['warnings'])


def test_lotto_catalogue_includes_zero_to_two_main_matches():
    from vietlott_engine.core.products import ProductCode
    from vlm.web.dashboard import prize_catalogue
    consolation=next(t for t in prize_catalogue(ProductCode.LOTTO_535) if t['code']=='consolation')
    assert consolation['condition'].startswith('0–2 số chính')
