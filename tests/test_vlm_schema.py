from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

from vlm.database.schema import DrawRecord, SQLRepository, DuckRepository, Base, DataConflict


def mega(**updates):
    return DrawRecord(game_type='mega645', draw_id=1, draw_date='2026-10-02 18:00:00',
                      winning_numbers=[1, 2, 3, 4, 5, 6], **updates)


@pytest.mark.parametrize('numbers', [[0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 46],
                                   [1, 1, 3, 4, 5, 6], [1, 2, 3], [True, 2, 3, 4, 5, 6], [1.5, 2, 3, 4, 5, 6]])
def test_invalid_mega_numbers_rejected(numbers):
    with pytest.raises(ValidationError):
        DrawRecord(game_type='mega645', draw_id=1, draw_date='2026-10-02 18:00:00', winning_numbers=numbers)


def test_bonus_and_vietnam_time():
    with pytest.raises(ValidationError):
        DrawRecord(game_type='power655', draw_id=1, draw_date='2026-10-02 18:00:00', winning_numbers=[1,2,3,4,5,6], bonus_number=6)
    r = mega()
    assert r.draw_date.utcoffset() == timedelta(hours=7)
    with pytest.raises(ValidationError):
        mega(bonus_number=7)


def test_bingo_repetition_and_zero_padded_max_are_preserved():
    assert DrawRecord.from_legacy('bingo18', {'id':'00001','date':'2026-10-02','result':[6,6,6]}).winning_numbers == (6,6,6)
    r = DrawRecord.from_legacy('max3d', {'id':1,'date':'2026-10-02','result':['000']*20})
    assert r.winning_numbers == ('000',)*20
    assert r.time_precision == 'day'


@pytest.mark.parametrize('factory', [lambda p: SQLRepository('sqlite:///'+str(p)), lambda p: DuckRepository(p)])
def test_repository_atomic_conflict_and_metadata_merge(factory, tmp_path):
    repo = factory(tmp_path/'draws.db')
    r = mega(jackpot1_value=12000000000, jackpot1_winners=0)
    assert repo.upsert([r]) == 1
    assert repo.upsert([mega()]) == 0
    assert repo.get('mega645',1).jackpot1_value == 12000000000
    assert repo.get('mega645',1).jackpot1_winners == 0
    bad = r.model_copy(update={'winning_numbers':(1,2,3,4,5,7)})
    with pytest.raises(DataConflict):
        repo.upsert([r.model_copy(update={'draw_id':2}), bad])
    assert repo.get('mega645',2) is None
    assert repo.get('mega645',1).winning_numbers == (1,2,3,4,5,6)
    out = tmp_path/'roundtrip.parquet'
    repo.export_parquet(out)
    other = SQLRepository('sqlite:///'+str(tmp_path/'other.db'))
    assert other.import_parquet(out) == 1
    assert other.get('mega645',1) == repo.get('mega645',1)
    repo.close()
    other.close()


def test_sql_schema_compiles_for_postgresql():
    for name in ['draws_mega645','draws_power655','draws_max3d','draws_keno','draws_bingo18']:
        sql = str(CreateTable(Base.metadata.tables[name]).compile(dialect=postgresql.dialect()))
        assert 'PRIMARY KEY (game_type, draw_id)' in sql
        assert 'winning_numbers JSON' in sql
        assert 'jackpot1_winners' in sql


@pytest.mark.parametrize('value',['02/10/2026 18:00','2026-10-02', '2026-02-30 18:00:00'])
def test_datetime_requires_explicit_precision(value):
    with pytest.raises(ValidationError):
        DrawRecord(game_type='mega645',draw_id=1,draw_date=value,winning_numbers=[1,2,3,4,5,6])


@pytest.mark.parametrize('metadata', [{'game':'power655'}, {'product':'power655'}, {'game_type':'power655'}, {'draw_status':'not_confirmed'}])
def test_legacy_rejects_wrong_product_or_unconfirmed(metadata):
    with pytest.raises(ValueError):
        DrawRecord.from_legacy('mega645', {'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6]} | metadata)


@pytest.mark.parametrize('factory', [lambda p: SQLRepository('sqlite:///'+str(p)), lambda p: DuckRepository(p)])
def test_nested_finance_merge_preserves_observations_and_rejects_conflicts(factory,tmp_path):
    repo=factory(tmp_path/'metadata.db')
    original=mega(jackpot1_value=12000000000,source='vietlott.vn',sub_prizes_json={
        'winners':{'first':45,'second':2105},'prizes':[{'code':'first','winner_count':45,'amount_vnd':10000000}]})
    repo.upsert([original])
    repo.upsert([mega(sub_prizes_json={'winners':{'jackpot1':0},'prizes':[]})])
    stored=repo.get('mega645',1)
    assert stored.sub_prizes_json['winners']=={'first':45,'second':2105,'jackpot1':0}
    assert stored.sub_prizes_json['prizes']==original.sub_prizes_json['prizes']
    with pytest.raises(DataConflict):
        repo.upsert([mega(jackpot1_value=999,source='mirror')])
    assert repo.get('mega645',1).jackpot1_value==12000000000
    repo.close()


def test_partial_legacy_null_columns_are_filled_from_known_pool():
    row={'id':1,'date':'2026-10-02','result':[1,2,3,4,5,6],
         'jackpot1_value':None,'jackpot1_winners':None,
         'jackpot_pots':{'jackpot1':12000000000},'winners':{'jackpot1':0}}
    r=DrawRecord.from_legacy('mega645',row)
    assert (r.jackpot1_value,r.jackpot1_winners)==(12000000000,0)
    with pytest.raises(ValueError):
        DrawRecord.from_legacy('mega645',row|{'jackpot1_value':100})
