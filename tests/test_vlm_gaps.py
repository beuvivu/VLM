import asyncio
import json
from vlm.database.schema import DrawRecord, SQLRepository
from vlm.tools.gap_analyzer import analyze_game, repair_missing, write_report
from vlm.crawler.sources import TargetedFetcher


def row(did):
    return {'id':did,'date':'2026-10-02','result':[1,2,3,4,5,6]}


def test_leading_internal_trailing_and_excluded_gaps():
    r=analyze_game('mega645',[row(3),row(5),row(5)],latest_id=8,excluded_ids={4})
    assert r.missing_ids==[1,2,6,7,8]
    assert r.duplicate_ids==[5]
    assert r.excluded_ids==[4]
    assert r.leading_count==2
    assert r.trailing_count==3
    assert r.boundary_verified


def test_empty_and_invalid_rows_cannot_look_complete():
    r=analyze_game('mega645',[],latest_id=3)
    assert r.missing_ids==[1,2,3]
    r=analyze_game('mega645',[row(1),row(2)|{'result':[99]}])
    assert r.missing_ids==[2]
    assert len(r.invalid_rows)==1
    assert not r.boundary_verified


def test_report_is_atomic_and_missing_list_is_readable(tmp_path):
    path=tmp_path/'missing_draws.json'
    r=analyze_game('mega645',[row(2)])
    write_report(path,[r])
    data=json.loads(path.read_text())
    assert data['games']['mega645']['missing_ids']==[1]
    assert not list(tmp_path.glob('*.tmp'))


def test_fallback_rejects_wrong_id_and_tries_next():
    async def run():
        class Wrong:
            name='wrong'
            async def fetch(self,game,draw_id,date_hint=None): return DrawRecord.from_legacy(game,row(100))
        class Good:
            name='good'
            async def fetch(self,game,draw_id,date_hint=None): return DrawRecord.from_legacy(game,row(draw_id))
        f=TargetedFetcher([Wrong(),Good()])
        assert (await f.fetch('mega645',2)).draw_id==2
        assert [a['status'] for a in f.attempts]==['rejected','ok']
    asyncio.run(run())


def test_repair_fetches_only_missing_and_survives_partial_failure(tmp_path):
    async def run():
        seen=[]
        class Source:
            async def fetch(self,game,draw_id,date_hint=None):
                seen.append(draw_id)
                if draw_id==3: raise ValueError('not available')
                return DrawRecord.from_legacy(game,row(draw_id))
        repo=SQLRepository('sqlite:///'+str(tmp_path/'repair.db'))
        report=analyze_game('mega645',[row(2)],latest_id=3)
        result=await repair_missing([report],Source(),repo,max_draws=2,output_dir=tmp_path)
        assert sorted(seen)==[1,3]
        assert result['repaired']==[{'game_type':'mega645','draw_id':1}]
        assert len(result['failed'])==1
        assert repo.get('mega645',1) is not None
        assert repo.get('mega645',2) is None
        assert json.loads((tmp_path/'repair_status.json').read_text())['failed'][0]['draw_id']==3
        repo.close()
    asyncio.run(run())


def test_repair_attempts_both_month_hints(tmp_path):
    async def run():
        calls=[]
        class Source:
            name='monthly'
            async def fetch(self,game,draw_id,date_hint=None):
                calls.append(date_hint)
                if date_hint!='2026-04-01': return None
                return DrawRecord.from_legacy('keno',{'id':draw_id,'date':date_hint,'result':list(range(1,21))})
        rows=[{'id':1,'date':'2026-03-31','result':list(range(1,21))},
              {'id':3,'date':'2026-04-01','result':list(range(1,21))}]
        report=analyze_game('keno',rows)
        repo=SQLRepository('sqlite:///'+str(tmp_path/'months.db'))
        result=await repair_missing([report],TargetedFetcher([Source()]),repo,output_dir=tmp_path)
        assert calls==['2026-03-31','2026-04-01']
        assert result['repaired']==[{'game_type':'keno','draw_id':2}]
        repo.close()
    asyncio.run(run())


def test_lotto_same_day_evening_draw_remains_due():
    from datetime import datetime
    r=DrawRecord(game_type='lotto535',draw_id=1,draw_date='2026-10-03 13:00:00',winning_numbers=[1,2,3,4,5],bonus_number=7)
    report=analyze_game('lotto535',[r],latest_id=1,as_of=datetime.fromisoformat('2026-10-03T22:00:00+07:00'))
    assert report.scheduled_updates_due==['2026-10-03T21:00:00+07:00']
    assert not report.complete_through_present


def test_cli_repair_checkpoint_changes_report_and_next_run(tmp_path,monkeypatch):
    from vlm.tools.gap_analyzer import main
    import vlm.crawler.sources as sources
    seed=tmp_path/'seed'
    seed.mkdir()
    (seed/'power645.jsonl').write_text('\n'.join(json.dumps(row(i)) for i in [1,3]))
    calls=[]
    class Source:
        name='test'
        async def fetch(self,game,draw_id,date_hint=None):
            calls.append(draw_id)
            return DrawRecord.from_legacy(game,row(draw_id))
    monkeypatch.setattr(sources,'default_sources',lambda client:[Source()])
    report=tmp_path/'missing.json'
    argv=['--seed-dir',str(seed),'--game','mega645','--repair','--max-repair','1',
          '--database','sqlite:///'+str(tmp_path/'repair.db'),'--repair-dir',str(tmp_path/'repair'),
          '--output',str(report)]
    assert main(argv)==0
    assert json.loads(report.read_text())['games']['mega645']['missing_ids']==[]
    assert main(argv)==0
    assert calls==[2]


def test_audit_joins_finance_seed_by_game_id_and_date(tmp_path):
    from vlm.tools.gap_analyzer import audit_rows
    seed=tmp_path/'power645.jsonl'
    seed.write_text(json.dumps(row(1)))
    finances={'game':'mega645','draw_id':1,'draw_date':'2026-10-02',
              'jackpot_pots':{'jackpot1':12345678900},'winners':{'jackpot1':0,'first':3}}
    (tmp_path/'prizes_mega645.jsonl').write_text(json.dumps(finances))
    repo=SQLRepository('sqlite:///:memory:')
    report=analyze_game('mega645',audit_rows('mega645',seed,repo))
    assert report.finance_coverage['jackpot1_value']==1
    assert report.finance_coverage['sub_prizes_json']==1
    repo.close()


def test_finance_join_cannot_silently_replace_known_pool(tmp_path):
    from vlm.tools.gap_analyzer import audit_rows
    seed=tmp_path/'power645.jsonl'
    seed.write_text(json.dumps(row(1)|{'jackpot_pots':{'jackpot1':100}}))
    (tmp_path/'prizes_mega645.jsonl').write_text(json.dumps({'game':'mega645','draw_id':1,
        'draw_date':'2026-10-02','jackpot_pots':{'jackpot1':200},'winners':{'jackpot1':0}}))
    repo=SQLRepository('sqlite:///:memory:')
    report=analyze_game('mega645',audit_rows('mega645',seed,repo))
    assert report.valid_unique==1
    assert len(report.invalid_rows)==1
    assert not report.complete_in_checked_range
    repo.close()
