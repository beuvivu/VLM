import asyncio
import httpx
import pytest
from vlm.crawler.sources import OfficialSource, CanonicalGithubSource, TargetedFetcher, MonthlyArchiveSource
from vietlott_engine.core.exceptions import SourceError
from vlm.rules.calculator import settle_matrix


class Client:
    def __init__(self,text): self.text=text
    async def get(self,url): return httpx.Response(200,text=self.text,request=httpx.Request('GET',url))
    async def post(self,url,**kwargs):
        return httpx.Response(200,json={'value':{'HtmlContent':self.text}},request=httpx.Request('POST',url))


def detail(numbers,prizes):
    balls=''.join(f'<span>{n}</span>' for n in numbers)
    rows=''.join(f'<tr><td>{name}</td><td>{winners}</td><td>{value}</td></tr>' for name,winners,value in prizes)
    return f'<div>Kỳ quay #00001 ngày 02/10/2026</div><div class="day_so_ket_qua_v2">{balls}</div><table><tr><th>Giải thưởng</th><th>Số lượng giải</th><th>Giá trị giải</th></tr>{rows}</table>'


def test_official_mega_per_winner_amount_becomes_pool_then_settles_once():
    html=detail([1,2,3,4,5,6],[('Jackpot',2,79409173250),('Giải Nhất',45,10000000)])
    r=asyncio.run(OfficialSource(Client(html)).fetch('mega645',1))
    assert r.jackpot1_value==158818346500
    assert r.jackpot1_winners==2
    assert r.sub_prizes_json['winners']['first']==45
    result=settle_matrix('mega645',[1,2,3,4,5,6],list(r.winning_numbers),
                         jackpot_pots={'jackpot1':r.jackpot1_value},
                         total_jackpot_winners={'jackpot1':r.jackpot1_winners})
    assert result.gross_payout==79409173250


@pytest.mark.parametrize('game,numbers,prizes,expected',[
    ('power655',[1,2,3,4,5,6,7],[('Jackpot 1',0,300000000000),('Jackpot 2',3,2000000000)],(300000000000,6000000000,0,3)),
    ('lotto535',[1,2,3,4,5,7],[('Giải Độc Đắc',1,7000000000)],(7000000000,None,1,None)),
])
def test_official_finance_slugs_and_canonical_prizes(game,numbers,prizes,expected):
    r=asyncio.run(OfficialSource(Client(detail(numbers,prizes))).fetch(game,1))
    assert (r.jackpot1_value,r.jackpot2_value,r.jackpot1_winners,r.jackpot2_winners)==expected


def test_wrong_game_canonical_cannot_be_relabelled():
    import json
    row={'game':'power655','draw_id':1,'draw_date':'2026-10-02','result':{'main_numbers':[1,2,3,4,5,6],'bonus_numbers':[]}}
    source=CanonicalGithubSource(Client(json.dumps(row)))
    with pytest.raises(SourceError):
        asyncio.run(TargetedFetcher([source]).fetch('mega645',1))


def test_monthly_archive_rejects_unconfirmed():
    import csv
    import io
    import json
    output=io.StringIO()
    writer=csv.DictWriter(output,fieldnames=['draw_id','draw_date','result_json','draw_status'])
    writer.writeheader()
    writer.writerow({'draw_id':1,'draw_date':'2026-10-02','result_json':json.dumps({'numbers':list(range(1,21))}),'draw_status':'not_confirmed'})
    assert asyncio.run(MonthlyArchiveSource(Client(output.getvalue())).fetch('keno',1,'2026-10-02')) is None


def test_monthly_archive_cannot_discard_explicit_wrong_product():
    import csv
    import io
    import json
    output=io.StringIO()
    writer=csv.DictWriter(output,fieldnames=['product','draw_id','draw_date','result_json'])
    writer.writeheader()
    writer.writerow({'product':'power655','draw_id':1,'draw_date':'2026-10-02',
                     'result_json':json.dumps({'numbers':[1,2,3,4,5,6]})})
    with pytest.raises(SourceError):
        asyncio.run(TargetedFetcher([MonthlyArchiveSource(Client(output.getvalue()))]).fetch('mega645',1))


def test_malformed_unrelated_official_row_does_not_discard_valid_target():
    def table_row(did,nums):
        balls=''.join(f'<span>{n}</span>' for n in nums)
        return f'<tr><td><a>02/10/2026</a><a>#{did:05d}</a></td><td><div class="day_so_ket_qua_v2">{balls}</div></td></tr>'
    html='<table>'+table_row(1,range(1,7))+table_row(2,[])+'</table>'
    r=asyncio.run(OfficialSource(Client(html)).fetch('mega645',1))
    assert r.draw_id==1
    assert r.winning_numbers==(1,2,3,4,5,6)


def test_timed_out_source_cannot_starve_fallback():
    from vlm.database.schema import DrawRecord
    async def run():
        cancelled=[]
        class Blocked:
            name='blocked'
            async def fetch(self,*args):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(True)
        class Mirror:
            name='mirror'
            async def fetch(self,game,draw_id,date_hint=None):
                return DrawRecord.from_legacy(game,{'id':draw_id,'date':'2026-10-02','result':[1,2,3,4,5,6]})
        fetcher=TargetedFetcher([Blocked(),Mirror()],source_timeout_s=0.02)
        r=await fetcher.fetch('mega645',1)
        assert r.draw_id==1
        assert cancelled==[True]
        assert fetcher.attempts[0]['error']=='TimeoutError'
        assert fetcher.attempts[-1]['status']=='ok'
    asyncio.run(run())
