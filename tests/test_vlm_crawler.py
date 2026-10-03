import asyncio
import httpx
import pytest

from vlm.crawler.async_scraper import AsyncScraper, ScraperConfig, ScrapeError, SolvedSession


def response(status=200, text='{"ok":true}', headers=None):
    return httpx.Response(status,text=text,headers=headers or {},request=httpx.Request('GET','https://vietlott.vn/result'))


def test_retry_403_429_503_and_rotates_profiles():
    async def run():
        seen,sleeps=[],[]
        async def request(method,url,**kw):
            seen.append((kw['proxy'],kw['impersonate']))
            return response([403,429,503,403,503,200][len(seen)-1])
        async def sleep(seconds): sleeps.append(seconds)
        async with AsyncScraper(ScraperConfig(proxies=('http://p1','http://p2'),rate_per_s=10000),
                                requester=request,sleep=sleep) as client:
            assert (await client.get('https://vietlott.vn/result')).json()=={'ok':True}
        assert sleeps==[1,2,4,8,16]
        assert {p for p,b in seen}=={'http://p1','http://p2'}
        assert {b for p,b in seen}=={'chrome','firefox'}
    asyncio.run(run())


def test_200_challenge_is_not_accepted():
    async def run():
        async def request(*a,**kw): return response(200,'<html><title>Just a moment...</title><script src="/cdn-cgi/challenge-platform/"></script></html>')
        async def sleep(s): pass
        async with AsyncScraper(ScraperConfig(max_retries=0),requester=request,sleep=sleep) as c:
            with pytest.raises(ScrapeError): await c.get('https://vietlott.vn/result')
    asyncio.run(run())


def test_solver_session_pins_proxy_user_agent_and_cookies():
    async def run():
        seen=[]
        async def request(method,url,**kw):
            seen.append(kw)
            if len(seen)==1: return response(403,'<title>Just a moment...</title><div class="cf-chl">')
            return response()
        class Solver:
            async def solve(self,url,proxy):
                return SolvedSession(response(),user_agent='SolvedChrome',cookies={'cf_clearance':'token'})
        async with AsyncScraper(ScraperConfig(proxies=('http://p1','http://p2'),rate_per_s=10000),
                                requester=request,solver=Solver()) as c:
            await c.get('https://vietlott.vn/result')
            await c.post('https://vietlott.vn/ajax',json={'id':1})
        assert seen[-1]['proxy']=='http://p1'
        assert seen[-1]['headers']['User-Agent']=='SolvedChrome'
        assert seen[-1]['cookies']['cf_clearance']=='token'
    asyncio.run(run())


def test_chrome_solver_clearance_uses_chrome_profile_after_firefox_attempt():
    async def run():
        seen=[]
        async def request(method,url,**kw):
            seen.append(kw)
            return response(403,'<title>Just a moment...</title>') if len(seen)==1 else response()
        class Solver:
            async def solve(self,url,proxy):
                return SolvedSession(response(),'Mozilla/5.0 Chrome/136.0.0.0 Safari/537.36',{'cf_clearance':'token'})
        async with AsyncScraper(ScraperConfig(browsers=('firefox',),rate_per_s=10000),requester=request,solver=Solver()) as client:
            await client.get('https://vietlott.vn/result')
            await client.post('https://vietlott.vn/ajax',json={'id':1})
        assert seen[0]['impersonate']=='firefox'
        assert seen[-1]['impersonate']=='chrome'
        assert 'Chrome/136' in seen[-1]['headers']['User-Agent']
    asyncio.run(run())


def test_network_policy_block_and_404_are_not_retried():
    async def run():
        for status,headers in [(404,{}),(403,{'x-mitmproxy-blocked-reason':'ROBOTS_DENIED'})]:
            seen=[]
            async def request(*a,_seen=seen,_status=status,_headers=headers,**kw):
                _seen.append(1)
                return response(_status,headers=_headers)
            async with AsyncScraper(requester=request) as c:
                with pytest.raises(ScrapeError): await c.get('https://vietlott.vn/result')
            assert len(seen)==1
    asyncio.run(run())


def test_retry_after_beyond_budget_fails_without_early_retry():
    async def run():
        seen=[]
        async def request(*a,**kw): seen.append(1); return response(429,headers={'Retry-After':'600'})
        async with AsyncScraper(ScraperConfig(request_budget_s=30),requester=request) as c:
            with pytest.raises(ScrapeError): await c.get('https://vietlott.vn/result')
        assert len(seen)==1
    asyncio.run(run())


def test_curl_decompressed_response_is_not_decompressed_twice(monkeypatch):
    from types import SimpleNamespace
    AsyncSession = pytest.importorskip('curl_cffi.requests').AsyncSession
    async def request(self,*args,**kwargs):
        return SimpleNamespace(status_code=200,content=b'{"ok":true}',
                               headers={'Content-Encoding':'gzip','Content-Type':'application/json','Content-Length':'99'})
    monkeypatch.setattr(AsyncSession,'request',request)
    async def run():
        async with AsyncScraper() as client:
            assert (await client.get('https://vietlott.vn/result')).json()=={'ok':True}
    asyncio.run(run())
