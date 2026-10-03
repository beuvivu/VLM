"""Browser TLS transport, bounded retry, optional solver and pinned clearance.

Proxies are supplied by the operator. Clearance is scoped to origin, proxy,
browser and solver User-Agent. No challenge response is accepted as draw data.
"""
from __future__ import annotations
import asyncio
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlparse
import httpx
from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.crawler.http import AsyncRateLimiter


class ScrapeError(SourceError):
    pass


@dataclass(frozen=True)
class ScraperConfig:
    proxies: tuple[str, ...] = ()
    browsers: tuple[str, ...] = ("chrome", "firefox")
    max_retries: int = 5
    timeout_s: float = 20
    request_budget_s: float = 120
    rate_per_s: float = 1
    max_concurrency: int = 4
    clearance_ttl_s: float = 300
    max_response_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.browsers or any(b not in ("chrome","firefox") for b in self.browsers):
            raise ValueError("use supported Chrome/Firefox profiles")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be 0..10")
        if any(x <= 0 for x in (self.timeout_s,self.request_budget_s,self.rate_per_s,
                                self.max_concurrency,self.clearance_ttl_s,self.max_response_bytes)):
            raise ValueError("transport limits must be positive")
        for proxy in self.proxies:
            p = urlparse(proxy)
            if p.scheme not in ("http","https","socks5","socks5h") or not p.hostname:
                raise ValueError("proxy must be an HTTP(S)/SOCKS5 URL")

    @classmethod
    def from_env(cls) -> ScraperConfig:
        # Credentials remain in memory; they never enter error messages or reports.
        return cls(proxies=tuple(p.strip() for p in os.getenv("VLM_PROXIES","").split(",") if p.strip()))


@dataclass(frozen=True)
class SolvedSession:
    response: httpx.Response
    user_agent: str
    cookies: dict[str,str]


class ChallengeSolver(Protocol):
    async def solve(self, url: str, proxy: str | None) -> SolvedSession: ...


def is_challenge(response: httpx.Response) -> bool:
    if response.headers.get("cf-mitigated","").lower() == "challenge":
        return True
    head = response.text[:16384].lower()
    return any(m in head for m in ("cf-chl-", "cf_chl_opt", "/cdn-cgi/challenge-platform/",
                                  "<title>just a moment", "attention required! | cloudflare"))


class FlareSolverrSolver:
    """Optional external service. It may fail on unsupported challenge types."""
    def __init__(self, endpoint: str, timeout_s: float = 60) -> None:
        self.endpoint, self.timeout_s = endpoint, timeout_s

    async def solve(self, url: str, proxy: str | None) -> SolvedSession:
        payload: dict[str,Any] = {"cmd":"request.get", "url":url, "maxTimeout":int(self.timeout_s*1000)}
        if proxy:
            p=urlparse(proxy)
            proxy_data = {"url":f"{p.scheme}://{p.hostname}:{p.port or 80}"}
            if p.username:
                from urllib.parse import unquote
                proxy_data.update(username=unquote(p.username),password=unquote(p.password or ""))
            payload["proxy"]=proxy_data
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s+5,trust_env=False) as client:
                res = await client.post(self.endpoint,json=payload)
                res.raise_for_status()
                data=res.json()
            if data.get("status") != "ok":
                raise ScrapeError("solver did not produce a clearance session")
            sol=data["solution"]
            if not sol.get("userAgent"):
                raise ScrapeError("solver session is missing its User-Agent")
            host = urlparse(url).hostname or ""
            cookies={}
            for c in sol.get("cookies",[]):
                domain=c.get("domain","").lstrip(".")
                if domain and (host==domain or host.endswith("."+domain)):
                    cookies[c["name"]]=c["value"]
            response=httpx.Response(int(sol.get("status",200)),text=sol.get("response",""),
                                    headers=sol.get("headers",{}),request=httpx.Request("GET",url))
            if response.status_code>=400 or is_challenge(response):
                raise ScrapeError("solver returned a blocked response")
            return SolvedSession(response,sol["userAgent"],cookies)
        except (httpx.HTTPError,KeyError,TypeError,ValueError) as exc:
            raise ScrapeError(f"solver failed ({type(exc).__name__})") from None


class AsyncScraper:
    """Compatible get/post/set_cookie interface for existing Vietlott sources."""
    def __init__(self, config: ScraperConfig | None = None, *,
                 requester: Callable[...,Awaitable[httpx.Response]] | None = None,
                 solver: ChallengeSolver | None = None,
                 sleep: Callable[[float],Awaitable[None]] = asyncio.sleep) -> None:
        self.config=config or ScraperConfig()
        self.solver, self._requester, self._sleep=solver,requester,sleep
        self._limiter=AsyncRateLimiter(self.config.rate_per_s,burst=1)
        self._sem=asyncio.Semaphore(self.config.max_concurrency)
        self._counter=0
        self._sessions: dict[tuple[str | None,str],Any]={}
        self._sticky: dict[str,tuple[str | None,str,SolvedSession,float]]={}
        self._cookies: dict[str,dict[str,str]]={}
        self._last_origin=""
        self.stats={"requests":0,"retries":0,"failures":0,"solved":0}

    async def __aenter__(self) -> AsyncScraper:
        return self

    async def __aexit__(self,*exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()

    def set_cookie(self,name: str,value: str,domain: str = "") -> None:
        origin = domain.lstrip(".") if domain else self._last_origin
        if not origin:
            raise ValueError("cookie requires an origin")
        self._cookies.setdefault(origin,{})[name]=value

    async def _send(self,method: str,url: str,*,proxy: str | None,impersonate: str,**kwargs: Any) -> httpx.Response:
        if self._requester:
            return await self._requester(method,url,proxy=proxy,impersonate=impersonate,**kwargs)
        try:
            from curl_cffi.requests import AsyncSession
            from curl_cffi.const import CurlHttpVersion
        except ImportError:
            raise ScrapeError("install vlm browser transport with: pip install '.[crawler]'") from None
        key=(proxy,impersonate)
        if key not in self._sessions:
            self._sessions[key]=AsyncSession(max_clients=self.config.max_concurrency)
        res=await self._sessions[key].request(method,url,proxy=proxy,impersonate=impersonate,
                                             http_version=CurlHttpVersion.V2_0,**kwargs)
        if len(res.content)>self.config.max_response_bytes:
            raise ScrapeError("response exceeds configured size limit")
        # libcurl has already decoded gzip/br and chunked transfer encoding.
        headers={k:v for k,v in res.headers.items() if k.lower() not in ("content-encoding","content-length","transfer-encoding")}
        return httpx.Response(res.status_code,content=res.content,headers=headers,
                              request=httpx.Request(method,url))

    async def request(self,method: str,url: str,**kwargs: Any) -> httpx.Response:
        parsed=urlparse(url)
        if parsed.scheme not in ("http","https") or not parsed.hostname:
            raise ValueError("request URL must use http(s)")
        origin=f"{parsed.scheme}://{parsed.netloc}"
        self._last_origin=origin
        start=time.monotonic()
        solved_once=False
        last="unavailable"
        for attempt in range(self.config.max_retries+1):
            remaining=self.config.request_budget_s-(time.monotonic()-start)
            if remaining<=0:
                last="request time budget exhausted"
                break
            sticky=self._sticky.get(origin)
            if sticky and sticky[3] > time.monotonic():
                proxy,browser,clearance,_=sticky
            else:
                index=self._counter
                self._counter+=1
                proxy=self.config.proxies[index%len(self.config.proxies)] if self.config.proxies else None
                browser=self.config.browsers[index%len(self.config.browsers)]
                clearance=None
            options=dict(kwargs)
            headers=dict(options.pop("headers",{}))
            cookies=dict(self._cookies.get(origin,{}))
            cookies.update(self._cookies.get(parsed.hostname,{}))
            cookies.update(options.pop("cookies",{}))
            if clearance:
                headers["User-Agent"]=clearance.user_agent
                cookies.update(clearance.cookies)
            options.update(headers=headers,cookies=cookies,timeout=min(self.config.timeout_s,remaining))
            delay=2**attempt
            try:
                async with asyncio.timeout(remaining):
                    await self._limiter.acquire()
                    async with self._sem:
                        self.stats["requests"]+=1
                        response=await self._send(method,url,proxy=proxy,impersonate=browser,**options)
                if len(response.content)>self.config.max_response_bytes:
                    raise ScrapeError("response exceeds configured size limit")
                if response.headers.get("x-mitmproxy-blocked-reason") or response.headers.get("x-squid-error"):
                    raise ScrapeError("network egress policy blocked the request")
                challenge=is_challenge(response)
                if not challenge and response.status_code<400:
                    return response
                if challenge and self.solver and not solved_once:
                    solved_once=True
                    solver_url=url if method.upper()=="GET" else origin+"/"
                    try:
                        async with asyncio.timeout(max(0.01,self.config.request_budget_s-(time.monotonic()-start))):
                            solved=await self.solver.solve(solver_url,proxy)
                        if is_challenge(solved.response) or solved.response.status_code>=400:
                            raise ScrapeError("solver returned a challenge")
                        ua=solved.user_agent.lower()
                        # FlareSolverr uses Chrome even if the blocked attempt was
                        # Firefox; pair its clearance UA with the same browser family.
                        solved_browser="firefox" if "firefox/" in ua else "chrome" if "chrome/" in ua else browser
                        self._sticky[origin]=(proxy,solved_browser,solved,time.monotonic()+self.config.clearance_ttl_s)
                        self.stats["solved"]+=1
                        if method.upper()=="GET":
                            return solved.response
                        delay=0
                    except (SourceError,TimeoutError):
                        last="challenge solver unavailable"
                else:
                    last="challenge response" if challenge else f"HTTP {response.status_code}"
                if not challenge and response.status_code not in (403,408,425,429,500,502,503,504):
                    raise ScrapeError(f"non-retryable HTTP {response.status_code}")
                retry_after=response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay=max(delay,float(retry_after))
                    except ValueError:
                        try:
                            delay=max(delay,parsedate_to_datetime(retry_after).timestamp()-time.time())
                        except (ValueError,TypeError):
                            pass
            except ScrapeError:
                self.stats["failures"]+=1
                raise
            except (httpx.TransportError,TimeoutError) as exc:
                last=type(exc).__name__
            except Exception as exc:
                # curl exceptions may contain proxy credentials; keep only their class.
                if exc.__class__.__module__.startswith("curl_cffi"):
                    last=type(exc).__name__
                else:
                    raise
            if attempt==self.config.max_retries:
                break
            if delay>=self.config.request_budget_s-(time.monotonic()-start):
                last="Retry-After/backoff exceeds request budget"
                break
            self.stats["retries"]+=1
            await self._sleep(delay)
        self.stats["failures"]+=1
        raise ScrapeError(f"request exhausted: {last}")

    async def get(self,url: str,**kwargs: Any) -> httpx.Response:
        return await self.request("GET",url,**kwargs)

    async def post(self,url: str,**kwargs: Any) -> httpx.Response:
        return await self.request("POST",url,**kwargs)
