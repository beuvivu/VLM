"""Resilient async HTTP client: token-bucket rate limiting, bounded concurrency,
retry with exponential backoff + full jitter, and ``Retry-After`` support.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

import httpx

from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.logging import get_logger

log = get_logger(__name__)


class AsyncRateLimiter:
    """Token bucket: at most ``rate`` acquisitions per second, bursts up to ``burst``."""

    def __init__(self, rate: float, burst: int = 1, clock: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = max(1, burst)
        self._tokens = float(self.capacity)
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1 - 1e-9:  # tolerance: float round-off must not cause a spin
                    self._tokens = max(0.0, self._tokens - 1)
                    return
                await asyncio.sleep(max((1 - self._tokens) / self.rate, 1e-3))


PROXY_BLOCK_PREFIX = "blocked by this network's HTTPS proxy (egress policy) — not retried"
CF_CHALLENGE_PREFIX = "answered with a Cloudflare 'verify you are human' challenge — not retried, not bypassed"
_CF_MARKERS = ("cf-chl", "challenge-platform", "just a moment", "cf_chl_opt", "turnstile", "attention required! | cloudflare")


def is_cloudflare_challenge(response: httpx.Response) -> bool:
    """True when the response is a Cloudflare interstitial (managed / JS challenge, Turnstile).

    The engine never tries to solve or evade such a challenge: it is the site's way of saying
    automated clients are not welcome right now. A person can pass it in a browser and save
    the pages (``products import-pages``)."""
    if response.status_code not in (403, 429, 503):
        return False
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    if "cloudflare" not in response.headers.get("server", "").lower():
        return False
    head = response.text[:4000].lower()
    return any(m in head for m in _CF_MARKERS)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter (AWS architecture blog formulation)."""

    max_retries: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    retry_statuses: frozenset[int] = field(default_factory=lambda: frozenset({408, 425, 429, 500, 502, 503, 504}))

    def backoff(self, attempt: int, rng: random.Random | None = None) -> float:
        cap = min(self.max_delay, self.base_delay * (2**attempt))
        return (rng or random).uniform(0, cap)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class AsyncHttpClient:
    """Thin wrapper around ``httpx.AsyncClient`` used by every data source.

    Use as an async context manager. ``transport`` can be injected for tests
    (``httpx.MockTransport``).
    """

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        rate_limit_per_s: float = 2.0,
        burst: int = 2,
        max_concurrency: int = 4,
        retry: RetryPolicy | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.retry = retry or RetryPolicy()
        self._limiter = AsyncRateLimiter(rate_limit_per_s, burst)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
            transport=transport,
            follow_redirects=True,
        )
        self.stats = {"requests": 0, "retries": 0, "failures": 0}

    async def __aenter__(self) -> "AsyncHttpClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request, retrying transient failures. Raises ``SourceError`` when exhausted."""
        last_error: str = ""
        for attempt in range(self.retry.max_retries + 1):
            await self._limiter.acquire()
            async with self._semaphore:
                self.stats["requests"] += 1
                try:
                    response = await self._client.request(method, url, **kwargs)
                except httpx.ProxyError as exc:
                    if any(code in str(exc) for code in ("403", "407")):  # egress policy, not a transient fault
                        self.stats["failures"] += 1
                        raise SourceError(f"{method} {url}: {PROXY_BLOCK_PREFIX} ({exc})") from exc
                    last_error = f"{type(exc).__name__}: {exc}"
                    delay = self.retry.backoff(attempt)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    delay = self.retry.backoff(attempt)
                else:
                    if response.status_code < 400:
                        return response
                    if is_cloudflare_challenge(response):
                        self.stats["failures"] += 1
                        raise SourceError(f"{method} {url}: {CF_CHALLENGE_PREFIX} (HTTP {response.status_code})")
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code not in self.retry.retry_statuses:
                        self.stats["failures"] += 1
                        raise SourceError(f"{method} {url} failed with non-retryable {last_error}")
                    retry_after = _retry_after_seconds(response)
                    delay = (
                        min(self.retry.max_delay, retry_after)
                        if retry_after is not None
                        else self.retry.backoff(attempt)
                    )
            if attempt < self.retry.max_retries:
                self.stats["retries"] += 1
                log.warning(
                    "retrying %s %s after %s (attempt %d/%d, sleep %.2fs)",
                    method, url, last_error, attempt + 1, self.retry.max_retries, delay,
                )
                await self._sleep(delay)
        self.stats["failures"] += 1
        raise SourceError(f"{method} {url} failed after {self.retry.max_retries} retries: {last_error}")

    def set_cookie(self, name: str, value: str, domain: str = "") -> None:
        """Add a cookie to the jar (e.g. one a site sets through ``document.cookie``)."""
        self._client.cookies.set(name, value, domain=domain)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)
