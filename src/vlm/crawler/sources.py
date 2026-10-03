"""Targeted official/AjaxPro and pinned GitHub archive adapters.

Every fallback must yield exactly the requested game and draw id. A success
with empty/stale data or an unrelated draw continues to the next source.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import math
from collections import OrderedDict
from datetime import date
from typing import Any, Protocol
from urllib.parse import urlencode
from vietlott_engine.core.exceptions import SourceError
from vietlott_engine.core.games import get_game
from vietlott_engine.core.products import ProductCode
from vietlott_engine.crawler.sources.vietlott_official import (
    ENDPOINTS, parse_detail_page, parse_results_html, parse_keno_html,
    parse_bingo18_html, parse_three_digit_html,
)
from vietlott_engine.crawler.sources.nhanaz import convert_draws, read_csv
from vlm.database.schema import DrawRecord, game_code


class DrawSource(Protocol):
    name: str
    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord | None: ...


class TextCache:
    """Bounded shared cache: one fetch per URL while concurrent repairs wait."""
    def __init__(self, client: Any, capacity: int = 8) -> None:
        self.client, self.capacity = client, capacity
        self._cache: OrderedDict[str,str] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, url: str) -> str:
        async with self._lock:
            if url in self._cache:
                self._cache.move_to_end(url)
                return self._cache[url]
            text = (await self.client.get(url)).text
            self._cache[url] = text
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
            return text


class OfficialSource:
    name = "vietlott.vn"
    def __init__(self, client: Any, base_url: str = "https://www.vietlott.vn") -> None:
        self.client, self.base_url = client, base_url.rstrip("/")

    @staticmethod
    def _product(game: str) -> ProductCode:
        return ProductCode("max3d" if game_code(game)=="max3dplus" else game_code(game))

    async def _ajax(self, game: str, draw_id: int | None = None, date_hint: str | None = None) -> tuple[str,str]:
        product=self._product(game)
        if product not in ENDPOINTS:
            raise SourceError("this discontinued product has no active official endpoint")
        ep=ENDPOINTS[product]
        body=ep.body(0)
        if draw_id is not None:
            key="GameDrawNo" if product in (ProductCode.KENO,ProductCode.BINGO18) else "GameDrawId"
            body[key]=f"{draw_id:05d}"
        if date_hint and product in (ProductCode.KENO,ProductCode.BINGO18):
            body["DrawDate"]=date.fromisoformat(date_hint).strftime("%d/%m/%Y")
        url=self.base_url+ep.path
        response=await self.client.post(url,json=body,headers={
            "Content-Type":"text/plain; charset=utf-8", "X-AjaxPro-Method":"ServerSideDrawResult",
            "X-Requested-With":"XMLHttpRequest", "Origin":self.base_url, "Referer":self.base_url+ep.detail_path,
        })
        try:
            html=response.json()["value"]["HtmlContent"]
            if not isinstance(html,str):
                raise ValueError("HtmlContent is not text")
            return html,url
        except (KeyError,TypeError,ValueError) as exc:
            raise SourceError("unexpected official AjaxPro response") from exc

    def _parse(self, game: str, html: str, url: str) -> list[DrawRecord]:
        g=game_code(game)
        if g in ("mega645","power655","lotto535"):
            spec=get_game(g)
            rows=[]
            for r in parse_results_html(html):
                try:
                    d=r.to_draw(spec,source=self.name)
                    rows.append({"id":d.draw_id,"date":d.draw_date.isoformat(),"result":list(d.numbers),
                                 "bonus_number":d.bonus})
                except (ValueError,KeyError,TypeError):
                    continue
        else:
            parser= parse_keno_html if g=="keno" else parse_bingo18_html if g=="bingo18" else parse_three_digit_html
            rows=parser(html)
        sha=hashlib.sha256(html.encode()).hexdigest()
        out=[]
        for row in rows:
            try:
                out.append(DrawRecord.from_legacy(g,row | {"source":self.name,"source_url":url,"source_sha256":sha}))
            except (ValueError,KeyError,TypeError):
                continue
        return out

    async def latest(self, game: str) -> DrawRecord:
        html,url=await self._ajax(game)
        rows=self._parse(game,html,url)
        if not rows:
            raise SourceError("official latest endpoint returned no valid draws")
        return max(rows,key=lambda r:r.draw_id)

    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord | None:
        g=game_code(game)
        product=self._product(g)
        if product not in ENDPOINTS:
            return None
        detail_paths={"keno":"/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result",
                      "bingo18":"/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result"}
        path=detail_paths.get(g,ENDPOINTS[product].detail_path)
        url=self.base_url+path+"?"+urlencode({"id":f"{draw_id:05d}","nocatche":1})
        try:
            html=(await self.client.get(url)).text
            detail=parse_detail_page(html,url)
            # Do not fill a missing id from the request: stale/error pages must fail.
            if detail["draw_id"]==draw_id and detail["draw_date"] and g in ("mega645","power655","lotto535","keno","bingo18"):
                prizes=detail["prizes"]
                return DrawRecord.from_legacy(g,detail | {"result":detail["numbers"],"prizes":prizes,
                                                         "source":self.name})
            rows=self._parse(g,html,url)
            exact=next((r for r in rows if r.draw_id==draw_id),None)
            if exact:
                return exact
        except (SourceError,ValueError,KeyError,TypeError):
            pass
        html,url=await self._ajax(g,draw_id,date_hint)
        return next((r for r in self._parse(g,html,url) if r.draw_id==draw_id),None)


MIRROR_FILES={"mega645":"power645.jsonl","power655":"power655.jsonl","lotto535":"power535.jsonl",
              "max3d":"3d.jsonl","max3dplus":"3d.jsonl","max3dpro":"3d_pro.jsonl",
              "keno":"keno.jsonl","bingo18":"bingo18.jsonl"}


class GithubMirrorSource:
    name="vietvudanh/vietlott-data"
    def __init__(self, client: Any, base_url: str = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data") -> None:
        self.cache,self.base_url=TextCache(client,3),base_url.rstrip("/")

    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord | None:
        g=game_code(game)
        if g not in MIRROR_FILES:
            return None
        url=self.base_url+"/"+MIRROR_FILES[g]
        text=await self.cache.get(url)
        for line in text.splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            if int(str(row.get("id",row.get("draw_id"))).lstrip("#"))==draw_id:
                return DrawRecord.from_legacy(g,row | {"source":self.name,"source_url":url})
        return None


class CanonicalGithubSource:
    name="pqminh-4/vietlott-data"
    def __init__(self, client: Any, base_url: str = "https://raw.githubusercontent.com/pqminh-4/vietlott-data/main/data/canonical") -> None:
        self.cache,self.base_url=TextCache(client,3),base_url.rstrip("/")

    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord | None:
        g=game_code(game)
        if g not in ("mega645","power655","lotto535"):
            return None
        url=self.base_url+"/"+g+".jsonl"
        for line in (await self.cache.get(url)).splitlines():
            if not line.strip():
                continue
            r=json.loads(line)
            if int(r["draw_id"])==draw_id:
                return DrawRecord.from_legacy(g,r | {"source":self.name,"source_url":r.get("source_url",url)})
        return None


class MonthlyArchiveSource:
    name="NhanAZ-Data/vietlott-research"
    def __init__(self, client: Any, base_url: str = "https://raw.githubusercontent.com/NhanAZ-Data/vietlott-research/main/datasets") -> None:
        self.cache,self.base_url=TextCache(client,8),base_url.rstrip("/")

    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord | None:
        g=game_code(game)
        product=ProductCode("max3d" if g=="max3dplus" else g)
        if g in ("keno","bingo18"):
            if not date_hint:
                return None
            filename=date.fromisoformat(date_hint).strftime("%Y-%m")+".csv"
        else:
            filename="all.csv"
        url=f"{self.base_url}/draws/{product.value}/{filename}"
        # Convert only the requested row; do not hydrate an entire month of models per id.
        selected=[r for r in read_csv(await self.cache.get(url)) if int(str(r["draw_id"]).lstrip("#"))==draw_id]
        if not selected:
            return None
        for raw in selected:
            for field in ("product", "game", "game_type"):
                if raw.get(field) and game_code(raw[field]) not in {g, product.value}:
                    raise ValueError("archive row declares a different product")
        converted=convert_draws(product,selected)
        if not converted.rows:
            return None
        return DrawRecord.from_legacy(g,converted.rows[0] | {"source":self.name,"source_url":url})


class TargetedFetcher:
    def __init__(self, sources: list[DrawSource], *, source_timeout_s: float = 40) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        if not math.isfinite(source_timeout_s) or source_timeout_s <= 0:
            raise ValueError("source_timeout_s must be finite and positive")
        self.sources=sources
        self.source_timeout_s=source_timeout_s
        self.attempts: list[dict[str,Any]]=[]

    async def fetch(self, game: str, draw_id: int, date_hint: str | None = None) -> DrawRecord:
        g=game_code(game)
        for source in self.sources:
            name=getattr(source,"name",type(source).__name__)
            try:
                # Bound the whole adapter (detail + Ajax), not just each request.
                # A blocked first source must leave time for the mirrors.
                async with asyncio.timeout(self.source_timeout_s):
                    row=await source.fetch(g,draw_id,date_hint)
                if row is None:
                    self.attempts.append({"source":name,"game_type":g,"draw_id":draw_id,"status":"empty"})
                    continue
                row=DrawRecord.model_validate(row.model_dump())
                if row.draw_id!=draw_id or row.game_type.value!=g:
                    raise ValueError("source returned a different draw or game")
            except (SourceError,OSError,ValueError,KeyError,TypeError) as exc:
                # Never persist exception strings that may embed credentials/cookies.
                self.attempts.append({"source":name,"game_type":g,"draw_id":draw_id,"status":"rejected","error":type(exc).__name__})
                continue
            self.attempts.append({"source":name,"game_type":g,"draw_id":draw_id,"status":"ok"})
            return row
        raise SourceError(f"all sources exhausted for {g}/{draw_id}")


def default_sources(client: Any) -> list[DrawSource]:
    return [OfficialSource(client),MonthlyArchiveSource(client),CanonicalGithubSource(client),GithubMirrorSource(client)]
