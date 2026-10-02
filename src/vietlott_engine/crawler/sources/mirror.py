"""JSONL sources: the community GitHub mirror (HTTP) and local files (offline / air-gapped)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from vietlott_engine.core.games import GameCode, GameSpec
from vietlott_engine.core.logging import get_logger
from vietlott_engine.crawler.http import AsyncHttpClient
from vietlott_engine.crawler.schemas import MirrorRecord
from vietlott_engine.crawler.sources.base import DrawSource, FetchResult

log = get_logger(__name__)

MIRROR_FILES: dict[GameCode, str] = {
    GameCode.MEGA_645: "power645.jsonl",
    GameCode.POWER_655: "power655.jsonl",
    GameCode.LOTTO_535: "power535.jsonl",
}


def parse_jsonl(text: str, spec: GameSpec, since_id: int | None, source: str) -> FetchResult:
    result = FetchResult()
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            draw = MirrorRecord.model_validate(json.loads(line)).to_draw(spec, source=source)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            reason = str(exc).splitlines()[0]
            log.warning("line %d rejected: %s", line_no, reason)
            result.rejected.append((line[:200], reason))
            continue
        if since_id is None or draw.draw_id > since_id:
            result.draws.append(draw)
    return result


class GithubMirrorSource(DrawSource):
    """Downloads ``<base_url>/<game>.jsonl`` (one JSON record per draw)."""

    name = "github_mirror"

    def __init__(self, client: AsyncHttpClient, base_url: str) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")

    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        url = f"{self.base_url}/{MIRROR_FILES[spec.code]}"
        log.info("downloading %s", url)
        response = await self.client.get(url)
        return parse_jsonl(response.text, spec, since_id, self.name)


class JsonlFileSource(DrawSource):
    """Reads the same JSONL format from a local directory or file."""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    async def fetch(self, spec: GameSpec, since_id: int | None = None) -> FetchResult:
        file = self.path / MIRROR_FILES[spec.code] if self.path.is_dir() else self.path
        log.info("reading %s", file)
        return parse_jsonl(file.read_text(encoding="utf-8"), spec, since_id, self.name)
