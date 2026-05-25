"""SHA256-based ingest cache to skip unchanged documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import aiofiles


def _cache_path(project_dir: str) -> Path:
    return Path(project_dir) / ".llm-wiki" / "ingest-cache.json"


async def load_cache(project_dir: str) -> dict:
    p = _cache_path(project_dir)
    if not p.exists():
        return {}
    async with aiofiles.open(p, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def save_cache(project_dir: str, cache: dict) -> None:
    p = _cache_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(p, "w", encoding="utf-8") as f:
        await f.write(json.dumps(cache, ensure_ascii=False, indent=2))


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def check_cache(project_dir: str, source_identity: str, content: str) -> bool:
    """Return True if the content is unchanged since last ingest."""
    cache = await load_cache(project_dir)
    h = content_hash(content)
    return cache.get(source_identity) == h


async def update_cache(project_dir: str, source_identity: str, content: str, files_written: list[str]) -> None:
    cache = await load_cache(project_dir)
    cache[source_identity] = content_hash(content)
    await save_cache(project_dir, cache)
