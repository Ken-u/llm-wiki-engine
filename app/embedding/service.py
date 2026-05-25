"""Embedding service: chunk wiki pages and upsert into LanceDB.

Each project has its own LanceDB directory under .llm-wiki/lancedb/.
Table schema matches wiki_chunks_v2 from the original project.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import aiofiles
import lancedb
import litellm
import pyarrow as pa

from app.config import get_config
from app.embedding.chunker import chunk_markdown

logger = logging.getLogger(__name__)

TABLE_NAME = "wiki_chunks_v2"

SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("page_id", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("chunk_text", pa.string()),
    pa.field("heading_path", pa.string()),
    pa.field("vector", pa.list_(pa.float32())),
])


def _page_id_from_path(wiki_path: str) -> str:
    """Extract page_id from a wiki path like 'wiki/entities/foo.md' -> 'foo'."""
    name = Path(wiki_path).stem
    return re.sub(r"[^a-z0-9_-]", "-", name.lower())


def _lancedb_path(project_dir: str) -> str:
    p = Path(project_dir) / ".llm-wiki" / "lancedb"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    cfg = get_config().embedding
    if not cfg.enabled or not texts:
        return []

    kwargs = {
        "model": cfg.model if cfg.provider == "openai" or "/" in cfg.model else f"{cfg.provider}/{cfg.model}",
        "input": texts,
        "api_key": cfg.api_key or None,
        "dimensions": cfg.dimensions,
    }
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base

    resp = await litellm.aembedding(**kwargs)
    return [item["embedding"] for item in resp.data]


async def embed_pages(project_dir: str, wiki_paths: list[str]) -> int:
    """Embed a list of wiki page paths. Returns number of chunks upserted."""
    cfg = get_config().embedding
    if not cfg.enabled:
        return 0

    base = Path(project_dir)
    all_chunks = []

    for wp in wiki_paths:
        full_path = base / wp
        if not full_path.exists() or not full_path.suffix == ".md":
            continue

        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Strip frontmatter before chunking
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                content = content[end_idx + 3:].strip()

        if not content:
            continue

        page_id = _page_id_from_path(wp)
        chunks = chunk_markdown(content, chunk_size=512, overlap=64)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": f"{page_id}#{chunk.chunk_index}",
                "page_id": page_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.text,
                "heading_path": chunk.heading_path,
                "wiki_path": wp,
            })

    if not all_chunks:
        return 0

    # Batch embed
    texts = [c["chunk_text"] for c in all_chunks]
    try:
        vectors = await _get_embeddings(texts)
    except Exception:
        logger.exception("Embedding API call failed")
        return 0

    if len(vectors) != len(all_chunks):
        logger.error("Vector count mismatch: %d texts, %d vectors", len(texts), len(vectors))
        return 0

    # Prepare LanceDB records
    records = []
    for chunk_meta, vec in zip(all_chunks, vectors):
        records.append({
            "chunk_id": chunk_meta["chunk_id"],
            "page_id": chunk_meta["page_id"],
            "chunk_index": chunk_meta["chunk_index"],
            "chunk_text": chunk_meta["chunk_text"],
            "heading_path": chunk_meta["heading_path"],
            "vector": vec,
        })

    # Upsert to LanceDB
    db_path = _lancedb_path(project_dir)
    db = await lancedb.connect_async(db_path)
    table_names = await db.table_names()

    if TABLE_NAME in table_names:
        table = await db.open_table(TABLE_NAME)
        # Delete existing chunks for pages being re-embedded
        page_ids = list({r["page_id"] for r in records})
        for pid in page_ids:
            try:
                await table.delete(f"page_id = '{pid}'")
            except Exception:
                pass
        await table.add(records)
    else:
        await db.create_table(TABLE_NAME, records, schema=SCHEMA)

    logger.info("Embedded %d chunks for %d pages", len(records), len(wiki_paths))
    return len(records)
