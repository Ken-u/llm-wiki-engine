"""LanceDB vector search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lancedb

from app.embedding.service import _get_embeddings, _lancedb_path, TABLE_NAME

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    path: str
    page_id: str
    chunk_text: str
    heading_path: str
    score: float


async def search_vector(project_dir: str, query: str, top_k: int = 10) -> list[VectorResult]:
    db_path = _lancedb_path(project_dir)
    db = await lancedb.connect_async(db_path)
    table_names = await db.table_names()

    if TABLE_NAME not in table_names:
        return []

    # Get query embedding
    embeddings = await _get_embeddings([query])
    if not embeddings:
        return []

    query_vec = embeddings[0]
    table = await db.open_table(TABLE_NAME)
    results = (
        await table.vector_search(query_vec)
        .limit(top_k)
        .to_pandas()
    )

    output = []
    for _, row in results.iterrows():
        output.append(VectorResult(
            path=f"wiki/{row.get('page_id', '')}.md",
            page_id=row.get("page_id", ""),
            chunk_text=row.get("chunk_text", ""),
            heading_path=row.get("heading_path", ""),
            score=float(row.get("_distance", 0)),
        ))

    return output
