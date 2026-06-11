"""Tests for LanceDB vector search resilience."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.search import vector


class _DimensionMismatchQuery:
    def limit(self, _top_k: int):
        return self

    async def to_pandas(self):
        raise ValueError(
            "Invalid input, No vector column found to match with the query vector dimension: 4096"
        )


class _FakeTable:
    def vector_search(self, _query_vec):
        return _DimensionMismatchQuery()


class _FakeDb:
    async def table_names(self):
        return [vector.TABLE_NAME]

    async def open_table(self, _name: str):
        return _FakeTable()


def test_search_vector_returns_empty_on_lancedb_dimension_mismatch():
    async def run():
        with patch("app.search.vector._get_embeddings", AsyncMock(return_value=[[0.1] * 4096])):
            with patch("app.search.vector.lancedb.connect_async", AsyncMock(return_value=_FakeDb())):
                return await vector.search_vector("/tmp/project", "query", top_k=5)

    results = asyncio.run(run())
    assert results == []
