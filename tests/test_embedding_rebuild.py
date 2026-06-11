"""Tests for project-wide embedding rebuilds."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pyarrow as pa


def test_rebuild_project_embeddings_clears_lancedb_and_embeds_all_wiki_pages(tmp_path):
    project_dir = tmp_path
    (project_dir / "wiki" / "concepts").mkdir(parents=True)
    (project_dir / "wiki" / "index.md").write_text("# Index\n", encoding="utf-8")
    (project_dir / "wiki" / "concepts" / "ota.md").write_text("# OTA\n", encoding="utf-8")
    (project_dir / "raw" / "sources").mkdir(parents=True)
    (project_dir / "raw" / "sources" / "source.md").write_text("# Raw\n", encoding="utf-8")
    old_db = project_dir / ".llm-wiki" / "lancedb"
    old_db.mkdir(parents=True)
    (old_db / "old-file").write_text("stale", encoding="utf-8")

    async def run():
        from app.embedding.service import rebuild_project_embeddings

        with patch("app.embedding.service.embed_pages", AsyncMock(return_value=7)) as embed_pages:
            result = await rebuild_project_embeddings(str(project_dir))

        return result, embed_pages.await_args

    result, await_args = asyncio.run(run())

    assert result == {"pages": 2, "chunks": 7}
    assert not (old_db / "old-file").exists()
    assert await_args.args[0] == str(project_dir)
    assert await_args.args[1] == ["wiki/concepts/ota.md", "wiki/index.md"]


def test_search_router_exposes_reindex_response_model():
    from app.search.router import ReindexResponse, router

    routes = [
        route for route in router.routes
        if getattr(route, "path", "") == "/api/projects/{project_id}/search/reindex"
    ]

    assert routes
    assert ReindexResponse.model_fields["pages"].annotation is int
    assert ReindexResponse.model_fields["chunks"].annotation is int


def test_embed_pages_creates_fixed_size_vector_schema(tmp_path):
    project_dir = tmp_path
    (project_dir / "wiki").mkdir(parents=True)
    (project_dir / "wiki" / "index.md").write_text("# Index\n\nEmbedding content.\n", encoding="utf-8")

    class _FakeDb:
        created_schema = None

        async def table_names(self):
            return []

        async def create_table(self, _name, _records, *, schema):
            self.created_schema = schema
            return schema

    async def run():
        from app.embedding.service import embed_pages

        fake_db = _FakeDb()
        with patch("app.embedding.service._get_embeddings", AsyncMock(return_value=[[0.1] * 4096])):
            with patch("app.embedding.service.lancedb.connect_async", AsyncMock(return_value=fake_db)):
                await embed_pages(str(project_dir), ["wiki/index.md"])
        return fake_db.created_schema

    schema = asyncio.run(run())

    assert schema.field("vector").type == pa.list_(pa.float32(), list_size=4096)


def test_embed_pages_uses_configured_dimension_when_it_matches_embedding(tmp_path):
    project_dir = tmp_path
    (project_dir / "wiki").mkdir(parents=True)
    (project_dir / "wiki" / "index.md").write_text("# Index\n\nEmbedding content.\n", encoding="utf-8")

    class _FakeDb:
        created_schema = None

        async def table_names(self):
            return []

        async def create_table(self, _name, _records, *, schema):
            self.created_schema = schema
            return schema

    async def run():
        from app.embedding.service import embed_pages

        fake_db = _FakeDb()
        config = SimpleNamespace(embedding=SimpleNamespace(enabled=True, dimensions=1024))
        with patch("app.embedding.service.get_config", return_value=config):
            with patch("app.embedding.service._get_embeddings", AsyncMock(return_value=[[0.1] * 1024])):
                with patch("app.embedding.service.lancedb.connect_async", AsyncMock(return_value=fake_db)):
                    await embed_pages(str(project_dir), ["wiki/index.md"])
        return fake_db.created_schema

    schema = asyncio.run(run())

    assert schema.field("vector").type == pa.list_(pa.float32(), list_size=1024)


def test_embed_pages_falls_back_to_actual_dimension_when_config_conflicts(tmp_path, caplog):
    project_dir = tmp_path
    (project_dir / "wiki").mkdir(parents=True)
    (project_dir / "wiki" / "index.md").write_text("# Index\n\nEmbedding content.\n", encoding="utf-8")

    class _FakeDb:
        created_schema = None

        async def table_names(self):
            return []

        async def create_table(self, _name, _records, *, schema):
            self.created_schema = schema
            return schema

    async def run():
        from app.embedding.service import embed_pages

        fake_db = _FakeDb()
        config = SimpleNamespace(embedding=SimpleNamespace(enabled=True, dimensions=1024))
        with patch("app.embedding.service.get_config", return_value=config):
            with patch("app.embedding.service._get_embeddings", AsyncMock(return_value=[[0.1] * 4096])):
                with patch("app.embedding.service.lancedb.connect_async", AsyncMock(return_value=fake_db)):
                    await embed_pages(str(project_dir), ["wiki/index.md"])
        return fake_db.created_schema

    schema = asyncio.run(run())

    assert schema.field("vector").type == pa.list_(pa.float32(), list_size=4096)
    assert "Configured embedding dimension 1024 does not match actual embedding dimension 4096" in caplog.text
