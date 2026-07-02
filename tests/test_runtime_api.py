from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path, PureWindowsPath
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.runtime.config import load_runtime_config
from app.search.bm25 import _project_relative_path


def _write_runtime_fixture(tmp_path: Path) -> Path:
    knowledge = tmp_path / "data" / "knowledge"
    wiki = knowledge / "wiki" / "entities"
    wiki.mkdir(parents=True)
    (wiki / "edla.md").write_text(
        "---\ntitle: EDLA\n---\n# EDLA\n\n## 定义\n\nEDLA 是本地测试知识。\n",
        encoding="utf-8",
    )
    (wiki / "other.md").write_text(
        "---\ntitle: Other\n---\n# Other\n\n## 定义\n\n这是无关内容。\n",
        encoding="utf-8",
    )
    (wiki / "another.md").write_text(
        "---\ntitle: Another\n---\n# Another\n\n## 定义\n\n这是另一篇无关内容。\n",
        encoding="utf-8",
    )
    (knowledge / "raw" / "sources").mkdir(parents=True)
    cases = tmp_path / "data" / "cases"
    (cases / ".llm-wiki" / "case-index").mkdir(parents=True)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
server:
  host: 127.0.0.1
  port: 8012
  open_browser: false
  api_key: ""
knowledge:
  name: Test Knowledge
  path: {knowledge}
  model_name: test-wiki
  system_prompt: Test prompt
case_library:
  enabled: false
  name: Cases
  path: {cases}
llm:
  provider: openai
  model: test-model
  api_key: ""
  api_base: null
  max_context_size: 128000
  context_compress_threshold: 0.85
  context_compress_target: 0.65
  timeout: 120
  ingest_temperature: 0.1
  chat_temperature: 0.7
  stream: true
embedding:
  enabled: false
  provider: openai
  model: test-embedding
  api_key: ""
  api_base: null
  dimensions: null
search:
  rrf_k: 60
  default_top_k: 10
  filename_exact_bonus: 200
  phrase_in_title_bonus: 50
  wiki_fallback_vector_distance_threshold: 0.45
runtime:
  mode: auto
  max_tool_calls: 20
  debug_result_limit: 2000
hooks:
  enabled: false
  scripts: []
""",
        encoding="utf-8",
    )
    return config


def test_project_relative_path_is_posix_for_windows_paths():
    path = PureWindowsPath("C:/project/wiki/entities/edla.md")
    base = PureWindowsPath("C:/project")

    assert _project_relative_path(path, base) == "wiki/entities/edla.md"


def test_runtime_status_models_and_wiki(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    from app.runtime_main import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        status = client.get("/api/status").json()
        assert status["knowledge"]["wiki_exists"] is True

        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "test-wiki"

        tree = client.get("/api/wiki").json()
        assert tree[0]["name"] == "entities"

        page = client.get("/api/wiki/wiki/entities/edla.md").json()
        assert "EDLA 是本地测试知识" in page["content"]


def test_runtime_search_keyword(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    from app.runtime_main import app

    with TestClient(app) as client:
        resp = client.post("/api/search", json={"query": "EDLA", "mode": "keyword"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["keyword_hits"] >= 1
        assert data["results"][0]["path"] == "wiki/entities/edla.md"


def test_runtime_reindex_rebuilds_knowledge_embeddings(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    from app.runtime_main import app

    with patch("app.embedding.service.rebuild_project_embeddings", AsyncMock(return_value={"pages": 3, "chunks": 9})) as rebuild:
        with TestClient(app) as client:
            resp = client.post("/api/search/reindex")

    assert resp.status_code == 200
    assert resp.json() == {"pages": 3, "chunks": 9}
    assert rebuild.await_args.args[0] == str(tmp_path / "data" / "knowledge")


def test_runtime_indexes_rebuild_all_rebuilds_knowledge_and_cases(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    text = config.read_text(encoding="utf-8").replace("enabled: false", "enabled: true", 1)
    config.write_text(text, encoding="utf-8")
    load_runtime_config(config)

    from app.case_index.models import CaseManifest
    from app.runtime_main import app

    manifest = CaseManifest(
        status="ready",
        built_at="2026-07-02T00:00:00+00:00",
        source_count=2,
        case_count=2,
        chunk_count=5,
        embedding_model="test-embedding",
        embedding_dimensions=1536,
        errors=[],
    )

    with patch("app.embedding.service.rebuild_project_embeddings", AsyncMock(return_value={"pages": 3, "chunks": 9})) as rebuild_kb:
        with patch("app.case_index.builder.rebuild_case_index", AsyncMock(return_value=manifest)) as rebuild_cases:
            with TestClient(app) as client:
                resp = client.post("/api/indexes/rebuild", json={"target": "all"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["knowledge"] == {"pages": 3, "chunks": 9}
    assert data["cases"]["status"] == "ready"
    assert data["cases"]["case_count"] == 2
    assert rebuild_kb.await_args.args[0] == str(tmp_path / "data" / "knowledge")
    assert rebuild_cases.await_args.args[0] == str(tmp_path / "data" / "cases")


def test_runtime_indexes_rebuild_cases_requires_enabled_case_library(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    from app.runtime_main import app

    with TestClient(app) as client:
        resp = client.post("/api/indexes/rebuild", json={"target": "cases"})

    assert resp.status_code == 404


def test_runtime_chat_fast_stream(tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    from app.runtime_main import app

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "EDLA", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: token" in body
        assert "EDLA 是本地测试知识" in body
        assert "event: done" in body


def test_runtime_wiki_missing_page_falls_back_to_vector_match(monkeypatch, tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    import app.wiki.resolve as wiki_resolve
    from app.search.vector import VectorResult

    async def fake_search_vector(*_args, **_kwargs):
        return [
            VectorResult(
                path="wiki/edla.md",
                page_id="edla",
                chunk_text="EDLA 是本地测试知识。",
                heading_path="定义",
                score=0.2,
            )
        ]

    monkeypatch.setattr(wiki_resolve, "search_vector", fake_search_vector)

    from app.runtime_main import app

    with TestClient(app) as client:
        resp = client.get("/api/wiki/wiki/EDLA%20Alias.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "wiki/entities/edla.md"
        assert data["resolved_from"] == "wiki/EDLA Alias.md"
        assert data["resolution"]["method"] == "vector"
        assert data["resolution"]["distance"] == 0.2
        assert "EDLA 是本地测试知识" in data["content"]


def test_runtime_wiki_missing_page_rejects_low_similarity_vector_match(monkeypatch, tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    import app.wiki.resolve as wiki_resolve
    from app.search.vector import VectorResult

    async def fake_search_vector(*_args, **_kwargs):
        return [
            VectorResult(
                path="wiki/edla.md",
                page_id="edla",
                chunk_text="EDLA 是本地测试知识。",
                heading_path="定义",
                score=0.9,
            )
        ]

    monkeypatch.setattr(wiki_resolve, "search_vector", fake_search_vector)

    from app.runtime_main import app

    with TestClient(app) as client:
        resp = client.get("/api/wiki/wiki/EDLA%20Alias.md")
        assert resp.status_code == 404


def test_runtime_openai_stream_with_agent(monkeypatch, tmp_path: Path):
    config = _write_runtime_fixture(tmp_path)
    load_runtime_config(config)

    async def fake_run_agent_turn(*_args, **_kwargs) -> AsyncGenerator[str, None]:
        yield json.dumps({"token": "hello"})
        yield json.dumps({"done": True, "tool_traces": []})

    import app.runtime.router as runtime_router

    monkeypatch.setattr(runtime_router, "run_agent_turn", fake_run_agent_turn)

    from app.runtime_main import app

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-wiki",
                "stream": True,
                "messages": [{"role": "user", "content": "复杂问题 为什么"}],
            },
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "chat.completion.chunk" in body
        assert "hello" in body
        assert "data: [DONE]" in body
