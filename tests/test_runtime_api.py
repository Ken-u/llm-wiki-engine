from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi.testclient import TestClient

from app.runtime.config import load_runtime_config


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
