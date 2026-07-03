"""Tests for Runtime Skill export endpoints."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from app.runtime_main import app


def _runtime_settings(tmp_path, *, api_key: str = ""):
    project_dir = tmp_path / "knowledge"
    source_dir = project_dir / "raw" / "sources"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# Guide\n\nRuntime source text", encoding="utf-8")
    return SimpleNamespace(
        server=SimpleNamespace(api_key=api_key),
        knowledge=SimpleNamespace(
            name="Runtime Knowledge",
            path=str(project_dir),
            model_name="local-wiki",
            system_prompt="Use runtime knowledge.",
            system_prompt_override="",
        ),
        case_library=SimpleNamespace(enabled=False, name="Cases", path=""),
        runtime=SimpleNamespace(mode="auto", max_tool_calls=20, debug_result_limit=2000),
    )


def test_runtime_skill_download_includes_auth_when_configured(tmp_path, monkeypatch):
    settings = _runtime_settings(tmp_path, api_key="runtime-secret")
    monkeypatch.setattr("app.runtime.router.get_runtime_config", lambda: settings)

    async def run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/skill",
                headers={"Authorization": "Bearer runtime-secret"},
            )
        assert response.status_code == 200
        assert "Runtime Knowledge" in response.text
        assert "POST http://test/api/skill/chat" in response.text
        assert "Authorization: Bearer runtime-secret" in response.text

    asyncio.run(run())


def test_runtime_skill_chat_returns_signed_sources(tmp_path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    monkeypatch.setattr("app.runtime.router.get_runtime_config", lambda: settings)

    async def fake_events(_body):
        yield {"event": "token", "data": {"text": "Answer"}}
        yield {
            "event": "tool_result",
            "data": {
                "name": "read_raw",
                "result": {"path": "guide.md"},
            },
        }

    monkeypatch.setattr("app.runtime.router._chat_events", fake_events)

    async def run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/skill/chat", json={"message": "查 guide"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Answer"
        assert body["sources"][0]["name"] == "guide.md"
        assert body["sources"][0]["source_ref"]

    asyncio.run(run())


def test_runtime_skill_document_content_reads_signed_ref(tmp_path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    monkeypatch.setattr("app.runtime.router.get_runtime_config", lambda: settings)

    async def run():
        from app.runtime.router import sign_runtime_source_ref

        ref = sign_runtime_source_ref("guide.md")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/skill/documents/content", params={"ref": ref})
        assert response.status_code == 200
        assert "Runtime source text" in response.text

    asyncio.run(run())
