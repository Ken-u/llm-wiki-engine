"""Tests for public Agent Skill endpoints."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent, AgentProject
from app.agents import service
from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import Base, get_db
from app.main import app
from app.projects.models import Project


async def _build_client(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with Session() as db:
        user = User(id=1, username="owner", password_hash="x", role="user")
        project = Project(
            id="project-1",
            name="Knowledge",
            slug="knowledge",
            description="",
            _disk_path=str(tmp_path / "project"),
            created_by=1,
            project_type="knowledge_base",
        )
        agent = Agent(
            id="agent-1",
            name="Skill Agent",
            description="Agent for Skill",
            system_prompt="Use knowledge.",
            is_public=True,
            require_api_key=True,
            created_by=1,
        )
        db.add(user)
        db.add(project)
        db.add(agent)
        db.add(AgentProject(agent_id=agent.id, project_id=project.id))
        await db.commit()

    async def override_db():
        async with Session() as db:
            yield db

    async def override_user():
        return User(id=1, username="owner", password_hash="x", role="user")

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, Session


def test_regenerate_skill_token_and_download_skill_markdown(tmp_path):
    async def run():
        client, _ = await _build_client(tmp_path)
        try:
            regen = await client.post("/api/agents/agent-1/regenerate-skill-token")
            assert regen.status_code == 200
            token = regen.json()["skill_token"]
            assert token.startswith("lws_")

            download = await client.get(f"/api/public/skills/{token}")
            assert download.status_code == 200
            assert "text/markdown" in download.headers["content-type"]
            assert "Skill Agent" in download.text
            assert "POST http://test/api/public/skills/chat" in download.text
            assert token in download.text
        finally:
            await client.aclose()
            app.dependency_overrides.clear()

    asyncio.run(run())


def test_public_skill_download_rejects_invalid_token(tmp_path):
    async def run():
        client, _ = await _build_client(tmp_path)
        try:
            response = await client.get("/api/public/skills/lws_invalid")
            assert response.status_code == 401
        finally:
            await client.aclose()
            app.dependency_overrides.clear()

    asyncio.run(run())


def test_skill_chat_resolves_agent_from_token_without_agent_id(tmp_path, monkeypatch):
    async def fake_collect_answer(db, agent, message):
        assert agent.id == "agent-1"
        assert message == "查一下 GMS"
        return "GMS answer", []

    async def run():
        client, Session = await _build_client(tmp_path)
        async with Session() as db:
            agent = await db.get(Agent, "agent-1")
            token = await service.regenerate_skill_token(db, agent)

        monkeypatch.setattr(
            "app.agents.skill_router.collect_skill_answer",
            fake_collect_answer,
        )

        try:
            response = await client.post(
                "/api/public/skills/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "查一下 GMS"},
            )
            assert response.status_code == 200
            assert response.json() == {"answer": "GMS answer", "sources": []}
        finally:
            await client.aclose()
            app.dependency_overrides.clear()

    asyncio.run(run())
