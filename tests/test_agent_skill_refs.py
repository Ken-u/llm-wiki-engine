"""Tests for signed source refs used by Agent Skills."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent, AgentProject
from app.agents import service
from app.agents.skill_refs import sign_source_ref, verify_source_ref
from app.auth.models import User
from app.database import Base, get_db
from app.main import app
from app.projects.models import Project


def test_signed_source_ref_round_trip_and_tamper_rejection():
    ref = sign_source_ref(
        agent_id="agent-1",
        project_id="project-1",
        doc_name="guide.md",
        display_name="Guide",
        max_age_seconds=60,
    )

    payload = verify_source_ref(ref, max_age_seconds=60)
    assert payload.agent_id == "agent-1"
    assert payload.project_id == "project-1"
    assert payload.doc_name == "guide.md"
    assert payload.display_name == "Guide"

    tampered = ref[:-1] + ("a" if ref[-1] != "a" else "b")
    assert verify_source_ref(tampered, max_age_seconds=60) is None


def test_signed_source_ref_rejects_expired_payload():
    ref = sign_source_ref(
        agent_id="agent-1",
        project_id="project-1",
        doc_name="guide.md",
        display_name="Guide",
        max_age_seconds=-1,
    )

    assert verify_source_ref(ref, max_age_seconds=60) is None


async def _build_client(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    projects_dir = tmp_path / "projects"
    cfg = SimpleNamespace(server=SimpleNamespace(projects_dir=str(projects_dir)))
    monkeypatch.setattr("app.projects.models.get_config", lambda: cfg)

    project_dir = projects_dir / "project-1"
    source_dir = project_dir / "raw" / "sources"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# Guide\n\nOriginal text", encoding="utf-8")

    async with Session() as db:
        user = User(id=1, username="owner", password_hash="x", role="user")
        project = Project(
            id="project-1",
            name="Knowledge",
            slug="knowledge",
            description="",
            _disk_path=str(project_dir),
            created_by=1,
            project_type="knowledge_base",
        )
        other_project = Project(
            id="project-2",
            name="Other",
            slug="other",
            description="",
            _disk_path=str(tmp_path / "other"),
            created_by=1,
            project_type="knowledge_base",
        )
        agent = Agent(
            id="agent-1",
            name="Skill Agent",
            description="",
            system_prompt="",
            is_public=True,
            require_api_key=True,
            created_by=1,
        )
        db.add(user)
        db.add(project)
        db.add(other_project)
        db.add(agent)
        db.add(AgentProject(agent_id=agent.id, project_id=project.id))
        await db.commit()
        token = await service.regenerate_skill_token(db, agent)

    async def override_db():
        async with Session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, token


def test_skill_document_content_reads_signed_source_ref(tmp_path, monkeypatch):
    async def run():
        client, token = await _build_client(tmp_path, monkeypatch)
        ref = sign_source_ref(
            agent_id="agent-1",
            project_id="project-1",
            doc_name="guide.md",
            display_name="Guide",
            max_age_seconds=60,
        )
        try:
            response = await client.get(
                "/api/public/skills/documents/content",
                params={"ref": ref},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            assert "Original text" in response.text
        finally:
            await client.aclose()
            app.dependency_overrides.clear()

    asyncio.run(run())


def test_skill_document_content_rejects_cross_agent_or_unbound_project(tmp_path, monkeypatch):
    async def run():
        client, token = await _build_client(tmp_path, monkeypatch)
        ref = sign_source_ref(
            agent_id="agent-1",
            project_id="project-2",
            doc_name="guide.md",
            display_name="Guide",
            max_age_seconds=60,
        )
        try:
            response = await client.get(
                "/api/public/skills/documents/content",
                params={"ref": ref},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 404
        finally:
            await client.aclose()
            app.dependency_overrides.clear()

    asyncio.run(run())
