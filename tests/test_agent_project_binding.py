"""Agents must bind knowledge_base projects only; case libraries use ticket_project_id."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents import service
from app.agents.models import Agent, AgentProject  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects.models import Project


def _make_env(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    Session = async_sessionmaker(engine, expire_on_commit=False)
    cfg = SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos")))
    monkeypatch.setattr("app.agents.service.get_config", lambda: cfg)
    monkeypatch.setattr("app.projects.service.get_config", lambda: cfg)
    monkeypatch.setattr("app.projects.models.get_config", lambda: cfg)
    return engine, Session


async def _seed_projects(Session, tmp_path):
    async with Session() as db:
        user = User(id=1, username="owner", password_hash="x", role="user")
        db.add(user)
        main = Project(
            id="main-1",
            name="Main KB",
            slug="main",
            description="",
            _disk_path=str(tmp_path / "repos" / "main"),
            created_by=1,
            project_type="knowledge_base",
        )
        case = Project(
            id="case-1",
            name="Cases",
            slug="cases",
            description="",
            _disk_path=str(tmp_path / "repos" / "cases"),
            created_by=1,
            project_type="case_library",
        )
        main.ticket_project_id = case.id
        db.add(main)
        db.add(case)
        await db.commit()


def test_validate_agent_project_ids_rejects_case_library(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_projects(Session, tmp_path)

        async with Session() as db:
            with pytest.raises(HTTPException) as exc:
                await service.validate_agent_project_ids(db, ["case-1"])
            assert exc.value.status_code == 400

    asyncio.run(run())


def test_validate_agent_project_ids_accepts_knowledge_base(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_projects(Session, tmp_path)

        async with Session() as db:
            await service.validate_agent_project_ids(db, ["main-1"])

    asyncio.run(run())


def test_create_agent_rejects_case_library_binding(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_projects(Session, tmp_path)

        async with Session() as db:
            with pytest.raises(HTTPException):
                await service.create_agent(
                    db,
                    name="Test Agent",
                    description="",
                    system_prompt="",
                    project_ids=["case-1"],
                    is_public=False,
                    require_api_key=False,
                    user_id=1,
                )

    asyncio.run(run())


def test_create_agent_binds_knowledge_base_and_resolves_ticket(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_projects(Session, tmp_path)

        async with Session() as db:
            agent, _ = await service.create_agent(
                db,
                name="Test Agent",
                description="",
                system_prompt="",
                project_ids=["main-1"],
                is_public=False,
                require_api_key=False,
                user_id=1,
            )
            projects = await service.get_agent_projects(db, agent.id)
            assert len(projects) == 1
            assert projects[0].id == "main-1"
            ticket = await service.get_ticket_project(db, projects)
            assert ticket is not None
            assert ticket.id == "case-1"

    asyncio.run(run())


def test_get_agent_projects_excludes_case_library_direct_binding(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _seed_projects(Session, tmp_path)

        async with Session() as db:
            agent = Agent(
                id="agent-1",
                name="Legacy",
                description="",
                system_prompt="",
                is_public=False,
                require_api_key=False,
                created_by=1,
            )
            db.add(agent)
            db.add(AgentProject(agent_id=agent.id, project_id="main-1"))
            db.add(AgentProject(agent_id=agent.id, project_id="case-1"))
            await db.commit()

            projects = await service.get_agent_projects(db, agent.id)
            assert [p.id for p in projects] == ["main-1"]

    asyncio.run(run())
