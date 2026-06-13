"""Tests for project_type first-class field."""

import asyncio
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects.models import Project, ProjectMember


def _make_env(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    Session = async_sessionmaker(engine, expire_on_commit=False)
    cfg = SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos")))
    monkeypatch.setattr("app.projects.service.get_config", lambda: cfg)
    monkeypatch.setattr("app.projects.models.get_config", lambda: cfg)
    return engine, Session


def test_project_model_has_project_type_field(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            proj = Project(
                id="p1", name="Test", slug="test", description="",
                created_by=1, project_type="knowledge_base",
            )
            db.add(proj)
            await db.commit()
            await db.refresh(proj)
            assert proj.project_type == "knowledge_base"
            assert proj.case_index_auto_rebuild is False

        await engine.dispose()

    asyncio.run(run())


def test_project_type_defaults_to_knowledge_base(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            proj = Project(
                id="p2", name="Default", slug="default", description="",
                created_by=1,
            )
            db.add(proj)
            await db.commit()
            await db.refresh(proj)
            assert proj.project_type == "knowledge_base"

        await engine.dispose()

    asyncio.run(run())
