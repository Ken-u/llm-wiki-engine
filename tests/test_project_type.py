"""Tests for project_type first-class field."""

import asyncio
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects import service
from app.projects.models import Project, ProjectMember
from app.projects.router import ProjectResponse


def _make_env(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    Session = async_sessionmaker(engine, expire_on_commit=False)
    cfg = SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos")))
    monkeypatch.setattr("app.projects.service.get_config", lambda: cfg)
    monkeypatch.setattr("app.projects.models.get_config", lambda: cfg)
    return engine, Session


# --- Task 1: Model fields ---

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


# --- Task 2: create_project writes project_type ---

def test_create_case_library_sets_project_type(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            main = Project(id="main-1", name="Main", slug="main", description="", created_by=1)
            db.add(main)
            db.add(ProjectMember(project_id="main-1", user_id=1, role="owner"))
            await db.commit()

            case_proj = await service.create_project(
                db, name="Cases", slug="cases", description="",
                user=user, case_library_main_project_id="main-1",
            )
            assert case_proj.project_type == "case_library"

        await engine.dispose()

    asyncio.run(run())


def test_create_normal_project_sets_knowledge_base(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            await db.commit()

            proj = await service.create_project(
                db, name="KB", slug="kb", description="", user=user,
            )
            assert proj.project_type == "knowledge_base"

        await engine.dispose()

    asyncio.run(run())


def test_create_case_library_without_main_project(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            await db.commit()

            proj = await service.create_project(
                db, name="Cases", slug="cases", description="",
                user=user, as_case_library=True,
            )
            assert proj.project_type == "case_library"

        await engine.dispose()

    asyncio.run(run())


# --- Task 3: API response ---

def test_project_response_includes_project_type():
    resp = ProjectResponse(
        id="p1", name="Test", slug="test", description="",
        created_by=1, created_at="2026-01-01T00:00:00",
        project_type="case_library",
        case_index_auto_rebuild=True,
    )
    assert resp.project_type == "case_library"
    assert resp.case_index_auto_rebuild is True


def test_project_response_defaults_project_type():
    resp = ProjectResponse(
        id="p2", name="Test", slug="test", description="",
        created_by=1, created_at="2026-01-01T00:00:00",
    )
    assert resp.project_type == "knowledge_base"
    assert resp.case_index_auto_rebuild is False


# --- Task 4: project_type immutable, case_index_auto_rebuild ---

def test_update_project_type_is_rejected(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            proj = Project(
                id="p1", name="KB", slug="kb", description="", created_by=1,
                project_type="knowledge_base",
            )
            db.add(proj)
            await db.commit()

            try:
                await service.update_project(db, proj, project_type="case_library")
            except HTTPException as exc:
                assert exc.status_code == 400
                assert "project_type" in exc.detail.lower()
            else:
                raise AssertionError("project_type update should be rejected")

        await engine.dispose()

    asyncio.run(run())


def test_update_case_index_auto_rebuild_for_case_library(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            proj = Project(
                id="cl1", name="Cases", slug="cases", description="",
                created_by=1, project_type="case_library",
            )
            db.add(proj)
            await db.commit()

            proj = await service.update_project(db, proj, case_index_auto_rebuild=True)
            assert proj.case_index_auto_rebuild is True

        await engine.dispose()

    asyncio.run(run())


def test_update_case_index_auto_rebuild_rejected_for_knowledge_base(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            proj = Project(
                id="kb1", name="KB", slug="kb", description="", created_by=1,
                project_type="knowledge_base",
            )
            db.add(proj)
            await db.commit()

            try:
                await service.update_project(db, proj, case_index_auto_rebuild=True)
            except HTTPException as exc:
                assert exc.status_code == 400
            else:
                raise AssertionError("case_index_auto_rebuild on knowledge_base should be rejected")

        await engine.dispose()

    asyncio.run(run())


# --- Task 5: Binding rules ---

def test_bind_ticket_project_must_be_case_library(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            main = Project(id="main-1", name="Main", slug="main", description="", created_by=1)
            other_kb = Project(id="kb-2", name="Other KB", slug="other-kb", description="", created_by=1)
            db.add(main)
            db.add(other_kb)
            await db.commit()

            try:
                await service.update_project(db, main, ticket_project_id="kb-2")
            except HTTPException as exc:
                assert exc.status_code == 400
                assert "case_library" in exc.detail.lower()
            else:
                raise AssertionError("binding a knowledge_base as ticket project should be rejected")

        await engine.dispose()

    asyncio.run(run())


def test_bind_ticket_project_accepts_case_library(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            main = Project(id="main-1", name="Main", slug="main", description="", created_by=1)
            case_lib = Project(
                id="cl-1", name="Cases", slug="cases", description="",
                created_by=1, project_type="case_library",
            )
            db.add(main)
            db.add(case_lib)
            await db.commit()

            main = await service.update_project(db, main, ticket_project_id="cl-1")
            assert main.ticket_project_id == "cl-1"

        await engine.dispose()

    asyncio.run(run())


def test_case_library_cannot_bind_another_case_library(tmp_path, monkeypatch):
    async def run():
        engine, Session = _make_env(tmp_path, monkeypatch)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="x", role="user")
            db.add(user)
            cl1 = Project(id="cl-1", name="Cases1", slug="cases1", description="", created_by=1, project_type="case_library")
            cl2 = Project(id="cl-2", name="Cases2", slug="cases2", description="", created_by=1, project_type="case_library")
            db.add(cl1)
            db.add(cl2)
            await db.commit()

            try:
                await service.update_project(db, cl1, ticket_project_id="cl-2")
            except HTTPException as exc:
                assert exc.status_code == 400
            else:
                raise AssertionError("case_library should not bind another case_library")

        await engine.dispose()

    asyncio.run(run())


# --- Task 6: Stale marker ---

def test_case_index_stale_marker(tmp_path):
    from app.case_index.builder import (
        mark_case_index_stale,
        clear_case_index_stale,
        is_case_index_stale,
    )

    project_dir = str(tmp_path / "proj")
    (tmp_path / "proj" / ".llm-wiki").mkdir(parents=True)

    assert is_case_index_stale(project_dir) is False

    mark_case_index_stale(project_dir)
    assert is_case_index_stale(project_dir) is True

    clear_case_index_stale(project_dir)
    assert is_case_index_stale(project_dir) is False


# --- Task 7: Case index API guard ---

def test_case_index_require_case_library_rejects_kb():
    from app.case_index.router import _require_case_library
    proj = Project(
        id="kb1", name="KB", slug="kb", description="", created_by=1,
        project_type="knowledge_base",
    )
    try:
        _require_case_library(proj)
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("knowledge_base should be rejected")


def test_case_index_require_case_library_accepts_cl():
    from app.case_index.router import _require_case_library
    proj = Project(
        id="cl1", name="Cases", slug="cases", description="", created_by=1,
        project_type="case_library",
    )
    _require_case_library(proj)
