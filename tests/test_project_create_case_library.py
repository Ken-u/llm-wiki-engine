"""Tests for creating a project as a case library."""

import asyncio
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects import service
from app.projects.models import Project, ProjectMember
from app.projects.router import CreateProjectRequest


def test_create_project_request_accepts_case_library_fields():
    body = CreateProjectRequest(
        name="Cases",
        slug="cases",
        as_case_library=True,
        main_project_id="main-1",
    )

    assert body.as_case_library is True
    assert body.main_project_id == "main-1"


def test_create_project_as_case_library_binds_main_project(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'projects.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        monkeypatch.setattr(
            "app.projects.service.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos"))),
        )
        monkeypatch.setattr(
            "app.projects.models.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos"))),
        )

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="hash", role="user")
            db.add(user)
            db.add(Project(id="main-1", name="Main", slug="main", description="", created_by=1))
            db.add(ProjectMember(project_id="main-1", user_id=1, role="owner"))
            await db.commit()

            case_project = await service.create_project(
                db,
                name="Cases",
                slug="cases",
                description="",
                user=user,
                case_library_main_project_id="main-1",
            )
            main_project = (
                await db.execute(select(Project).where(Project.id == "main-1"))
            ).scalar_one()

            assert main_project.ticket_project_id == case_project.id
            assert (tmp_path / "repos" / case_project.id / "wiki" / "index.md").exists()

        await engine.dispose()

    asyncio.run(run())


def test_create_project_as_case_library_rejects_already_bound_main(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'projects.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        monkeypatch.setattr(
            "app.projects.service.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos"))),
        )

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="hash", role="user")
            db.add(user)
            db.add(Project(id="main-1", name="Main", slug="main", description="", created_by=1, ticket_project_id="old-case"))
            db.add(Project(id="old-case", name="Old Cases", slug="old-cases", description="", created_by=1))
            db.add(ProjectMember(project_id="main-1", user_id=1, role="owner"))
            await db.commit()

            try:
                await service.create_project(
                    db,
                    name="Cases",
                    slug="cases",
                    description="",
                    user=user,
                    case_library_main_project_id="main-1",
                )
            except HTTPException as exc:
                assert exc.status_code == 400
            else:
                raise AssertionError("already-bound main project should be rejected")

        await engine.dispose()

    asyncio.run(run())


def test_delete_case_library_clears_main_project_binding(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'projects.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        monkeypatch.setattr(
            "app.projects.service.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos"))),
        )
        monkeypatch.setattr(
            "app.projects.models.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(tmp_path / "repos"))),
        )

        async with Session() as db:
            user = User(id=1, username="owner", password_hash="hash", role="user")
            main = Project(id="main-1", name="Main", slug="main", description="", created_by=1)
            case = Project(id="case-1", name="Cases", slug="cases", description="", created_by=1)
            main.ticket_project_id = case.id
            db.add(user)
            db.add(main)
            db.add(case)
            db.add(ProjectMember(project_id="main-1", user_id=1, role="owner"))
            db.add(ProjectMember(project_id="case-1", user_id=1, role="owner"))
            await db.commit()

            await service.delete_project(db, case)

            main_project = (
                await db.execute(select(Project).where(Project.id == "main-1"))
            ).scalar_one()
            assert main_project.ticket_project_id is None

        await engine.dispose()

    asyncio.run(run())
