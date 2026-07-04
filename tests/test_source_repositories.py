from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects.models import Project, ProjectSourceRepository
from app.projects.source_repositories import (
    create_default_source_repository,
    get_source_repository_or_404,
    infer_repo_name,
    list_source_repositories,
    normalize_source_repo_key,
    source_repo_checkout_root,
)


def test_normalize_source_repo_key_accepts_safe_values():
    assert normalize_source_repo_key("Docs_API-1") == "docs_api-1"


@pytest.mark.parametrize("value", ["", "../repo", "repo/name", "Repo.Name", "中文"])
def test_normalize_source_repo_key_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_source_repo_key(value)


def test_infer_repo_name_from_git_url():
    assert infer_repo_name("https://git.example.com/org/frontend-docs.git") == "frontend-docs"
    assert infer_repo_name("git@git.example.com:org/backend-docs.git") == "backend-docs"
    assert infer_repo_name("") == "默认源仓库"


def test_source_repo_checkout_root_uses_new_multi_source_path(tmp_path):
    project = SimpleNamespace(disk_path=str(tmp_path / "project"))
    repo = SimpleNamespace(key="frontend-docs")
    assert source_repo_checkout_root(project, repo) == tmp_path / "project" / ".llm-wiki" / "source-repos" / "frontend-docs"


def test_list_source_repositories_creates_default_when_empty(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repos.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            project = Project(
                id="project-1",
                name="Project",
                slug="project",
                description="",
                created_by=1,
                git_repo_url="https://git.example.com/org/docs.git",
            )
            db.add(project)
            await db.commit()

            repos = await list_source_repositories(db, project)

            assert len(repos) == 1
            assert repos[0].project_id == "project-1"
            assert repos[0].key == "default"
            assert repos[0].name == "默认源仓库"
            persisted = (
                await db.execute(select(ProjectSourceRepository).where(ProjectSourceRepository.project_id == "project-1"))
            ).scalar_one()
            assert persisted.id == repos[0].id

        await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_create_default_source_repository_copies_legacy_git_fields_and_flushes(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repos.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            project = Project(
                id="project-1",
                name="Project",
                slug="project",
                description="",
                created_by=1,
                git_repo_url="https://git.example.com/org/docs.git",
                git_branch="mainline",
                git_username="bot",
                git_auth_token="secret",
                git_author_name="Docs Bot",
                git_author_email="docs@example.com",
                git_sync_enabled=True,
                git_sync_auto_compile=True,
                git_sync_time="03:30",
                last_git_sync_status="failed",
                last_git_sync_error="old error",
            )
            db.add(project)
            await db.commit()

            commit_calls = 0
            refresh_calls = 0
            original_commit = db.commit
            original_refresh = db.refresh

            async def commit_spy():
                nonlocal commit_calls
                commit_calls += 1
                await original_commit()

            async def refresh_spy(instance, *args, **kwargs):
                nonlocal refresh_calls
                refresh_calls += 1
                await original_refresh(instance, *args, **kwargs)

            monkeypatch.setattr(db, "commit", commit_spy)
            monkeypatch.setattr(db, "refresh", refresh_spy)

            repo = await create_default_source_repository(db, project)

            assert repo.key == "default"
            assert repo.name == "默认源仓库"
            assert repo.repo_url == "https://git.example.com/org/docs.git"
            assert repo.branch == "mainline"
            assert repo.username == "bot"
            assert repo.auth_token == "secret"
            assert repo.author_name == "Docs Bot"
            assert repo.author_email == "docs@example.com"
            assert repo.sync_enabled is True
            assert repo.auto_compile is True
            assert repo.sync_time == "03:30"
            assert repo.last_sync_status == "failed"
            assert repo.last_sync_error == "old error"
            assert commit_calls == 1
            assert refresh_calls == 1

        await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_get_source_repository_or_404_scopes_to_project_and_raises_404(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repos.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            project = Project(id="project-1", name="One", slug="one", description="", created_by=1)
            other_project = Project(id="project-2", name="Two", slug="two", description="", created_by=1)
            other_repo = ProjectSourceRepository(
                id="repo-2",
                project_id="project-2",
                key="default",
                name="Other default",
            )
            db.add_all([project, other_project, other_repo])
            await db.commit()

            with pytest.raises(HTTPException) as exc:
                await get_source_repository_or_404(db, project, "default")

            assert exc.value.status_code == 404
            assert exc.value.detail == "Source repository not found"
            created = (
                await db.execute(select(ProjectSourceRepository).where(ProjectSourceRepository.project_id == "project-1"))
            ).scalars().all()
            assert created == []

        await engine.dispose()

    import asyncio

    asyncio.run(run())
