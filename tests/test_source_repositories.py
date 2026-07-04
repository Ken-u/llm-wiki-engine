from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects.models import Project, ProjectMember, ProjectSourceRepository
from app.projects.router import router as projects_router
from app.projects.source_repositories import (
    create_default_source_repository,
    get_source_repository_or_404,
    infer_repo_name,
    list_source_repositories,
    normalize_source_repo_key,
    source_repo_checkout_root,
)


def _route_paths() -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in projects_router.routes
        for method in getattr(route, "methods", set())
    }


def test_source_repository_api_routes_are_registered():
    assert ("GET", "/api/projects/{project_id}/source-repositories") in _route_paths()
    assert ("POST", "/api/projects/{project_id}/source-repositories") in _route_paths()
    assert ("PATCH", "/api/projects/{project_id}/source-repositories/{repo_id}") in _route_paths()
    assert ("DELETE", "/api/projects/{project_id}/source-repositories/{repo_id}") in _route_paths()
    assert ("POST", "/api/projects/{project_id}/source-repositories/{repo_id}/test") in _route_paths()
    assert ("POST", "/api/projects/{project_id}/source-repositories/{repo_id}/sync") in _route_paths()
    assert ("POST", "/api/projects/{project_id}/source-repositories/sync-all") in _route_paths()


def test_source_repository_crud_api_hides_auth_and_enforces_owner_permissions(tmp_path):
    async def run():
        from app.auth.deps import get_current_user
        from app.database import get_db
        from app.main import app

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repos-api.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as seed_db:
            owner = User(id=1, username="owner", password_hash="x", role="user")
            viewer = User(id=2, username="viewer", password_hash="x", role="user")
            outsider = User(id=3, username="outsider", password_hash="x", role="user")
            project = Project(id="project-1", name="Project", slug="project", description="", created_by=1)
            db_viewer = ProjectMember(project_id="project-1", user_id=2, role="viewer")
            db_owner = ProjectMember(project_id="project-1", user_id=1, role="owner")
            seed_db.add_all([owner, viewer, outsider, project, db_owner, db_viewer])
            await seed_db.commit()

        current_user = owner

        async def override_get_db():
            async with Session() as db:
                yield db

        async def override_get_current_user():
            return current_user

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_current_user] = override_get_current_user
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                create_resp = await client.post(
                    "/api/projects/project-1/source-repositories",
                    json={
                        "key": "Frontend-Docs",
                        "name": "Frontend Docs",
                        "repo_url": "https://git.example.com/org/frontend-docs.git",
                        "branch": "main",
                        "username": "bot",
                        "auth_token": "secret-token",
                        "author_name": "Docs Bot",
                        "author_email": "docs@example.com",
                        "sync_enabled": True,
                        "auto_compile": True,
                        "sync_time": "03:15",
                    },
                )
                assert create_resp.status_code == 201
                created = create_resp.json()
                assert created["key"] == "frontend-docs"
                assert created["name"] == "Frontend Docs"
                assert created["auth_configured"] is True
                assert "auth_token" not in created

                duplicate_resp = await client.post(
                    "/api/projects/project-1/source-repositories",
                    json={
                        "key": "frontend-docs",
                        "name": "Duplicate",
                        "repo_url": "https://git.example.com/org/duplicate.git",
                    },
                )
                assert duplicate_resp.status_code == 409

                list_resp = await client.get("/api/projects/project-1/source-repositories")
                assert list_resp.status_code == 200
                listed = list_resp.json()
                assert listed[0]["id"] == created["id"]
                assert listed[0]["auth_configured"] is True
                assert "auth_token" not in listed[0]

                clear_token_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"clear_auth_token": True},
                )
                assert clear_token_resp.status_code == 200
                assert clear_token_resp.json()["auth_configured"] is False

                get_after_clear_resp = await client.get("/api/projects/project-1/source-repositories")
                assert get_after_clear_resp.status_code == 200
                assert get_after_clear_resp.json()[0]["auth_configured"] is False

                set_token_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"auth_token": "new-secret-token"},
                )
                assert set_token_resp.status_code == 200
                assert set_token_resp.json()["auth_configured"] is True

                empty_token_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"auth_token": ""},
                )
                assert empty_token_resp.status_code == 200
                assert empty_token_resp.json()["auth_configured"] is True

                current_user = viewer
                viewer_list_resp = await client.get("/api/projects/project-1/source-repositories")
                assert viewer_list_resp.status_code == 200

                viewer_patch_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"branch": "release"},
                )
                assert viewer_patch_resp.status_code == 403

                current_user = owner
                update_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"branch": "release", "key": "ignored-key"},
                )
                assert update_resp.status_code == 200
                updated = update_resp.json()
                assert updated["branch"] == "release"
                assert updated["key"] == "frontend-docs"

                current_user = outsider
                outsider_list_resp = await client.get("/api/projects/project-1/source-repositories")
                assert outsider_list_resp.status_code == 403

                current_user = owner
                delete_resp = await client.delete(
                    f"/api/projects/project-1/source-repositories/{created['id']}"
                )
                assert delete_resp.status_code == 204

                empty_list_resp = await client.get("/api/projects/project-1/source-repositories")
                assert empty_list_resp.status_code == 200
                assert empty_list_resp.json() == []

                missing_resp = await client.patch(
                    f"/api/projects/project-1/source-repositories/{created['id']}",
                    json={"branch": "main"},
                )
                assert missing_resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db, None)
            app.dependency_overrides.pop(get_current_user, None)
            await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_source_repository_create_enabled_and_delete_refresh_sync_jobs(monkeypatch):
    async def run():
        import asyncio

        from app.projects import router as projects_router

        created_tasks = []
        register_calls = 0

        async def fake_register_sync_jobs():
            nonlocal register_calls
            register_calls += 1

        def fake_create_task(coro):
            created_tasks.append(coro)
            return SimpleNamespace()

        async def fake_check_membership(db, project_id, user, require=None):
            return None

        async def fake_get_project_or_404(db, project_id):
            return SimpleNamespace(id=project_id)

        async def fake_create_source_repository(db, project, **kwargs):
            return SimpleNamespace(
                id="repo-1",
                project_id=project.id,
                key=kwargs["key"],
                name=kwargs["name"],
                repo_url=kwargs["repo_url"],
                branch=kwargs["branch"],
                username=kwargs["username"],
                auth_token=kwargs["auth_token"],
                author_name=kwargs["author_name"],
                author_email=kwargs["author_email"],
                sync_enabled=kwargs["sync_enabled"],
                auto_compile=kwargs["auto_compile"],
                sync_time=kwargs["sync_time"],
                last_sync_at=None,
                last_sync_status="idle",
                last_sync_error="",
                created_at=None,
                updated_at=None,
            )

        async def fake_delete_source_repository(db, project, repo_id):
            return None

        import app.projects.git_sync as git_sync

        monkeypatch.setattr(projects_router.service, "check_membership", fake_check_membership)
        monkeypatch.setattr(projects_router.service, "get_project_or_404", fake_get_project_or_404)
        monkeypatch.setattr(projects_router.source_repo_service, "create_source_repository", fake_create_source_repository)
        monkeypatch.setattr(projects_router.source_repo_service, "delete_source_repository", fake_delete_source_repository)
        monkeypatch.setattr(git_sync, "register_sync_jobs", fake_register_sync_jobs)
        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        body = projects_router.SourceRepositoryCreateRequest(
            key="docs",
            name="Docs",
            repo_url="https://git.example.com/org/docs.git",
            auth_token="secret",
            sync_enabled=True,
        )

        await projects_router.create_project_source_repository(
            "project-1",
            body,
            user=SimpleNamespace(id=1),
            db=object(),
        )
        await projects_router.delete_project_source_repository(
            "project-1",
            "repo-1",
            user=SimpleNamespace(id=1),
            db=object(),
        )

        assert len(created_tasks) == 2
        for coro in created_tasks:
            await coro
        assert register_calls == 2

    import asyncio

    asyncio.run(run())


def test_project_git_patch_updates_existing_default_source_repository(tmp_path, monkeypatch):
    async def run():
        from app.projects import router as projects_router

        async def fake_register_sync_jobs():
            return None

        import app.projects.git_sync as git_sync

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy-default-sync.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as seed_db:
            owner = User(id=1, username="owner", password_hash="x", role="user")
            project = Project(
                id="project-1",
                name="Project",
                slug="project",
                description="",
                created_by=1,
                git_repo_url="https://git.example.com/org/old.git",
                git_branch="old",
                git_username="old-user",
                git_auth_token="old-token",
                git_author_name="Old Bot",
                git_author_email="old@example.com",
                git_sync_enabled=False,
                git_sync_auto_compile=False,
                git_sync_time="01:00",
            )
            member = ProjectMember(project_id="project-1", user_id=1, role="owner")
            default_repo = ProjectSourceRepository(
                id="default-repo",
                project_id="project-1",
                key="default",
                name="old",
                repo_url="https://git.example.com/org/old.git",
                branch="old",
                username="old-user",
                auth_token="old-token",
                author_name="Old Bot",
                author_email="old@example.com",
                sync_enabled=False,
                auto_compile=False,
                sync_time="01:00",
            )
            seed_db.add_all([owner, project, member, default_repo])
            await seed_db.commit()

        monkeypatch.setattr(git_sync, "register_sync_jobs", fake_register_sync_jobs)
        try:
            async with Session() as db:
                resp = await projects_router.update_project(
                    "project-1",
                    projects_router.UpdateProjectRequest(
                        git_repo_url="https://git.example.com/org/new.git",
                        git_branch="main",
                        git_username="new-user",
                        git_auth_token="new-token",
                        git_author_name="New Bot",
                        git_author_email="new@example.com",
                        git_sync_enabled=True,
                        git_sync_auto_compile=True,
                        git_sync_time="03:30",
                    ),
                    user=User(id=1, username="owner", password_hash="x", role="user"),
                    db=db,
                )
                assert resp.id == "project-1"

            async with Session() as db:
                repo = (
                    await db.execute(
                        select(ProjectSourceRepository).where(ProjectSourceRepository.id == "default-repo")
                    )
                ).scalar_one()
                assert repo.repo_url == "https://git.example.com/org/new.git"
                assert repo.branch == "main"
                assert repo.username == "new-user"
                assert repo.auth_token == "new-token"
                assert repo.author_name == "New Bot"
                assert repo.author_email == "new@example.com"
                assert repo.sync_enabled is True
                assert repo.auto_compile is True
                assert repo.sync_time == "03:30"
        finally:
            await engine.dispose()

    import asyncio

    asyncio.run(run())


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
            assert repos[0].name == "docs"
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
            assert repo.name == "docs"
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


def test_get_source_repository_or_404_uses_id_and_scopes_to_project(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repos.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            project = Project(id="project-1", name="One", slug="one", description="", created_by=1)
            other_project = Project(id="project-2", name="Two", slug="two", description="", created_by=1)
            repo = ProjectSourceRepository(
                id="repo-1",
                project_id="project-1",
                key="default",
                name="Default",
            )
            other_repo = ProjectSourceRepository(
                id="repo-2",
                project_id="project-2",
                key="default",
                name="Other default",
            )
            db.add_all([project, other_project, repo, other_repo])
            await db.commit()

            with pytest.raises(HTTPException) as exc:
                await get_source_repository_or_404(db, project, "default")

            assert exc.value.status_code == 404
            assert exc.value.detail == "Source repository not found"

            with pytest.raises(HTTPException) as other_exc:
                await get_source_repository_or_404(db, project, "repo-2")

            assert other_exc.value.status_code == 404
            assert other_exc.value.detail == "Source repository not found"

            found = await get_source_repository_or_404(db, project, "repo-1")
            assert found.id == "repo-1"
            assert found.project_id == "project-1"

        await engine.dispose()

    import asyncio

    asyncio.run(run())
