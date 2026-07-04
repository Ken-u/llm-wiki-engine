"""Tests for git-related project API behavior."""

import sys
from types import ModuleType

if "apscheduler" not in sys.modules:
    apscheduler = ModuleType("apscheduler")
    schedulers = ModuleType("apscheduler.schedulers")
    asyncio_mod = ModuleType("apscheduler.schedulers.asyncio")
    triggers = ModuleType("apscheduler.triggers")
    cron_mod = ModuleType("apscheduler.triggers.cron")

    class AsyncIOScheduler:
        def __init__(self):
            self.running = False

    class CronTrigger:
        def __init__(self, *args, **kwargs):
            pass

    asyncio_mod.AsyncIOScheduler = AsyncIOScheduler
    cron_mod.CronTrigger = CronTrigger
    sys.modules["apscheduler"] = apscheduler
    sys.modules["apscheduler.schedulers"] = schedulers
    sys.modules["apscheduler.schedulers.asyncio"] = asyncio_mod
    sys.modules["apscheduler.triggers"] = triggers
    sys.modules["apscheduler.triggers.cron"] = cron_mod

from app.projects.models import Project


def test_project_model_has_git_fields():
    cols = {c.name for c in Project.__table__.columns}
    expected = {
        "git_repo_url", "git_branch", "git_username", "git_auth_token",
        "git_author_name", "git_author_email", "git_sync_enabled",
        "git_sync_time", "last_git_sync_at", "last_git_sync_status",
        "last_git_sync_error", "git_sync_auto_compile",
    }
    assert expected.issubset(cols)


def test_git_auth_token_not_in_response_model():
    from app.projects.router import ProjectResponse
    fields = set(ProjectResponse.model_fields.keys())
    assert "git_auth_token" not in fields
    assert "git_auth_configured" in fields


def test_source_repository_response_hides_auth_token():
    from app.projects.router import SourceRepositoryResponse
    fields = set(SourceRepositoryResponse.model_fields.keys())
    assert "auth_token" not in fields
    assert "auth_configured" in fields


def test_update_request_has_git_fields():
    from app.projects.router import UpdateProjectRequest
    fields = set(UpdateProjectRequest.model_fields.keys())
    expected = {
        "git_repo_url", "git_branch", "git_username", "git_auth_token",
        "clear_git_auth_token", "git_author_name", "git_author_email",
        "git_sync_enabled", "git_sync_time", "git_sync_auto_compile",
    }
    assert expected.issubset(fields)


def test_git_test_request_has_override_fields():
    from app.projects.router import TestGitConnectionRequest
    fields = set(TestGitConnectionRequest.model_fields.keys())
    expected = {
        "git_repo_url", "git_branch", "git_username", "git_auth_token",
    }
    assert expected.issubset(fields)


def test_legacy_git_routes_stay_registered():
    from app.projects.router import router

    paths = {
        (method, route.path)
        for route in router.routes
        for method in getattr(route, "methods", set())
    }

    assert ("POST", "/api/projects/{project_id}/git/test") in paths
    assert ("POST", "/api/projects/{project_id}/git/sync") in paths


def test_project_for_git_test_prefers_request_overrides():
    from types import SimpleNamespace
    from app.projects.router import TestGitConnectionRequest, _project_for_git_test

    project = SimpleNamespace(
        git_repo_url="https://saved.example/repo.git",
        git_branch="main",
        git_username="saved-user",
        git_auth_token="saved-token",
    )
    body = TestGitConnectionRequest(
        git_repo_url="https://draft.example/repo.git",
        git_branch="dev",
        git_username="draft-user",
        git_auth_token="draft-token",
    )

    effective = _project_for_git_test(project, body)

    assert effective.git_repo_url == "https://draft.example/repo.git"
    assert effective.git_branch == "dev"
    assert effective.git_username == "draft-user"
    assert effective.git_auth_token == "draft-token"


def test_sync_lock_isolation():
    from app.projects.git_sync import _get_sync_lock
    lock_a = _get_sync_lock("project-a")
    lock_b = _get_sync_lock("project-b")
    assert lock_a is not lock_b
    assert _get_sync_lock("project-a") is lock_a


def test_legacy_git_sync_checks_default_source_repository_lock(monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.projects import router as projects_router

    async def fake_check_membership(db, project_id, user):
        return None

    async def fake_get_project_or_404(db, project_id):
        return SimpleNamespace(
            id=project_id,
            git_repo_url="https://git.example.com/org/docs.git",
            git_branch="main",
            git_username="",
            git_auth_token="secret",
        )

    async def fake_sync_default_source_repository_from_project(db, project):
        return SimpleNamespace(
            id="default-repo",
            project_id=project.id,
            key="default",
            repo_url=project.git_repo_url,
            auth_token=project.git_auth_token,
        )

    class Lock:
        def __init__(self, locked):
            self._locked = locked

        def locked(self):
            return self._locked

    seen_lock_keys = []

    def fake_get_sync_lock(key):
        seen_lock_keys.append(key)
        return Lock(key == "project-1:default-repo")

    monkeypatch.setattr(projects_router.service, "check_membership", fake_check_membership)
    monkeypatch.setattr(projects_router.service, "get_project_or_404", fake_get_project_or_404)
    monkeypatch.setattr(
        projects_router.source_repo_service,
        "sync_default_source_repository_from_project",
        fake_sync_default_source_repository_from_project,
    )

    import app.projects.git_sync as git_sync

    monkeypatch.setattr(git_sync, "_get_sync_lock", fake_get_sync_lock)

    async def run():
        try:
            await projects_router.trigger_git_sync("project-1", user=SimpleNamespace(id=1), db=object())
        except HTTPException as exc:
            assert exc.status_code == 409
        else:
            raise AssertionError("legacy git sync should reject when default source repo sync is locked")

    import asyncio

    asyncio.run(run())
    assert seen_lock_keys == ["project-1:default-repo"]


def test_legacy_git_sync_requires_default_source_repository_token(monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.projects import router as projects_router
    import asyncio

    async def fake_check_membership(db, project_id, user):
        return None

    async def fake_get_project_or_404(db, project_id):
        return SimpleNamespace(
            id=project_id,
            git_repo_url="https://git.example.com/org/docs.git",
            git_auth_token="",
        )

    async def fake_sync_default_source_repository_from_project(db, project):
        return SimpleNamespace(
            id="default-repo",
            project_id=project.id,
            key="default",
            repo_url="https://git.example.com/org/docs.git",
            auth_token="",
        )

    monkeypatch.setattr(projects_router.service, "check_membership", fake_check_membership)
    monkeypatch.setattr(projects_router.service, "get_project_or_404", fake_get_project_or_404)
    monkeypatch.setattr(
        projects_router.source_repo_service,
        "sync_default_source_repository_from_project",
        fake_sync_default_source_repository_from_project,
    )
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())

    async def run():
        try:
            await projects_router.trigger_git_sync("project-1", user=SimpleNamespace(id=1), db=object())
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("legacy git sync should reject missing default source repo token")

    asyncio.run(run())


def test_git_publish_requires_token(monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.projects import router as projects_router
    import asyncio

    async def fake_check_membership(db, project_id, user):
        return None

    async def fake_get_project_or_404(db, project_id):
        return SimpleNamespace(
            id=project_id,
            publish_repo_url="https://git.example.com/org/publish.git",
            publish_auth_token="",
        )

    monkeypatch.setattr(projects_router.service, "check_membership", fake_check_membership)
    monkeypatch.setattr(projects_router.service, "get_project_or_404", fake_get_project_or_404)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())

    async def run():
        try:
            await projects_router.trigger_git_publish("project-1", user=SimpleNamespace(id=1), db=object())
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("git publish should reject missing publish token")

    asyncio.run(run())
