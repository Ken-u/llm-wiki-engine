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
        "last_git_sync_error",
    }
    assert expected.issubset(cols)


def test_git_auth_token_not_in_response_model():
    from app.projects.router import ProjectResponse
    fields = set(ProjectResponse.model_fields.keys())
    assert "git_auth_token" not in fields
    assert "git_auth_configured" in fields


def test_update_request_has_git_fields():
    from app.projects.router import UpdateProjectRequest
    fields = set(UpdateProjectRequest.model_fields.keys())
    expected = {
        "git_repo_url", "git_branch", "git_username", "git_auth_token",
        "clear_git_auth_token", "git_author_name", "git_author_email",
        "git_sync_enabled", "git_sync_time",
    }
    assert expected.issubset(fields)


def test_git_test_request_has_override_fields():
    from app.projects.router import TestGitConnectionRequest
    fields = set(TestGitConnectionRequest.model_fields.keys())
    expected = {
        "git_repo_url", "git_branch", "git_username", "git_auth_token",
    }
    assert expected.issubset(fields)


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
