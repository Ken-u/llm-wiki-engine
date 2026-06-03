"""Tests for git-related project API behavior."""

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


def test_sync_lock_isolation():
    from app.projects.git_sync import _get_sync_lock
    lock_a = _get_sync_lock("project-a")
    lock_b = _get_sync_lock("project-b")
    assert lock_a is not lock_b
    assert _get_sync_lock("project-a") is lock_a
