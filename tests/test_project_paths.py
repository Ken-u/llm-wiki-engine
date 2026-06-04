from types import SimpleNamespace

from app.projects.models import Project
from app.projects.service import project_dir


def test_project_dir_uses_configured_projects_dir(monkeypatch):
    monkeypatch.setattr(
        "app.projects.service.get_config",
        lambda: SimpleNamespace(server=SimpleNamespace(projects_dir="/tmp/projects-root")),
    )

    assert str(project_dir("proj-123")) == "/tmp/projects-root/proj-123"


def test_project_disk_path_is_derived_from_project_id(monkeypatch):
    monkeypatch.setattr(
        "app.projects.models.get_config",
        lambda: SimpleNamespace(server=SimpleNamespace(projects_dir="/tmp/projects-root")),
    )

    project = Project(id="proj-123", name="Demo", slug="demo", description="", created_by=1, _disk_path="legacy/path")

    assert project.disk_path == "/tmp/projects-root/proj-123"
