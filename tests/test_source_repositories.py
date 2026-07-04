from types import SimpleNamespace

import pytest

from app.projects.source_repositories import (
    infer_repo_name,
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
