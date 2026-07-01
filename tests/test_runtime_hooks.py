from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from app.runtime.config import load_runtime_config
from app.runtime.hooks import run_repository_sync_hooks


def _git(args: list[str], cwd=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_runtime_repository_hook_clones_and_fast_forwards(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "runtime" / "data" / "knowledge"
    config = tmp_path / "runtime" / "runtime-config.yaml"

    _git(["init", "-b", "main", str(source)])
    _git(["config", "user.email", "test@example.com"], cwd=source)
    _git(["config", "user.name", "Test"], cwd=source)
    (source / "README.md").write_text("v1\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=source)
    _git(["commit", "-m", "init"], cwd=source)

    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
hooks:
  enabled: true
  timeout_seconds: 30
  repositories:
    - name: knowledge
      url: {source.as_uri()}
      branch: main
      path: ./data/knowledge
""",
        encoding="utf-8",
    )

    settings = load_runtime_config(config, create=False)
    results = asyncio.run(run_repository_sync_hooks(settings))

    assert results[0].status == "ok"
    assert (target / "README.md").read_text(encoding="utf-8") == "v1\n"

    (source / "README.md").write_text("v2\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=source)
    _git(["commit", "-m", "update"], cwd=source)

    results = asyncio.run(run_repository_sync_hooks(settings))

    assert results[0].status == "ok"
    assert (target / "README.md").read_text(encoding="utf-8") == "v2\n"


def test_runtime_repository_path_resolves_relative_to_config(tmp_path):
    config = tmp_path / "nested" / "runtime-config.yaml"
    config.parent.mkdir()
    config.write_text(
        """
hooks:
  repositories:
    - name: knowledge
      url: https://example.com/repo.git
      path: ./data/knowledge
""",
        encoding="utf-8",
    )

    settings = load_runtime_config(config, create=False)

    assert settings.hooks.repositories[0].path == str((config.parent / "data" / "knowledge").resolve())
