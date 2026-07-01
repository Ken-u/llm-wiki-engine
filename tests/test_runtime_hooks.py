from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.runtime.config import load_runtime_config
from app.runtime.hooks import run_startup_hooks

SYNC_SCRIPT = Path("packaging/runtime/hooks/sync-repositories.py").resolve()


def _git(args: list[str], cwd=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _run_sync(config: Path) -> list[dict]:
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), str(config)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_sync_repositories_hook_script_clones_and_fast_forwards(tmp_path):
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
  repositories:
    - name: knowledge
      url: {source.as_uri()}
      branch: main
      path: ./data/knowledge
""",
        encoding="utf-8",
    )

    results = _run_sync(config)

    assert results[0]["status"] == "ok"
    assert (target / "README.md").read_text(encoding="utf-8") == "v1\n"

    (source / "README.md").write_text("v2\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=source)
    _git(["commit", "-m", "update"], cwd=source)

    results = _run_sync(config)

    assert results[0]["status"] == "ok"
    assert (target / "README.md").read_text(encoding="utf-8") == "v2\n"


def test_sync_repositories_hook_resolves_relative_paths(tmp_path):
    config = tmp_path / "nested" / "runtime-config.yaml"
    config.parent.mkdir()
    config.write_text(
        """
hooks:
  repositories:
    - name: knowledge
      url: https://example.com/repo.git
      path: ./data/knowledge
      enabled: false
""",
        encoding="utf-8",
    )

    results = _run_sync(config)

    assert results[0]["status"] == "skipped"
    assert results[0]["path"] == str((config.parent / "data" / "knowledge").resolve())


def test_runtime_hook_scripts_receive_runtime_config_env(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    output = tmp_path / "env.txt"
    config.write_text(
        f"""
hooks:
  enabled: true
  scripts:
    - name: env
      command:
        linux:
          - {sys.executable}
          - -c
          - "import os, pathlib; pathlib.Path(r'{output}').write_text(os.environ['RUNTIME_CONFIG'], encoding='utf-8')"
""",
        encoding="utf-8",
    )

    settings = load_runtime_config(config, create=False)
    results = asyncio.run(run_startup_hooks(settings))

    assert results[0].status == "ok"
    assert output.read_text(encoding="utf-8") == str(config.resolve())


def test_runtime_hook_scripts_receive_runtime_app_dir_env(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    output = tmp_path / "app_dir.txt"
    config.write_text(
        f"""
hooks:
  enabled: true
  scripts:
    - name: env
      command:
        linux:
          - {sys.executable}
          - -c
          - "import os, pathlib; pathlib.Path(r'{output}').write_text(os.environ['RUNTIME_APP_DIR'], encoding='utf-8')"
""",
        encoding="utf-8",
    )

    settings = load_runtime_config(config, create=False)
    results = asyncio.run(run_startup_hooks(settings))

    assert results[0].status == "ok"
    assert output.read_text(encoding="utf-8") == str(Path.cwd().resolve())
