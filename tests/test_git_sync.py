"""Unit tests for git sync service."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
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

        def start(self):
            self.running = True

        def shutdown(self, wait=False):
            self.running = False

        def remove_all_jobs(self):
            pass

        def add_job(self, *args, **kwargs):
            pass

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

from app.projects import git_sync
from app.projects.git_sync import _ensure_repo, _inject_auth


def test_inject_auth_with_token():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "my_token", username="bot")
    assert "bot:my_token@git.example.com" in result
    assert result.startswith("https://")


def test_inject_auth_without_token():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "")
    assert result == url


def test_inject_auth_special_chars():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "p@ss/w0rd!", username="us er")
    assert "us%20er" in result
    assert "p%40ss%2Fw0rd%21" in result


def test_inject_auth_with_port():
    url = "https://git.example.com:8443/repo.git"
    result = _inject_auth(url, "tok", username="u")
    assert ":8443" in result
    assert "u:tok@" in result


def test_inject_auth_non_https_unchanged():
    url = "git@github.com:org/repo.git"
    result = _inject_auth(url, "token")
    assert result == url


def test_inject_auth_default_username():
    url = "https://git.example.com/repo.git"
    result = _inject_auth(url, "token123")
    assert "oauth2:token123@" in result


def test_ensure_repo_empty_remote_creates_target_branch(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

    project = SimpleNamespace(
        disk_path=str(tmp_path / "worktree"),
        git_repo_url=remote.as_uri(),
        git_auth_token="",
        git_username="",
        git_branch="main",
    )

    _ensure_repo(project)

    head = (Path(project.disk_path) / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == "ref: refs/heads/main"


def test_ensure_repo_remote_default_branch_differs_from_target_branch(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True, text=True)
    (seed / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(seed), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "-C", str(seed), "branch", "-M", "master"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "master"], check=True, capture_output=True, text=True)

    project = SimpleNamespace(
        disk_path=str(tmp_path / "worktree"),
        git_repo_url=remote.as_uri(),
        git_auth_token="",
        git_username="",
        git_branch="main",
    )

    _ensure_repo(project)

    head = (Path(project.disk_path) / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == "ref: refs/heads/main"


def test_ensure_repo_raises_when_fallback_checkout_leaves_invalid_head(tmp_path, monkeypatch):
    project = SimpleNamespace(
        disk_path=str(tmp_path / "worktree"),
        git_repo_url="https://git.example.com/org/repo.git",
        git_auth_token="",
        git_username="",
        git_branch="main",
    )

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run_git(args, *, cwd=None, check=True):
        if args[:2] == ["clone", "--branch"]:
            return Result(returncode=1, stderr="Remote branch main not found")
        if args[:1] == ["clone"]:
            root = Path(args[-1])
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")
            return Result(returncode=0)
        if args[:2] == ["checkout", "main"]:
            return Result(returncode=1, stderr="pathspec 'main' did not match")
        if args[:3] == ["checkout", "-B", "main"]:
            return Result(returncode=1, stderr="fatal: not a valid branch name")
        if args[:3] == ["symbolic-ref", "HEAD", "refs/heads/main"]:
            return Result(returncode=1, stderr="fatal: ref refs/heads/main is not valid")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(git_sync, "_run_git", fake_run_git)

    try:
        _ensure_repo(project)
    except RuntimeError as exc:
        assert "main" in str(exc)
    else:
        raise AssertionError("_ensure_repo should reject invalid HEAD state")
