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
from app.projects.git_sync import _commit_and_publish, _ensure_repo, _inject_auth
from app.ingest.files import provider_source_root
from app.projects.source_repositories import source_repo_checkout_root


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


def test_should_auto_enqueue_compile_requires_project_flag():
    from app.projects.git_sync import _should_auto_enqueue_compile

    assert _should_auto_enqueue_compile(SimpleNamespace(git_sync_auto_compile=False, ingest_paused=False)) is False
    assert _should_auto_enqueue_compile(SimpleNamespace(git_sync_auto_compile=True, ingest_paused=True)) is False
    assert _should_auto_enqueue_compile(SimpleNamespace(git_sync_auto_compile=True, ingest_paused=False)) is True


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

    head = (provider_source_root(project.disk_path) / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == "ref: refs/heads/main"


def test_ensure_repo_source_repository_uses_multi_source_checkout_root(tmp_path, monkeypatch):
    project = SimpleNamespace(
        disk_path=str(tmp_path / "worktree"),
    )
    source_repo = SimpleNamespace(
        key="frontend-docs",
        repo_url="https://git.example.com/org/frontend-docs.git",
        auth_token="secret",
        username="bot",
        branch="main",
    )
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run_git(args, *, cwd=None, check=True):
        calls.append((args, cwd))
        if args[:1] == ["clone"]:
            root = Path(args[-1])
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        return Result()

    monkeypatch.setattr(git_sync, "_run_git", fake_run_git)

    _ensure_repo(project, source_repo)

    clone_target = Path(calls[0][0][-1])
    assert clone_target == source_repo_checkout_root(project, source_repo)
    assert clone_target.as_posix().endswith(".llm-wiki/source-repos/frontend-docs")


def test_source_repository_sync_lock_uses_project_and_repo_id(monkeypatch):
    seen_keys = []

    class Lock:
        def locked(self):
            return True

    def fake_get_sync_lock(key):
        seen_keys.append(key)
        return Lock()

    monkeypatch.setattr(git_sync, "_get_sync_lock", fake_get_sync_lock)

    async def run():
        try:
            await git_sync.sync_source_repository("project-1", "repo-1", triggered_by=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("sync_source_repository should reject a locked repo sync")

    import asyncio

    asyncio.run(run())
    assert seen_keys == ["project-1:repo-1"]


def test_register_sync_jobs_uses_source_repository_job_ids(tmp_path, monkeypatch):
    async def run():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.agents.models import Agent  # noqa: F401
        from app.auth.models import User
        from app.database import Base
        from app.projects.models import Project, ProjectSourceRepository

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scheduler.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            db.add(Project(id="project-1", name="Project", slug="project", description="", created_by=1))
            db.add_all([
                ProjectSourceRepository(
                    id="repo-1",
                    project_id="project-1",
                    key="docs",
                    name="Docs",
                    repo_url="https://git.example.com/org/docs.git",
                    sync_enabled=True,
                    sync_time="04:20",
                ),
                ProjectSourceRepository(
                    id="repo-2",
                    project_id="project-1",
                    key="disabled",
                    name="Disabled",
                    repo_url="https://git.example.com/org/disabled.git",
                    sync_enabled=False,
                    sync_time="05:30",
                ),
            ])
            await db.commit()

        added_jobs = []

        class Scheduler:
            def remove_all_jobs(self):
                added_jobs.clear()

            def add_job(self, func, *, trigger, args, id, replace_existing):
                added_jobs.append((func, args, id, replace_existing))

        monkeypatch.setattr(git_sync, "async_session", Session)
        monkeypatch.setattr(git_sync, "_scheduler", Scheduler())

        await git_sync.register_sync_jobs()

        assert added_jobs == [(git_sync._scheduled_sync_job, ["project-1", "repo-1"], "git_sync_project-1_repo-1", True)]
        await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_sync_source_repository_marks_missing_config_failed(tmp_path, monkeypatch):
    async def run():
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.agents.models import Agent  # noqa: F401
        from app.auth.models import User
        from app.database import Base
        from app.projects.models import Project, ProjectSourceRepository

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'missing-config.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            db.add(Project(id="project-1", name="Project", slug="project", description="", created_by=1))
            db.add(ProjectSourceRepository(
                id="repo-1",
                project_id="project-1",
                key="docs",
                name="Docs",
                repo_url="",
                auth_token="",
            ))
            await db.commit()

        monkeypatch.setattr(git_sync, "async_session", Session)

        try:
            await git_sync.sync_source_repository("project-1", "repo-1", triggered_by=1)
        except RuntimeError as exc:
            assert "Git 配置不完整" in str(exc)
        else:
            raise AssertionError("sync_source_repository should reject missing Git config")

        async with Session() as db:
            repo = (
                await db.execute(select(ProjectSourceRepository).where(ProjectSourceRepository.id == "repo-1"))
            ).scalar_one()
            assert repo.last_sync_status == "failed"
            assert "Git 配置不完整" in repo.last_sync_error

        await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_sync_all_source_repositories_records_bad_repo_and_continues(tmp_path, monkeypatch):
    async def run():
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.agents.models import Agent  # noqa: F401
        from app.auth.models import User
        from app.database import Base
        from app.projects.models import Project, ProjectSourceRepository

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sync-all.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        project_dir = tmp_path / "project-dir"
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            db.add(Project(
                id="project-1",
                name="Project",
                slug="project",
                description="",
                created_by=1,
                git_sync_auto_compile=False,
            ))
            db.add_all([
                ProjectSourceRepository(
                    id="bad-repo",
                    project_id="project-1",
                    key="bad",
                    name="Bad",
                    repo_url="",
                    auth_token="",
                ),
                ProjectSourceRepository(
                    id="good-repo",
                    project_id="project-1",
                    key="good",
                    name="Good",
                    repo_url="https://git.example.com/org/good.git",
                    auth_token="secret",
                ),
            ])
            await db.commit()

        def fake_disk_path(self):
            return str(project_dir)

        def fake_ensure_repo(project, source_repo):
            root = source_repo_checkout_root(project, source_repo)
            root.mkdir(parents=True, exist_ok=True)
            (root / "README.md").write_text("hello\n", encoding="utf-8")

        monkeypatch.setattr(git_sync, "async_session", Session)
        monkeypatch.setattr(Project, "disk_path", property(fake_disk_path))
        monkeypatch.setattr(git_sync, "_ensure_repo", fake_ensure_repo)

        results = await git_sync.sync_all_source_repositories("project-1", triggered_by=1)

        assert [result["repo_id"] for result in results] == ["bad-repo", "good-repo"]
        assert results[0]["status"] == "failed"
        assert results[1]["status"] == "success"

        async with Session() as db:
            repos = {
                repo.id: repo
                for repo in (
                    await db.execute(select(ProjectSourceRepository).order_by(ProjectSourceRepository.id))
                ).scalars().all()
            }
            assert repos["bad-repo"].last_sync_status == "failed"
            assert "Git 配置不完整" in repos["bad-repo"].last_sync_error
            assert repos["good-repo"].last_sync_status == "success"

        await engine.dispose()

    import asyncio

    asyncio.run(run())


def test_sync_source_repository_uses_repo_auto_compile_setting(tmp_path, monkeypatch):
    async def run():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.agents.models import Agent  # noqa: F401
        from app.auth.models import User
        from app.database import Base
        from app.projects.models import Project, ProjectSourceRepository

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'repo-auto-compile.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        project_dir = tmp_path / "project-dir"
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session() as db:
            db.add(User(id=1, username="owner", password_hash="x", role="user"))
            db.add(Project(
                id="project-1",
                name="Project",
                slug="project",
                description="",
                created_by=1,
                git_sync_auto_compile=True,
                ingest_paused=False,
            ))
            db.add_all([
                ProjectSourceRepository(
                    id="manual-repo",
                    project_id="project-1",
                    key="manual",
                    name="Manual",
                    repo_url="https://git.example.com/org/manual.git",
                    auth_token="secret",
                    auto_compile=False,
                ),
                ProjectSourceRepository(
                    id="auto-repo",
                    project_id="project-1",
                    key="auto",
                    name="Auto",
                    repo_url="https://git.example.com/org/auto.git",
                    auth_token="secret",
                    auto_compile=True,
                ),
            ])
            await db.commit()

        def fake_disk_path(self):
            return str(project_dir)

        def fake_ensure_repo(project, source_repo):
            root = source_repo_checkout_root(project, source_repo)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{source_repo.key}.md").write_text(f"{source_repo.key}\n", encoding="utf-8")

        enqueued = []

        async def fake_enqueue(**kwargs):
            enqueued.append(kwargs)
            return f"job-{len(enqueued)}"

        async def fake_wait_for_jobs(job_ids):
            return None

        async def fake_get_failed_jobs(job_ids):
            return []

        async def fake_check_cache(project_dir, source_identity, content):
            return False

        monkeypatch.setattr(git_sync, "async_session", Session)
        monkeypatch.setattr(Project, "disk_path", property(fake_disk_path))
        monkeypatch.setattr(git_sync, "_ensure_repo", fake_ensure_repo)
        monkeypatch.setattr(git_sync.ingest_queue, "enqueue", fake_enqueue)
        monkeypatch.setattr(git_sync, "_wait_for_jobs", fake_wait_for_jobs)
        monkeypatch.setattr(git_sync, "_get_failed_jobs", fake_get_failed_jobs)
        monkeypatch.setattr(git_sync, "check_cache", fake_check_cache)

        await git_sync.sync_source_repository("project-1", "manual-repo", triggered_by=7)
        assert enqueued == []

        await git_sync.sync_source_repository("project-1", "auto-repo", triggered_by=7)
        assert len(enqueued) == 1
        assert enqueued[0]["project_id"] == "project-1"
        assert enqueued[0]["user_id"] == 7
        assert enqueued[0]["source_path"].endswith(".llm-wiki/source-repos/auto/auto.md")

        await engine.dispose()

    import asyncio

    asyncio.run(run())


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

    head = (provider_source_root(project.disk_path) / ".git" / "HEAD").read_text(encoding="utf-8").strip()
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


def test_commit_and_publish_uses_separate_remote_without_changing_origin(tmp_path):
    source_remote = tmp_path / "source.git"
    publish_remote = tmp_path / "publish.git"
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "init", "--bare", str(source_remote)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "init", "--bare", str(publish_remote)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "init", str(worktree)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(worktree), "remote", "add", "origin", str(source_remote)], check=True, capture_output=True, text=True)
    (worktree / "README.md").write_text("hello\n", encoding="utf-8")
    (worktree / ".llm-wiki" / "source-repo").mkdir(parents=True)
    (worktree / ".llm-wiki" / "source-repo" / "provider.md").write_text("provider\n", encoding="utf-8")
    (worktree / ".llm-wiki" / "source-repos" / "frontend").mkdir(parents=True)
    (worktree / ".llm-wiki" / "source-repos" / "frontend" / "provider.md").write_text("provider\n", encoding="utf-8")
    (worktree / ".llm-wiki" / "chats").mkdir(parents=True)
    (worktree / ".llm-wiki" / "chats" / "chat.json").write_text("{}", encoding="utf-8")
    (worktree / ".llm-wiki" / "checkpoints").mkdir(parents=True)
    (worktree / ".llm-wiki" / "checkpoints" / "job.json").write_text("{}", encoding="utf-8")
    (worktree / ".llm-wiki").mkdir(exist_ok=True)
    (worktree / ".llm-wiki" / "source-map.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(worktree), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )

    project = SimpleNamespace(
        disk_path=str(worktree),
        git_author_name="",
        git_author_email="",
        publish_repo_url=str(publish_remote),
        publish_auth_token="token",
        publish_username="",
        publish_branch="main",
        publish_author_name="Publisher",
        publish_author_email="publisher@example.com",
    )

    _commit_and_publish(project)

    origin_url = subprocess.run(
        ["git", "-C", str(worktree), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    publish_refs = subprocess.run(
        ["git", "--git-dir", str(publish_remote), "show-ref", "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    published_files = subprocess.run(
        ["git", "--git-dir", str(publish_remote), "ls-tree", "-r", "--name-only", "main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert origin_url == str(source_remote)
    assert publish_refs
    assert "README.md" in published_files
    assert ".gitignore" in published_files
    assert ".llm-wiki/source-map.json" in published_files
    assert ".llm-wiki/source-repo/provider.md" not in published_files
    assert ".llm-wiki/source-repos/frontend/provider.md" not in published_files
    assert ".llm-wiki/chats/chat.json" not in published_files
    assert ".llm-wiki/checkpoints/job.json" not in published_files
    published_gitignore = subprocess.run(
        ["git", "--git-dir", str(publish_remote), "show", "main:.gitignore"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert ".llm-wiki/source-repo/" in published_gitignore
    assert ".llm-wiki/source-repos/" in published_gitignore
    assert ".llm-wiki/chats/" in published_gitignore
    assert ".llm-wiki/checkpoints/" in published_gitignore
