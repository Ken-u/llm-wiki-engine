"""Git sync service for project-level repository binding and scheduled sync."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from sqlalchemy import select, update

from app.database import async_session
from app.documents.parser import parse_document
from app.ingest.cache import check_cache
from app.ingest.files import provider_source_root
from app.ingest.models import IngestJob
from app.ingest.queue import ingest_queue
from app.projects.models import Project, ProjectSourceRepository
from app.projects.source_repositories import (
    DEFAULT_SOURCE_REPO_KEY,
    list_source_repositories,
    source_repo_checkout_root,
    sync_default_source_repository_from_project,
)

logger = logging.getLogger(__name__)

_sync_locks: dict[str, asyncio.Lock] = {}


def _get_sync_lock(project_id: str) -> asyncio.Lock:
    if project_id not in _sync_locks:
        _sync_locks[project_id] = asyncio.Lock()
    return _sync_locks[project_id]


def _inject_auth(repo_url: str, token: str, *, username: str = "") -> str:
    """Inject auth credentials into HTTPS repo URL."""
    if not token:
        return repo_url
    parsed = urlparse(repo_url)
    if parsed.scheme not in ("http", "https"):
        return repo_url
    user = username or "oauth2"
    netloc = f"{quote(user, safe='')}:{quote(token, safe='')}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _run_git(
    args: list[str],
    *,
    cwd: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command synchronously."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    logger.debug("git %s (cwd=%s)", " ".join(args[:4]), cwd)
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        msg = f"git {' '.join(args)} failed (exit {result.returncode})"
        if details:
            msg = f"{msg}: {details}"
        raise RuntimeError(msg)
    return result


def _head_ref(root: Path) -> str:
    head_path = root / ".git" / "HEAD"
    if not head_path.exists():
        return ""
    return head_path.read_text(encoding="utf-8").strip()


def _ensure_branch_checkout(root: Path, branch: str) -> None:
    target_ref = f"refs/heads/{branch}"
    if _head_ref(root) == f"ref: {target_ref}":
        return

    checkout = _run_git(["checkout", branch], cwd=str(root), check=False)
    if checkout.returncode == 0 and _head_ref(root) == f"ref: {target_ref}":
        return

    create = _run_git(["checkout", "-B", branch], cwd=str(root), check=False)
    if create.returncode == 0 and _head_ref(root) == f"ref: {target_ref}":
        return

    symbolic = _run_git(["symbolic-ref", "HEAD", target_ref], cwd=str(root), check=False)
    if symbolic.returncode == 0 and _head_ref(root) == f"ref: {target_ref}":
        return

    detail = (
        (create.stderr or create.stdout).strip()
        or (checkout.stderr or checkout.stdout).strip()
        or (symbolic.stderr or symbolic.stdout).strip()
        or f"HEAD is {_head_ref(root) or 'missing'}"
    )
    raise RuntimeError(f"failed to checkout branch '{branch}': {detail}")


def _ensure_repo(project: Project, source_repo: ProjectSourceRepository | None = None) -> None:
    """Clone or update the git working copy for a project or source repo."""
    if source_repo is None:
        root = provider_source_root(project.disk_path)
        repo_url = project.git_repo_url
        auth_token = project.git_auth_token
        username = project.git_username
        branch = project.git_branch or "main"
    else:
        root = source_repo_checkout_root(project, source_repo)
        repo_url = source_repo.repo_url
        auth_token = source_repo.auth_token
        username = source_repo.username
        branch = source_repo.branch or "main"

    authed_url = _inject_auth(repo_url, auth_token, username=username)

    if not (root / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        r = _run_git(
            ["clone", "--branch", branch, "--single-branch", authed_url, str(root)],
            check=False,
        )
        if r.returncode != 0:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            _run_git(["clone", authed_url, str(root)])
        _ensure_branch_checkout(root, branch)
        logger.info("Cloned provider %s (%s) -> %s", repo_url, branch, root)
    else:
        _run_git(["remote", "set-url", "origin", authed_url], cwd=str(root))
        fetch_result = _run_git(["fetch", "origin", branch], cwd=str(root), check=False)
        if fetch_result.returncode == 0:
            checkout_result = _run_git(["checkout", branch], cwd=str(root), check=False)
            if checkout_result.returncode != 0:
                _run_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=str(root))
            _run_git(["reset", "--hard", f"origin/{branch}"], cwd=str(root))
        else:
            _ensure_branch_checkout(root, branch)
        _ensure_branch_checkout(root, branch)
        logger.info("Updated provider %s to origin/%s", root, branch)


def _commit_and_push(project: Project) -> bool:
    """Stage all, commit if changes exist, push. Returns True if pushed."""
    root = str(Path(project.disk_path))
    branch = project.git_branch or "main"

    _run_git(["add", "-A"], cwd=root)

    status = _run_git(["status", "--porcelain", "--", ".", ":!.llm-wiki/source-repo"], cwd=root)
    if not status.stdout.strip():
        logger.info("No changes to commit for project %s", project.id)
        return False

    commit_args = []
    if project.git_author_name:
        commit_args.extend(["-c", f"user.name={project.git_author_name}"])
    if project.git_author_email:
        commit_args.extend(["-c", f"user.email={project.git_author_email}"])
    commit_args.extend(["commit", "-m", f"sync: auto-compile {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"])
    _run_git(commit_args, cwd=root)

    authed_url = _inject_auth(project.git_repo_url, project.git_auth_token, username=project.git_username)
    _run_git(["remote", "set-url", "origin", authed_url], cwd=root)
    _run_git(["push", "origin", branch], cwd=root)
    logger.info("Pushed to origin/%s from %s", branch, root)
    return True


def _publish_remote_name() -> str:
    return "llmwiki-publish"


_PUBLISH_GITIGNORE_LINES = [
    ".llm-wiki/source-repo/",
    ".llm-wiki/source-repos/",
    ".llm-wiki/chats/",
    ".llm-wiki/checkpoints/",
]

_PUBLISH_IGNORED_PATHS = [
    ".llm-wiki/source-repo",
    ".llm-wiki/source-repos",
    ".llm-wiki/chats",
    ".llm-wiki/checkpoints",
]


def _ensure_publish_gitignore(root: Path) -> None:
    """Keep runtime-only project data out of the publish repository."""
    gitignore = root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    existing_rules = {line.strip() for line in existing}
    missing = [line for line in _PUBLISH_GITIGNORE_LINES if line not in existing_rules]
    if not missing:
        return

    next_lines = list(existing)
    if next_lines and next_lines[-1].strip():
        next_lines.append("")
    next_lines.append("# llm-wiki runtime data")
    next_lines.extend(missing)
    gitignore.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def _commit_and_publish(project: Project) -> bool:
    """Commit current worktree and push it to a separate publish remote."""
    publish_url = getattr(project, "publish_repo_url", "") or ""
    if not publish_url:
        raise RuntimeError("未配置发布仓库 URL")
    token = getattr(project, "publish_auth_token", "") or ""
    if not token:
        raise RuntimeError("未配置发布仓库访问凭证")

    root = str(Path(project.disk_path))
    branch = getattr(project, "publish_branch", "") or "main"
    remote = _publish_remote_name()

    if not (Path(root) / ".git").exists():
        _run_git(["init"], cwd=root)
        _run_git(["checkout", "-B", branch], cwd=root)

    _ensure_publish_gitignore(Path(root))
    _run_git(["rm", "-r", "--cached", "--ignore-unmatch", "--", *_PUBLISH_IGNORED_PATHS], cwd=root)
    _run_git(["add", "-A"], cwd=root)
    status = _run_git(["status", "--porcelain"], cwd=root)
    if status.stdout.strip():
        commit_args = []
        author_name = getattr(project, "publish_author_name", "") or project.git_author_name
        author_email = getattr(project, "publish_author_email", "") or project.git_author_email
        if author_name:
            commit_args.extend(["-c", f"user.name={author_name}"])
        if author_email:
            commit_args.extend(["-c", f"user.email={author_email}"])
        commit_args.extend(["commit", "-m", f"publish: mirror {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"])
        _run_git(commit_args, cwd=root)

    authed_url = _inject_auth(publish_url, token, username=getattr(project, "publish_username", "") or "")
    existing = _run_git(["remote"], cwd=root).stdout.splitlines()
    if remote in existing:
        _run_git(["remote", "set-url", remote, authed_url], cwd=root)
    else:
        _run_git(["remote", "add", remote, authed_url], cwd=root)
    _run_git(["push", remote, f"HEAD:{branch}"], cwd=root)
    logger.info("Published %s to %s/%s", root, remote, branch)
    return True


def _should_auto_enqueue_compile(project: Project, source_repo: ProjectSourceRepository | None = None) -> bool:
    auto_compile = (
        source_repo.auto_compile
        if source_repo is not None
        else project.git_sync_auto_compile
    )
    return bool(auto_compile) and not bool(project.ingest_paused)


# --- Public API ---


async def test_project_git_connection(project: Project) -> dict:
    """Test git connectivity. Returns {"ok": bool, "error": str}."""
    if not project.git_repo_url:
        return {"ok": False, "error": "未配置仓库 URL"}
    if not project.git_auth_token:
        return {"ok": False, "error": "未配置访问凭证"}

    source_repo = ProjectSourceRepository(
        id="legacy-test",
        project_id=getattr(project, "id", ""),
        key=DEFAULT_SOURCE_REPO_KEY,
        name=DEFAULT_SOURCE_REPO_KEY,
        repo_url=project.git_repo_url,
        branch=project.git_branch or "main",
        username=project.git_username,
        auth_token=project.git_auth_token,
    )
    return await test_source_repository_git_connection(project, source_repo)


async def test_source_repository_git_connection(project: Project, source_repo: ProjectSourceRepository) -> dict:
    """Test source repository git connectivity."""
    if not source_repo.repo_url:
        return {"ok": False, "error": "未配置仓库 URL"}
    if not source_repo.auth_token:
        return {"ok": False, "error": "未配置访问凭证"}

    authed_url = _inject_auth(source_repo.repo_url, source_repo.auth_token, username=source_repo.username)
    branch = source_repo.branch or "main"

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: _run_git(["ls-remote", "--heads", authed_url, branch]),
        )
        return {"ok": True, "error": ""}
    except RuntimeError as exc:
        error_msg = str(exc)
        if "Authentication" in error_msg or "403" in error_msg or "401" in error_msg:
            return {"ok": False, "error": "认证失败，请检查用户名和密码/Token"}
        return {"ok": False, "error": f"连接失败: {error_msg[:200]}"}


async def test_project_publish_connection(project: Project) -> dict:
    """Test publish repository connectivity without touching origin."""
    repo_url = getattr(project, "publish_repo_url", "") or ""
    token = getattr(project, "publish_auth_token", "") or ""
    branch = getattr(project, "publish_branch", "") or "main"
    username = getattr(project, "publish_username", "") or ""
    if not repo_url:
        return {"ok": False, "error": "未配置发布仓库 URL"}
    if not token:
        return {"ok": False, "error": "未配置发布仓库访问凭证"}

    authed_url = _inject_auth(repo_url, token, username=username)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_git(["ls-remote", "--heads", authed_url, branch], check=False),
        )
        if result.returncode in (0, 2):
            return {"ok": True, "error": ""}
        details = (result.stderr or result.stdout or "").strip()
        return {"ok": False, "error": f"连接失败: {details[:200]}"}
    except RuntimeError as exc:
        return {"ok": False, "error": f"连接失败: {str(exc)[:200]}"}


async def publish_project_to_git(project_id: str, *, triggered_by: int = 0) -> None:
    """Push the project worktree to a separate publish remote."""
    lock = _get_sync_lock(f"{project_id}:publish")
    if lock.locked():
        raise RuntimeError("该项目已有发布正在执行")

    async with lock:
        async with async_session() as db:
            project = (await db.execute(
                select(Project).where(Project.id == project_id)
            )).scalar_one_or_none()
            if not project:
                raise RuntimeError(f"Project {project_id} not found")
            if not project.publish_repo_url or not project.publish_auth_token:
                raise RuntimeError("发布仓库配置不完整")
            repo_ids = list(
                (
                    await db.execute(
                        select(ProjectSourceRepository.id).where(
                            ProjectSourceRepository.project_id == project_id
                        )
                    )
                ).scalars().all()
            )
            for repo_id in repo_ids:
                if _get_sync_lock(f"{project_id}:{repo_id}").locked():
                    raise RuntimeError("该项目已有源仓库同步正在执行，无法发布")

            project.last_publish_status = "syncing"
            project.last_publish_error = ""
            await db.commit()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: _commit_and_publish(project))
            async with async_session() as db:
                await db.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        last_publish_at=datetime.now(timezone.utc),
                        last_publish_status="success",
                        last_publish_error="",
                    )
                )
                await db.commit()
            logger.info("Git publish completed for project %s by user %s", project_id, triggered_by)
        except Exception as exc:
            logger.exception("Git publish failed for project %s", project_id)
            async with async_session() as db:
                await db.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        last_publish_status="failed",
                        last_publish_error=str(exc)[:500],
                    )
                )
                await db.commit()
            raise


def _list_source_files_under(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    from app.ingest.files import is_project_source_file

    return sorted(
        [
            path for path in source_root.rglob("*")
            if is_project_source_file(path, source_root)
        ],
        key=lambda path: path.resolve().relative_to(source_root.resolve()).as_posix().lower(),
    )


async def sync_source_repository(
    project_id: str,
    repo_id: str,
    *,
    triggered_by: int = 0,
    source: str = "manual",
) -> dict:
    """Execute a git sync cycle for one source repository."""
    lock = _get_sync_lock(f"{project_id}:{repo_id}")
    if lock.locked():
        raise RuntimeError("该项目已有同步正在执行")
    publish_lock = _get_sync_lock(f"{project_id}:publish")
    if publish_lock.locked():
        raise RuntimeError("该项目已有发布正在执行，无法同步源仓库")

    async with lock:
        async with async_session() as db:
            project = (await db.execute(
                select(Project).where(Project.id == project_id)
            )).scalar_one_or_none()
            if not project:
                raise RuntimeError(f"Project {project_id} not found")

            source_repo = (await db.execute(
                select(ProjectSourceRepository).where(
                    ProjectSourceRepository.project_id == project_id,
                    ProjectSourceRepository.id == repo_id,
                )
            )).scalar_one_or_none()
            if not source_repo:
                raise RuntimeError(f"Source repository {repo_id} not found")

            if not source_repo.repo_url or not source_repo.auth_token:
                error = "Git 配置不完整"
                source_repo.last_sync_status = "failed"
                source_repo.last_sync_error = error
                if source_repo.key == DEFAULT_SOURCE_REPO_KEY:
                    project.last_git_sync_status = "failed"
                    project.last_git_sync_error = error
                await db.commit()
                raise RuntimeError("Git 配置不完整")

            source_repo.last_sync_status = "syncing"
            source_repo.last_sync_error = ""
            if source_repo.key == DEFAULT_SOURCE_REPO_KEY:
                project.last_git_sync_status = "syncing"
                project.last_git_sync_error = ""
            await db.commit()

        try:
            loop = asyncio.get_event_loop()

            # Pull
            await loop.run_in_executor(None, lambda: _ensure_repo(project, source_repo))

            source_root = source_repo_checkout_root(project, source_repo)
            source_files = _list_source_files_under(source_root)

            if not source_files:
                logger.info("No source files found for project %s", project_id)

            # Pre-filter: only enqueue files whose content actually changed
            changed_files = []
            for src_file in source_files:
                try:
                    content = parse_document(src_file)
                    if not content.strip():
                        continue
                    source_identity = src_file.resolve().relative_to(source_root.resolve()).as_posix()
                    if not await check_cache(project.disk_path, source_identity, content):
                        changed_files.append(src_file)
                except Exception:
                    changed_files.append(src_file)

            if not changed_files:
                logger.info("All source files unchanged for project %s, skipping compile", project_id)

            # Enqueue ingest only for changed files when this source repository
            # enables auto compile and project-level compilation is not paused.
            job_ids = []
            if project.project_type == "case_library":
                if changed_files:
                    if project.case_index_auto_rebuild:
                        logger.info("Auto-rebuilding case index for case_library project %s", project_id)
                        from app.case_index.builder import rebuild_case_index
                        await rebuild_case_index(project.disk_path)
                    else:
                        from app.case_index.builder import mark_case_index_stale
                        mark_case_index_stale(project.disk_path)
                        logger.info("Marked case index as stale for project %s", project_id)
            elif not _should_auto_enqueue_compile(project, source_repo):
                logger.info("Auto compile disabled for project %s, skipping automatic enqueue", project_id)
            else:
                for src_file in changed_files:
                    job_id = await ingest_queue.enqueue(
                        project_id=project_id,
                        project_dir=project.disk_path,
                        source_path=str(src_file),
                        user_id=triggered_by,
                    )
                    job_ids.append(job_id)

            # Wait for all ingest jobs to complete
            if job_ids:
                await _wait_for_jobs(job_ids)

            # Check for failures
            failed_jobs = await _get_failed_jobs(job_ids)
            if failed_jobs:
                raise RuntimeError(f"{len(failed_jobs)} 个编译任务失败")

            async with async_session() as db:
                now = datetime.now(timezone.utc)
                await db.execute(
                    update(ProjectSourceRepository)
                    .where(ProjectSourceRepository.id == repo_id)
                    .values(
                        last_sync_at=now,
                        last_sync_status="success",
                        last_sync_error="",
                    )
                )
                if source_repo.key == DEFAULT_SOURCE_REPO_KEY:
                    await db.execute(
                        update(Project)
                        .where(Project.id == project_id)
                        .values(
                            last_git_sync_at=now,
                            last_git_sync_status="success",
                            last_git_sync_error="",
                        )
                    )
                await db.commit()

            logger.info("Git sync completed for project %s repo %s (source=%s)", project_id, repo_id, source)
            return {"repo_id": repo_id, "status": "success", "error": ""}

        except Exception as exc:
            logger.exception("Git sync failed for project %s repo %s", project_id, repo_id)
            async with async_session() as db:
                await db.execute(
                    update(ProjectSourceRepository)
                    .where(ProjectSourceRepository.id == repo_id)
                    .values(
                        last_sync_status="failed",
                        last_sync_error=str(exc)[:500],
                    )
                )
                if source_repo.key == DEFAULT_SOURCE_REPO_KEY:
                    await db.execute(
                        update(Project)
                        .where(Project.id == project_id)
                        .values(
                            last_git_sync_status="failed",
                            last_git_sync_error=str(exc)[:500],
                        )
                    )
                await db.commit()
            raise


async def sync_all_source_repositories(
    project_id: str,
    *,
    triggered_by: int = 0,
    source: str = "manual",
) -> list[dict]:
    """Sync all source repositories for a project, recording per-repo results."""
    async with async_session() as db:
        project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
        if not project:
            raise RuntimeError(f"Project {project_id} not found")
        repos = await list_source_repositories(db, project)

    results = []
    for repo in repos:
        try:
            results.append(await sync_source_repository(project_id, repo.id, triggered_by=triggered_by, source=source))
        except Exception as exc:
            results.append({"repo_id": repo.id, "status": "failed", "error": str(exc)[:500]})
    return results


async def sync_project_from_git(
    project_id: str,
    *,
    triggered_by: int = 0,
    source: str = "manual",
) -> None:
    """Compatibility wrapper syncing the default source repository."""
    async with async_session() as db:
        project = (await db.execute(
            select(Project).where(Project.id == project_id)
        )).scalar_one_or_none()
        if not project:
            raise RuntimeError(f"Project {project_id} not found")
        if not project.git_repo_url or not project.git_auth_token:
            raise RuntimeError("Git 配置不完整")
        source_repo = await sync_default_source_repository_from_project(db, project)

    await sync_source_repository(project_id, source_repo.id, triggered_by=triggered_by, source=source)


async def _wait_for_jobs(job_ids: list[str], timeout: float = 3600) -> None:
    """Poll until all ingest jobs reach a terminal state."""
    import time
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        async with async_session() as db:
            stmt = select(IngestJob).where(IngestJob.id.in_(job_ids))
            jobs = list((await db.execute(stmt)).scalars().all())
            if all(j.status in ("done", "failed") for j in jobs):
                return
        await asyncio.sleep(2)
    raise RuntimeError("同步超时：编译任务未在规定时间内完成")


async def _get_failed_jobs(job_ids: list[str]) -> list[str]:
    """Return IDs of failed jobs."""
    if not job_ids:
        return []
    async with async_session() as db:
        stmt = select(IngestJob.id).where(
            IngestJob.id.in_(job_ids),
            IngestJob.status == "failed",
        )
        return list((await db.execute(stmt)).scalars().all())


# --- Scheduler ---

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402

_scheduler = AsyncIOScheduler()


async def _scheduled_sync_job(project_id: str, repo_id: str | None = None) -> None:
    """Wrapper for scheduled execution."""
    try:
        if repo_id:
            await sync_source_repository(project_id, repo_id, triggered_by=0, source="schedule")
        else:
            await sync_project_from_git(project_id, triggered_by=0, source="schedule")
    except Exception:
        logger.exception("Scheduled sync failed for project %s repo %s", project_id, repo_id or "default")


async def register_sync_jobs() -> None:
    """Load all source-repo-sync-enabled repositories and register cron jobs."""
    _scheduler.remove_all_jobs()
    async with async_session() as db:
        stmt = (
            select(Project, ProjectSourceRepository)
            .join(ProjectSourceRepository, ProjectSourceRepository.project_id == Project.id)
            .where(ProjectSourceRepository.sync_enabled == True)  # noqa: E712
        )
        rows = list((await db.execute(stmt)).all())

    for proj, repo in rows:
        trigger_time = repo.sync_time or "02:00"
        parts = trigger_time.split(":")
        hour = int(parts[0]) if len(parts) > 0 else 2
        minute = int(parts[1]) if len(parts) > 1 else 0
        trigger = CronTrigger(hour=hour, minute=minute)
        _scheduler.add_job(
            _scheduled_sync_job,
            trigger=trigger,
            args=[proj.id, repo.id],
            id=f"git_sync_{proj.id}_{repo.id}",
            replace_existing=True,
        )
        logger.info(
            "Registered git sync for project '%s' source repo '%s' at %02d:%02d",
            proj.name,
            repo.key,
            hour,
            minute,
        )


def start_sync_scheduler() -> None:
    """Start the APScheduler and register all sync jobs."""
    if not _scheduler.running:
        _scheduler.start()
        asyncio.get_event_loop().create_task(register_sync_jobs())


def stop_sync_scheduler() -> None:
    """Stop the scheduler."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
