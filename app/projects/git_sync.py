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
from app.ingest.models import IngestJob
from app.ingest.queue import ingest_queue
from app.projects.models import Project

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


def _ensure_repo(project: Project) -> None:
    """Clone or update the git working copy at project.disk_path."""
    root = Path(project.disk_path)
    authed_url = _inject_auth(project.git_repo_url, project.git_auth_token, username=project.git_username)
    branch = project.git_branch or "main"

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
        logger.info("Cloned %s (%s) -> %s", project.git_repo_url, branch, project.disk_path)
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
        logger.info("Updated %s to origin/%s", project.disk_path, branch)


def _commit_and_push(project: Project) -> bool:
    """Stage all, commit if changes exist, push. Returns True if pushed."""
    root = str(Path(project.disk_path))
    branch = project.git_branch or "main"

    _run_git(["add", "-A"], cwd=root)

    status = _run_git(["status", "--porcelain"], cwd=root)
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


def _should_auto_enqueue_compile(project: Project) -> bool:
    return bool(project.git_sync_auto_compile) and not bool(project.ingest_paused)


# --- Public API ---


async def test_project_git_connection(project: Project) -> dict:
    """Test git connectivity. Returns {"ok": bool, "error": str}."""
    if not project.git_repo_url:
        return {"ok": False, "error": "未配置仓库 URL"}
    if not project.git_auth_token:
        return {"ok": False, "error": "未配置访问凭证"}

    authed_url = _inject_auth(project.git_repo_url, project.git_auth_token, username=project.git_username)
    branch = project.git_branch or "main"

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


async def sync_project_from_git(
    project_id: str,
    *,
    triggered_by: int = 0,
    source: str = "manual",
) -> None:
    """Execute a full git sync cycle for a project."""
    lock = _get_sync_lock(project_id)
    if lock.locked():
        raise RuntimeError("该项目已有同步正在执行")

    async with lock:
        async with async_session() as db:
            project = (await db.execute(
                select(Project).where(Project.id == project_id)
            )).scalar_one_or_none()
            if not project:
                raise RuntimeError(f"Project {project_id} not found")

            if not project.git_repo_url or not project.git_auth_token:
                raise RuntimeError("Git 配置不完整")

            project.last_git_sync_status = "syncing"
            project.last_git_sync_error = ""
            await db.commit()

        try:
            loop = asyncio.get_event_loop()

            # Pull
            await loop.run_in_executor(None, lambda: _ensure_repo(project))

            # Enumerate raw sources
            raw_dir = Path(project.disk_path) / "raw" / "sources"
            if not raw_dir.exists():
                raw_dir.mkdir(parents=True, exist_ok=True)

            source_files = [
                f for f in raw_dir.rglob("*")
                if f.is_file() and not f.name.startswith(".")
            ]

            if not source_files:
                logger.info("No raw source files found for project %s", project_id)

            # Pre-filter: only enqueue files whose content actually changed
            changed_files = []
            for src_file in source_files:
                try:
                    content = parse_document(src_file)
                    if not content.strip():
                        continue
                    source_identity = src_file.name
                    if not await check_cache(project.disk_path, source_identity, content):
                        changed_files.append(src_file)
                except Exception:
                    changed_files.append(src_file)

            if not changed_files:
                logger.info("All source files unchanged for project %s, skipping compile", project_id)

            # Enqueue ingest only for changed files, unless project-level
            # compilation is paused. The file status view will still surface
            # changed files for manual incremental enqueue.
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
            elif not _should_auto_enqueue_compile(project):
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

            # Commit and push
            await loop.run_in_executor(None, lambda: _commit_and_push(project))

            # Mark success
            async with async_session() as db:
                await db.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        last_git_sync_at=datetime.now(timezone.utc),
                        last_git_sync_status="success",
                        last_git_sync_error="",
                    )
                )
                await db.commit()

            logger.info("Git sync completed for project %s (source=%s)", project_id, source)

        except Exception as exc:
            logger.exception("Git sync failed for project %s", project_id)
            async with async_session() as db:
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


async def _scheduled_sync_job(project_id: str) -> None:
    """Wrapper for scheduled execution."""
    try:
        await sync_project_from_git(project_id, triggered_by=0, source="schedule")
    except Exception:
        logger.exception("Scheduled sync failed for project %s", project_id)


async def register_sync_jobs() -> None:
    """Load all git-sync-enabled projects and register cron jobs."""
    _scheduler.remove_all_jobs()
    async with async_session() as db:
        stmt = select(Project).where(Project.git_sync_enabled == True)  # noqa: E712
        projects = list((await db.execute(stmt)).scalars().all())

    for proj in projects:
        trigger_time = proj.git_sync_time or "02:00"
        parts = trigger_time.split(":")
        hour = int(parts[0]) if len(parts) > 0 else 2
        minute = int(parts[1]) if len(parts) > 1 else 0
        trigger = CronTrigger(hour=hour, minute=minute)
        _scheduler.add_job(
            _scheduled_sync_job,
            trigger=trigger,
            args=[proj.id],
            id=f"git_sync_{proj.id}",
            replace_existing=True,
        )
        logger.info("Registered git sync for project '%s' at %02d:%02d", proj.name, hour, minute)


def start_sync_scheduler() -> None:
    """Start the APScheduler and register all sync jobs."""
    if not _scheduler.running:
        _scheduler.start()
        asyncio.get_event_loop().create_task(register_sync_jobs())


def stop_sync_scheduler() -> None:
    """Stop the scheduler."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
