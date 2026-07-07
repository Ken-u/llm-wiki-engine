"""Async ingest task queue with per-project locking and crash recovery.

Each project's ingest jobs run serially (to avoid index.md race conditions),
but different projects can run in parallel.

On startup, any jobs left in 'pending' or 'processing' status from a previous
crash are automatically re-enqueued.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update

from app.database import async_session
from app.ingest.exceptions import JobPaused
from app.ingest.models import IngestJob
from app.ingest.pipeline import auto_ingest

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
PAUSE_POLL_INTERVAL = 0.5


class IngestQueue:
    def __init__(self) -> None:
        self._project_locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()

    def _get_lock(self, project_id: str) -> asyncio.Lock:
        if project_id not in self._project_locks:
            self._project_locks[project_id] = asyncio.Lock()
        return self._project_locks[project_id]

    async def _get_job(self, job_id: str) -> IngestJob | None:
        async with async_session() as db:
            return (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()

    async def _is_project_paused(self, project_id: str) -> bool:
        from app.projects.models import Project
        async with async_session() as db:
            paused = (
                await db.execute(select(Project.ingest_paused).where(Project.id == project_id))
            ).scalar_one_or_none()
            return bool(paused)

    async def _set_project_paused(self, project_id: str, paused: bool) -> None:
        from app.projects.models import Project
        async with async_session() as db:
            await db.execute(update(Project).where(Project.id == project_id).values(ingest_paused=paused))
            await db.commit()

    async def _is_paused(self, job_id: str) -> bool:
        job = await self._get_job(job_id)
        return job is not None and job.status == "paused"

    async def _wait_until_resumed(self, job_id: str) -> None:
        while await self._is_paused(job_id):
            await asyncio.sleep(PAUSE_POLL_INTERVAL)

    def _make_pause_checker(self, job_id: str):
        async def check_pause() -> None:
            if await self._is_paused(job_id):
                raise JobPaused()

        return check_pause

    def _schedule_task(
        self,
        job_id: str,
        project_id: str,
        project_dir: str,
        source_path: str,
        *,
        resume_step: int = 0,
    ) -> None:
        task = asyncio.create_task(
            self._run_job(job_id, project_id, project_dir, source_path, resume_step=resume_step)
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda _, jid=job_id: self._tasks.pop(jid, None))

    async def recover_interrupted_jobs(self) -> int:
        """Re-enqueue jobs that were pending/processing when the server last stopped.

        Returns the number of recovered jobs.
        """
        async with async_session() as db:
            stmt = (
                select(IngestJob)
                .where(IngestJob.status.in_(["pending", "processing"]))
                .order_by(IngestJob.created_at.asc())
            )
            jobs = list((await db.execute(stmt)).scalars().all())

            if not jobs:
                return 0

            # Need project disk_path — load them
            from app.projects.models import Project
            proj_cache: dict[str, str] = {}
            paused_projects: set[str] = set()
            for job in jobs:
                if job.project_id not in proj_cache:
                    proj = (await db.execute(
                        select(Project).where(Project.id == job.project_id)
                    )).scalar_one_or_none()
                    if proj:
                        proj_cache[job.project_id] = proj.disk_path
                        if proj.ingest_paused:
                            paused_projects.add(job.project_id)

        recovered = 0
        for job in jobs:
            disk_path = proj_cache.get(job.project_id)
            if not disk_path:
                logger.warning("Skipping recovery of job %s: project %s not found", job.id, job.project_id)
                continue
            if job.project_id in paused_projects:
                logger.info("Skipping recovery of job %s: project %s compile is paused", job.id, job.project_id)
                continue

            logger.info(
                "Recovering interrupted job %s (source=%s, step=%d, retry=%d)",
                job.id, job.source_path, job.step or 0, job.retry_count or 0,
            )
            self._schedule_task(
                job.id,
                job.project_id,
                disk_path,
                job.source_path,
                resume_step=job.step or 0,
            )
            recovered += 1

        return recovered

    async def enqueue(
        self,
        project_id: str,
        project_dir: str,
        source_path: str,
        user_id: int,
    ) -> str:
        """Create an IngestJob and schedule it."""
        job_ids = await self.enqueue_many(project_id, project_dir, [source_path], user_id)
        return job_ids[0]

    async def enqueue_many(
        self,
        project_id: str,
        project_dir: str,
        source_paths: list[str],
        user_id: int,
    ) -> list[str]:
        """Create multiple IngestJobs in one transaction and schedule them."""
        if not source_paths:
            return []
        job_ids = [str(uuid.uuid4()) for _ in source_paths]
        async with async_session() as db:
            for job_id, source_path in zip(job_ids, source_paths, strict=True):
                db.add(
                    IngestJob(
                        id=job_id,
                        project_id=project_id,
                        source_path=source_path,
                        status="pending",
                        progress="Queued",
                        step=0,
                        created_by=user_id,
                    )
                )
            await db.commit()

        if not await self._is_project_paused(project_id):
            for job_id, source_path in zip(job_ids, source_paths, strict=True):
                self._schedule_task(job_id, project_id, project_dir, source_path)
        return job_ids

    async def _create_job(
        self,
        project_id: str,
        source_path: str,
        user_id: int,
    ) -> str:
        job_id = str(uuid.uuid4())
        async with async_session() as db:
            db.add(
                IngestJob(
                    id=job_id,
                    project_id=project_id,
                    source_path=source_path,
                    status="pending",
                    progress="Queued",
                    step=0,
                    created_by=user_id,
                )
            )
            await db.commit()
        return job_id

    async def pause_job(self, job_id: str) -> bool:
        job = await self._get_job(job_id)
        if not job or job.status not in ("pending", "processing"):
            return False

        progress = "Pausing..." if job.status == "processing" else "Paused"
        await self._update_job(job_id, status="paused", progress=progress)
        return True

    async def resume_job(
        self,
        job_id: str,
        project_id: str,
        project_dir: str,
    ) -> bool:
        job = await self._get_job(job_id)
        if not job or job.project_id != project_id or job.status != "paused":
            return False

        resume_step = job.step or 0
        source_path = job.source_path
        await self._update_job(job_id, status="pending", progress="Queued")

        if job_id not in self._tasks and not await self._is_project_paused(project_id):
            self._schedule_task(
                job_id,
                project_id,
                project_dir,
                source_path,
                resume_step=resume_step,
            )
        return True

    async def pause_all(self, project_id: str, *, include_pending: bool = True) -> int:
        statuses = ["processing"]
        if include_pending:
            statuses.append("pending")
        async with async_session() as db:
            jobs = list(
                (
                    await db.execute(
                        select(IngestJob).where(
                            IngestJob.project_id == project_id,
                            IngestJob.status.in_(statuses),
                        )
                    )
                ).scalars().all()
            )

        paused = 0
        for job in jobs:
            if await self.pause_job(job.id):
                paused += 1
        return paused

    async def _delete_waiting_jobs(self, project_id: str) -> int:
        async with async_session() as db:
            result = await db.execute(
                delete(IngestJob).where(
                    IngestJob.project_id == project_id,
                    IngestJob.status.in_(["pending", "paused"]),
                )
            )
            await db.commit()
            return int(result.rowcount or 0)

    async def _get_pending_jobs(self, project_id: str) -> list[IngestJob]:
        async with async_session() as db:
            return list(
                (
                    await db.execute(
                        select(IngestJob)
                        .where(IngestJob.project_id == project_id, IngestJob.status == "pending")
                        .order_by(IngestJob.created_at.asc())
                    )
                ).scalars().all()
            )

    async def pause_project(self, project_id: str) -> dict[str, int]:
        await self._set_project_paused(project_id, True)
        cleared = await self._delete_waiting_jobs(project_id)
        paused = await self.pause_all(project_id, include_pending=False)
        return {"cleared": cleared, "paused": paused}

    async def resume_project(self, project_id: str, project_dir: str) -> int:
        await self._set_project_paused(project_id, False)
        jobs = await self._get_pending_jobs(project_id)
        scheduled = 0
        for job in jobs:
            if job.id not in self._tasks:
                self._schedule_task(
                    job.id,
                    project_id,
                    project_dir,
                    job.source_path,
                    resume_step=job.step or 0,
                )
                scheduled += 1
        return scheduled

    async def resume_all(self, project_id: str, project_dir: str) -> int:
        async with async_session() as db:
            jobs = list(
                (
                    await db.execute(
                        select(IngestJob).where(
                            IngestJob.project_id == project_id,
                            IngestJob.status == "paused",
                        )
                    )
                ).scalars().all()
            )

        resumed = 0
        for job in jobs:
            if await self.resume_job(job.id, project_id, project_dir):
                resumed += 1
        return resumed

    async def _run_job(
        self, job_id: str, project_id: str, project_dir: str, source_path: str, *, resume_step: int = 0
    ) -> None:
        lock = self._get_lock(project_id)
        async with lock:
            await self._wait_until_resumed(job_id)
            await self._execute(job_id, project_id, project_dir, source_path, resume_step=resume_step)

    async def _update_job(self, job_id: str, **kwargs) -> None:
        async with async_session() as db:
            await db.execute(update(IngestJob).where(IngestJob.id == job_id).values(**kwargs))
            await db.commit()

    async def _execute(
        self,
        job_id: str,
        project_id: str,
        project_dir: str,
        source_path: str,
        *,
        resume_step: int = 0,
    ) -> None:
        await self._update_job(job_id, status="processing", progress="Starting...", completed_at=None)

        async def on_progress(msg: str):
            await self._update_job(job_id, progress=msg)

        async def on_step(step: int):
            await self._update_job(job_id, step=step)

        try:
            written = await auto_ingest(
                project_dir,
                source_path,
                project_id=project_id,
                on_progress=on_progress,
                on_step=on_step,
                resume_step=resume_step,
                should_pause=self._make_pause_checker(job_id),
            )
            await self._update_job(
                job_id,
                status="done",
                progress="Complete",
                step=3,
                files_written=written,
                error=None,
                completed_at=datetime.now(timezone.utc),
            )
            if written:
                from app.projects.git_sync import maybe_auto_publish_project_to_git

                await maybe_auto_publish_project_to_git(
                    project_id,
                    triggered_by=0,
                    reason=f"ingest:{job_id}",
                )
        except JobPaused:
            job = await self._get_job(job_id)
            if job and job.status == "paused":
                await self._update_job(job_id, progress="Paused")
            logger.info("Ingest job %s paused at step %d", job_id, (job.step if job else 0) or 0)
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)

            async with async_session() as db:
                job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
                retry_count = (job.retry_count or 0) if job else 0
                current_step = (job.step or 0) if job else 0

            if retry_count < MAX_RETRIES:
                await self._update_job(
                    job_id,
                    status="pending",
                    progress=f"Retry {retry_count + 1}/{MAX_RETRIES}",
                    retry_count=retry_count + 1,
                    error=str(exc),
                )
                # Retry from the last completed step
                await self._execute(job_id, project_id, project_dir, source_path, resume_step=current_step)
            else:
                await self._update_job(
                    job_id,
                    status="failed",
                    progress="Failed",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc),
                )


ingest_queue = IngestQueue()
