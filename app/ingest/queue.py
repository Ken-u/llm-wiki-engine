"""Async ingest task queue with per-project locking.

Each project's ingest jobs run serially (to avoid index.md race conditions),
but different projects can run in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.ingest.models import IngestJob
from app.ingest.pipeline import auto_ingest

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


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

    async def enqueue(
        self,
        project_id: str,
        project_dir: str,
        source_path: str,
        user_id: int,
    ) -> str:
        """Create an IngestJob and schedule it."""
        job_id = str(uuid.uuid4())

        async with async_session() as db:
            job = IngestJob(
                id=job_id,
                project_id=project_id,
                source_path=source_path,
                status="pending",
                progress="Queued",
                created_by=user_id,
            )
            db.add(job)
            await db.commit()

        task = asyncio.create_task(self._run_job(job_id, project_id, project_dir, source_path))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return job_id

    async def _run_job(self, job_id: str, project_id: str, project_dir: str, source_path: str) -> None:
        lock = self._get_lock(project_id)
        async with lock:
            await self._execute(job_id, project_dir, source_path)

    async def _update_job(self, job_id: str, **kwargs) -> None:
        async with async_session() as db:
            await db.execute(update(IngestJob).where(IngestJob.id == job_id).values(**kwargs))
            await db.commit()

    async def _execute(self, job_id: str, project_dir: str, source_path: str) -> None:
        await self._update_job(job_id, status="processing", progress="Starting...")

        async def on_progress(msg: str):
            await self._update_job(job_id, progress=msg)

        try:
            written = await auto_ingest(project_dir, source_path, on_progress=on_progress)
            await self._update_job(
                job_id,
                status="done",
                progress="Complete",
                files_written=written,
                completed_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)

            async with async_session() as db:
                job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
                retry_count = (job.retry_count or 0) if job else 0

            if retry_count < MAX_RETRIES:
                await self._update_job(
                    job_id,
                    status="pending",
                    progress=f"Retry {retry_count + 1}/{MAX_RETRIES}",
                    retry_count=retry_count + 1,
                    error=str(exc),
                )
                await self._execute(job_id, project_dir, source_path)
            else:
                await self._update_job(
                    job_id,
                    status="failed",
                    progress="Failed",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc),
                )


ingest_queue = IngestQueue()
