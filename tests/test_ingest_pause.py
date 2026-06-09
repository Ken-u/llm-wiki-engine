"""Unit tests for ingest pause/resume behavior."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.ingest.exceptions import JobPaused
from app.ingest.pipeline import _ensure_not_paused
from app.ingest.queue import IngestQueue


def test_ensure_not_paused_raises_when_checker_fails():
    async def checker():
        raise JobPaused()

    with pytest.raises(JobPaused):
        asyncio.run(_ensure_not_paused(checker))


def test_ensure_not_paused_noop_without_checker():
    asyncio.run(_ensure_not_paused(None))


def test_pause_job_updates_status():
    queue = IngestQueue()
    job = AsyncMock(status="pending", step=0)

    async def run():
        with patch.object(queue, "_get_job", AsyncMock(return_value=job)):
            with patch.object(queue, "_update_job", AsyncMock()) as update:
                ok = await queue.pause_job("job-1")
        assert ok is True
        update.assert_awaited_once_with("job-1", status="paused", progress="Paused")

    asyncio.run(run())


def test_pause_job_processing_shows_pausing():
    queue = IngestQueue()
    job = AsyncMock(status="processing", step=1)

    async def run():
        with patch.object(queue, "_get_job", AsyncMock(return_value=job)):
            with patch.object(queue, "_update_job", AsyncMock()) as update:
                ok = await queue.pause_job("job-1")
        assert ok is True
        update.assert_awaited_once_with("job-1", status="paused", progress="Pausing...")

    asyncio.run(run())


def test_pause_job_rejects_inactive():
    queue = IngestQueue()
    job = AsyncMock(status="done")

    async def run():
        with patch.object(queue, "_get_job", AsyncMock(return_value=job)):
            ok = await queue.pause_job("job-1")
        assert ok is False

    asyncio.run(run())


def test_resume_job_requeues_when_no_task():
    queue = IngestQueue()
    job = AsyncMock(status="paused", step=2, source_path="/tmp/doc.pdf", project_id="p1")

    async def run():
        with patch.object(queue, "_get_job", AsyncMock(return_value=job)):
            with patch.object(queue, "_update_job", AsyncMock()) as update:
                with patch.object(queue, "_is_project_paused", AsyncMock(return_value=False)):
                    with patch.object(queue, "_schedule_task") as schedule:
                        ok = await queue.resume_job("job-1", "p1", "/data/proj")
        assert ok is True
        update.assert_awaited_once_with("job-1", status="pending", progress="Queued")
        schedule.assert_called_once_with(
            "job-1",
            "p1",
            "/data/proj",
            "/tmp/doc.pdf",
            resume_step=2,
        )

    asyncio.run(run())


def test_resume_job_skips_schedule_when_task_alive():
    queue = IngestQueue()
    job = AsyncMock(status="paused", step=1, source_path="/tmp/doc.pdf", project_id="p1")

    async def run():
        queue._tasks["job-1"] = asyncio.create_task(asyncio.sleep(60))
        try:
            with patch.object(queue, "_get_job", AsyncMock(return_value=job)):
                with patch.object(queue, "_update_job", AsyncMock()):
                    with patch.object(queue, "_is_project_paused", AsyncMock(return_value=False)):
                        with patch.object(queue, "_schedule_task") as schedule:
                            ok = await queue.resume_job("job-1", "p1", "/data/proj")
            assert ok is True
            schedule.assert_not_called()
        finally:
            queue._tasks["job-1"].cancel()
            with pytest.raises(asyncio.CancelledError):
                await queue._tasks["job-1"]

    asyncio.run(run())


def test_execute_handles_job_paused():
    queue = IngestQueue()

    async def run():
        with patch.object(queue, "_update_job", AsyncMock()):
            with patch(
                "app.ingest.queue.auto_ingest",
                AsyncMock(side_effect=JobPaused()),
            ):
                with patch.object(
                    queue,
                    "_get_job",
                    AsyncMock(return_value=AsyncMock(status="paused", step=1)),
                ):
                    await queue._execute("job-1", "/data/proj", "/tmp/doc.pdf")

    asyncio.run(run())


def test_enqueue_does_not_schedule_when_project_is_paused():
    queue = IngestQueue()

    async def run():
        with patch.object(queue, "_create_job", AsyncMock(return_value="job-1")):
            with patch.object(queue, "_is_project_paused", AsyncMock(return_value=True)):
                with patch.object(queue, "_schedule_task") as schedule:
                    job_id = await queue.enqueue("p1", "/data/proj", "/tmp/doc.pdf", 1)

        assert job_id == "job-1"
        schedule.assert_not_called()

    asyncio.run(run())


def test_pause_project_clears_pending_and_pauses_processing():
    queue = IngestQueue()

    async def run():
        with patch.object(queue, "_set_project_paused", AsyncMock()) as set_paused:
            with patch.object(queue, "_delete_waiting_jobs", AsyncMock(return_value=3)) as delete_waiting:
                with patch.object(queue, "pause_all", AsyncMock(return_value=1)) as pause_all:
                    result = await queue.pause_project("p1")

        assert result == {"cleared": 3, "paused": 1}
        set_paused.assert_awaited_once_with("p1", True)
        delete_waiting.assert_awaited_once_with("p1")
        pause_all.assert_awaited_once_with("p1", include_pending=False)

    asyncio.run(run())


def test_resume_project_schedules_existing_manual_queue():
    queue = IngestQueue()
    jobs = [AsyncMock(id="job-1", project_id="p1", source_path="/tmp/doc.pdf", step=0)]

    async def run():
        with patch.object(queue, "_set_project_paused", AsyncMock()) as set_paused:
            with patch.object(queue, "_get_pending_jobs", AsyncMock(return_value=jobs)):
                with patch.object(queue, "_schedule_task") as schedule:
                    count = await queue.resume_project("p1", "/data/proj")

        assert count == 1
        set_paused.assert_awaited_once_with("p1", False)
        schedule.assert_called_once_with("job-1", "p1", "/data/proj", "/tmp/doc.pdf", resume_step=0)

    asyncio.run(run())
