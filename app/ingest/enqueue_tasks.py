"""In-memory progress tracking for async ingest enqueue jobs."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


EnqueueTaskStatus = Literal["queued", "running", "succeeded", "failed"]


class EnqueueTaskResponse(BaseModel):
    task_id: str
    project_id: str
    status: EnqueueTaskStatus
    progress: int = 0
    stage: str = "等待开始"
    file_count: int = 0
    enqueued_count: int = 0
    job_ids: list[str] = Field(default_factory=list)
    error: str = ""


_tasks: dict[str, EnqueueTaskResponse] = {}


def create_enqueue_task(project_id: str, *, file_count: int = 0, stage: str = "等待开始") -> EnqueueTaskResponse:
    task = EnqueueTaskResponse(
        task_id=uuid.uuid4().hex,
        project_id=project_id,
        status="queued",
        progress=0,
        stage=stage,
        file_count=file_count,
    )
    _tasks[task.task_id] = task
    return task


def get_enqueue_task(task_id: str) -> EnqueueTaskResponse | None:
    return _tasks.get(task_id)


def get_active_enqueue_task(project_id: str) -> EnqueueTaskResponse | None:
    for task in reversed(list(_tasks.values())):
        if task.project_id != project_id:
            continue
        if task.status in ("queued", "running"):
            return task
    return None


def set_enqueue_task(
    task_id: str,
    *,
    status: EnqueueTaskStatus | None = None,
    progress: int | None = None,
    stage: str | None = None,
    file_count: int | None = None,
    enqueued_count: int | None = None,
    job_ids: list[str] | None = None,
    error: str | None = None,
) -> None:
    task = _tasks[task_id]
    if status is not None:
        task.status = status
    if progress is not None:
        task.progress = max(0, min(100, progress))
    if stage is not None:
        task.stage = stage
    if file_count is not None:
        task.file_count = file_count
    if enqueued_count is not None:
        task.enqueued_count = enqueued_count
    if job_ids is not None:
        task.job_ids = job_ids
    if error is not None:
        task.error = error
