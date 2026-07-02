"""Shared progress tracking for index rebuild jobs."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel


TaskStatus = Literal["queued", "running", "succeeded", "failed"]


class IndexRebuildTaskResponse(BaseModel):
    task_id: str
    target: str
    status: TaskStatus
    progress: int
    stage: str
    project_id: str | None = None
    result: dict[str, Any] | None = None
    error: str = ""


_tasks: dict[str, IndexRebuildTaskResponse] = {}


def create_index_task(target: str, *, project_id: str | None = None, stage: str = "等待开始") -> IndexRebuildTaskResponse:
    task = IndexRebuildTaskResponse(
        task_id=uuid.uuid4().hex,
        target=target,
        project_id=project_id,
        status="queued",
        progress=0,
        stage=stage,
    )
    _tasks[task.task_id] = task
    return task


def get_index_task(task_id: str) -> IndexRebuildTaskResponse | None:
    return _tasks.get(task_id)


def has_active_index_task(*, target: str | None = None, project_id: str | None = None) -> bool:
    for task in _tasks.values():
        if task.status not in ("queued", "running"):
            continue
        if target is not None and task.target != target:
            continue
        if project_id is not None and task.project_id != project_id:
            continue
        return True
    return False


def set_index_task(
    task_id: str,
    *,
    status: TaskStatus | None = None,
    progress: int | None = None,
    stage: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    task = _tasks[task_id]
    if status is not None:
        task.status = status
    if progress is not None:
        task.progress = max(0, min(100, progress))
    if stage is not None:
        task.stage = stage
    if result is not None:
        task.result = result
    if error is not None:
        task.error = error
