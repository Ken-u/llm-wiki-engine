"""Shared ingest enqueue execution for sync and async endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.ingest.enqueue_tasks import set_enqueue_task
from app.ingest.files import apply_selection, stage_sources_for_ingest
from app.ingest.queue import ingest_queue
from app.projects.service import get_project_or_404

logger = logging.getLogger(__name__)

ProgressReporter = Callable[[str, int], None]


def _report(task_id: str | None, reporter: ProgressReporter | None, stage: str, progress: int) -> None:
    if task_id:
        set_enqueue_task(task_id, status="running", stage=stage, progress=progress)
    if reporter:
        reporter(stage, progress)


async def execute_ingest_enqueue(
    project_id: str,
    body,
    user_id: int,
    *,
    db: AsyncSession | None = None,
    task_id: str | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    if db is None:
        async with async_session() as session:
            return await _execute_ingest_enqueue_with_db(
                project_id, body, user_id, session, task_id=task_id, reporter=reporter
            )
    return await _execute_ingest_enqueue_with_db(
        project_id, body, user_id, db, task_id=task_id, reporter=reporter
    )


async def _execute_ingest_enqueue_with_db(
    project_id: str,
    body,
    user_id: int,
    db: AsyncSession,
    *,
    task_id: str | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    from app.ingest.router import _build_file_items, _resolve_source_context, _selection_from_request

    project = await get_project_or_404(db, project_id)
    if project.project_type == "case_library":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Wiki compilation is not available for case_library projects. Use case index rebuild instead.",
        )

    source_root, source_repo = await _resolve_source_context(
        project,
        db,
        source_kind=body.source_kind,
        source_repo_id=body.source_repo_id,
    )

    _report(task_id, reporter, "校验选中文件...", 10)
    items = await _build_file_items(project, db, source_root=source_root, source_repo=source_repo)
    by_name = {item.source_file: item for item in items}

    if body.selection is not None:
        try:
            selected = apply_selection(items, _selection_from_request(body.selection))
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    elif body.all:
        if body.status not in ("updated", "not_queued"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only updated or not_queued files can be bulk enqueued")
        selected = [item for item in items if item.status == body.status]
    elif body.source_files:
        missing = [source_file for source_file in body.source_files if source_file not in by_name]
        if missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source file not found: {missing[0]}")
        selected = [by_name[source_file] for source_file in body.source_files]
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No source files selected")

    eligible = [item for item in selected if item.status in ("failed", "updated", "not_queued")]
    if task_id:
        set_enqueue_task(task_id, file_count=len(eligible))

    staging_pairs = [(item.source_path, str(source_root), source_repo) for item in eligible]

    def on_stage(index: int, total: int, rel: str) -> None:
        if total <= 0:
            return
        progress = 20 + int((index / total) * 55)
        _report(task_id, reporter, f"复制文件 ({index}/{total})：{rel}", progress)

    _report(task_id, reporter, "准备源文件...", 18)

    def stage() -> list:
        return stage_sources_for_ingest(
            project.disk_path,
            staging_pairs,
            on_item=on_stage if staging_pairs else None,
        )

    staged_paths = await asyncio.to_thread(stage)

    _report(task_id, reporter, "创建编译任务...", 85)
    job_ids = await ingest_queue.enqueue_many(
        project_id,
        project.disk_path,
        [str(path) for path in staged_paths],
        user_id,
    )
    return {"jobs": job_ids, "count": len(job_ids)}


async def run_enqueue_task(task_id: str, project_id: str, body, user_id: int) -> None:
    try:
        _report(task_id, None, "准备入队...", 5)
        result = await execute_ingest_enqueue(project_id, body, user_id, task_id=task_id)
        set_enqueue_task(
            task_id,
            status="succeeded",
            progress=100,
            stage=f"已成功入队 {result['count']} 个文件",
            enqueued_count=result["count"],
            job_ids=result["jobs"],
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        set_enqueue_task(task_id, status="failed", progress=100, stage="入队失败", error=detail)
    except Exception as exc:
        logger.exception("Enqueue task %s failed", task_id)
        set_enqueue_task(task_id, status="failed", progress=100, stage="入队失败", error=str(exc))
