"""Shared ingest enqueue execution."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.ingest.files import apply_selection, stage_source_for_ingest
from app.ingest.queue import ingest_queue
from app.projects.service import get_project_or_404


async def execute_ingest_enqueue(
    project_id: str,
    body,
    user_id: int,
    *,
    db: AsyncSession | None = None,
) -> dict[str, Any]:
    if db is None:
        async with async_session() as session:
            return await _execute_ingest_enqueue_with_db(project_id, body, user_id, session)
    return await _execute_ingest_enqueue_with_db(project_id, body, user_id, db)


async def _execute_ingest_enqueue_with_db(
    project_id: str,
    body,
    user_id: int,
    db: AsyncSession,
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
    else:
        if not body.source_files:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No source files selected")
        missing = [source_file for source_file in body.source_files if source_file not in by_name]
        if missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source file not found: {missing[0]}")
        selected = [by_name[source_file] for source_file in body.source_files]

    eligible = [item for item in selected if item.status in ("failed", "updated", "not_queued")]
    job_ids = []
    for item in eligible:
        staged = stage_source_for_ingest(project.disk_path, item.source_path, str(source_root), source_repo)
        job_ids.append(
            await ingest_queue.enqueue(
                project_id,
                project.disk_path,
                str(staged),
                user_id,
            )
        )
    return {"jobs": job_ids, "count": len(job_ids)}
