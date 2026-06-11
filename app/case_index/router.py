"""Case index API: rebuild, status, search."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.case_index.builder import load_manifest, rebuild_case_index
from app.case_index.search import search_cases
from app.database import get_db
from app.projects.service import check_membership, get_project_or_404

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/case-index", tags=["case-index"]
)


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


class StatusResponse(BaseModel):
    status: str
    built_at: str | None = None
    source_count: int = 0
    case_count: int = 0
    chunk_count: int = 0
    error_count: int = 0
    errors: list[str] = []


@router.get("/status", response_model=StatusResponse)
async def case_index_status(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    manifest = load_manifest(project.disk_path)
    if manifest is None:
        return StatusResponse(status="not_built")
    return StatusResponse(
        status=manifest.status,
        built_at=manifest.built_at,
        source_count=manifest.source_count,
        case_count=manifest.case_count,
        chunk_count=manifest.chunk_count,
        error_count=len(manifest.errors),
        errors=manifest.errors[:10],
    )


_rebuild_locks: dict[str, bool] = {}


async def _do_rebuild(project_id: str, disk_path: str):
    try:
        _rebuild_locks[project_id] = True
        await rebuild_case_index(disk_path)
    except Exception:
        logger.exception("Case index rebuild failed for %s", project_id)
    finally:
        _rebuild_locks.pop(project_id, None)


@router.post("/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def trigger_rebuild(
    project_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    if _rebuild_locks.get(project_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Rebuild already in progress"
        )
    background_tasks.add_task(_do_rebuild, project_id, project.disk_path)
    return {"status": "rebuilding"}


@router.post("/search")
async def search_case_index(
    project_id: str,
    body: SearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    manifest = load_manifest(project.disk_path)
    if manifest is None or not manifest.is_ready:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Case index is not built or not ready",
        )
    results = await search_cases(
        project.disk_path, body.query, limit=min(body.limit, 5)
    )
    return {
        "results": [
            {
                "case_id": r.case_id,
                "title": r.title,
                "domain": r.domain,
                "problem_summary": r.problem_summary,
                "root_cause": r.root_cause,
                "resolution": r.resolution,
                "score": r.score,
            }
            for r in results
        ]
    }
