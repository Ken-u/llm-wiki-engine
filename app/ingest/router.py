"""Ingest trigger / status / history endpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.documents.parser import parse_document
from app.ingest.cache import content_hash, load_cache
from app.ingest.files import (
    IngestFileStatus,
    IngestSelection,
    SortDirection,
    apply_selection,
    filter_and_sort_items,
    is_project_source_file,
    list_project_source_files,
    paginate_items,
    preferred_source_root,
    resolve_file_statuses,
    stage_source_for_ingest,
)
from app.ingest.models import IngestJob
from app.ingest.queue import ingest_queue
from app.projects.service import check_membership, get_project_or_404

router = APIRouter(prefix="/api/projects/{project_id}/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    source_file: str | None = None  # specific file in raw/sources/; None = all
    source_files: list[str] | None = None


class IngestSelectionRequest(BaseModel):
    statuses: list[IngestFileStatus] = Field(default_factory=list)
    include_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)
    q: str = ""
    limit: int | None = None

    def to_selection(self) -> IngestSelection:
        return IngestSelection(
            statuses=self.statuses,
            include_globs=self.include_globs,
            exclude_globs=self.exclude_globs,
            search=self.q,
            limit=self.limit,
        )


class EnqueueFilesRequest(BaseModel):
    source_files: list[str] | None = None
    status: IngestFileStatus | None = None
    all: bool = False
    selection: IngestSelectionRequest | None = None


class IngestJobResponse(BaseModel):
    id: str
    source_path: str
    status: str
    progress: str
    step: int = 0
    files_written: list[str] | None = None
    error: str | None = None
    retry_count: int
    created_at: datetime
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class IngestFileResponse(BaseModel):
    source_file: str
    source_path: str
    status: str
    job_id: str | None = None
    job_status: str | None = None
    progress: str = ""
    step: int = 0
    files_written: list[str] | None = None
    error: str | None = None
    retry_count: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None


class IngestFilePageResponse(BaseModel):
    items: list[IngestFileResponse]
    total: int
    page: int
    page_size: int
    project_paused: bool
    status_counts: dict[str, int]


class IngestFilePreviewResponse(BaseModel):
    items: list[IngestFileResponse]
    total: int
    eligible_count: int
    truncated: bool


async def _build_file_items(project, db: AsyncSession) -> list:
    source_dir, sources = list_project_source_files(project.disk_path)
    cache = await load_cache(project.disk_path)
    cached_identities = set(cache.keys())
    changed_identities: set[str] = set()

    for src in sources:
        identity = src.resolve().relative_to(source_dir.resolve()).as_posix()
        cache_key = identity if identity in cache else src.name
        if cache_key not in cache:
            continue
        cached_identities.add(identity)
        try:
            if content_hash(parse_document(src)) != cache[cache_key]:
                changed_identities.add(identity)
        except Exception:
            changed_identities.add(identity)

    jobs = list(
        (
            await db.execute(
                select(IngestJob)
                .where(IngestJob.project_id == project.id)
                .where(IngestJob.status != "superseded")
                .order_by(IngestJob.created_at.asc())
            )
        ).scalars().all()
    )
    return resolve_file_statuses(
        source_paths=[str(src) for src in sources],
        jobs=jobs,
        changed_identities=changed_identities,
        cached_identities=cached_identities,
        source_root=str(source_dir),
    )


def _resolve_requested_sources(project_dir: str, source_files: list[str]) -> list[Path]:
    source_dir = preferred_source_root(project_dir)
    resolved: list[Path] = []
    for source_file in source_files:
        src = (source_dir / source_file).resolve()
        try:
            src.relative_to(source_dir.resolve())
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid source file: {source_file}") from None
        if not src.exists() or not src.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source file not found: {source_file}")
        if not is_project_source_file(src, source_dir):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid source file: {source_file}")
        resolved.append(src)
    return resolved


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingest(
    project_id: str,
    body: IngestRequest = IngestRequest(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    if project.project_type == "case_library":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Wiki compilation is not available for case_library projects. Use case index rebuild instead.",
        )

    source_dir = preferred_source_root(project.disk_path)
    requested_files = body.source_files or ([body.source_file] if body.source_file else None)
    if requested_files:
        sources = _resolve_requested_sources(project.disk_path, requested_files)
    else:
        if not source_dir.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No sources directory")
        _, sources = list_project_source_files(project.disk_path)

    if not sources:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No source files found")

    job_ids = []
    for src in sources:
        staged = stage_source_for_ingest(project.disk_path, str(src), str(source_dir))
        jid = await ingest_queue.enqueue(project_id, project.disk_path, str(staged), user.id)
        job_ids.append(jid)

    return {"jobs": job_ids, "count": len(job_ids)}


@router.get("/files", response_model=IngestFilePageResponse)
async def ingest_files(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    status_filter: IngestFileStatus | None = Query(default=None, alias="status"),
    q: str = Query(default=""),
    sort_dir: SortDirection = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    items = await _build_file_items(project, db)
    counts = {key: 0 for key in ["processing", "queued", "failed", "updated", "not_queued", "compiled"]}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    if status_filter:
        items = [item for item in items if item.status == status_filter]
    items = filter_and_sort_items(items, search=q, sort_dir=sort_dir)
    page_data = paginate_items(items, page=page, page_size=page_size)
    return {
        "items": [asdict(item) for item in page_data.items],
        "total": page_data.total,
        "page": page_data.page,
        "page_size": page_data.page_size,
        "project_paused": bool(project.ingest_paused),
        "status_counts": counts,
    }


def _selection_from_request(body: IngestSelectionRequest) -> IngestSelection:
    try:
        return body.to_selection()
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@router.post("/files/preview", response_model=IngestFilePreviewResponse)
async def preview_ingest_files(
    project_id: str,
    body: IngestSelectionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    if project.project_type == "case_library":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Wiki compilation is not available for case_library projects. Use case index rebuild instead.",
        )

    items = await _build_file_items(project, db)
    try:
        matched = apply_selection(items, _selection_from_request(body))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    eligible = [item for item in matched if item.status in ("failed", "updated", "not_queued")]
    preview_limit = 100
    return {
        "items": [asdict(item) for item in matched[:preview_limit]],
        "total": len(matched),
        "eligible_count": len(eligible),
        "truncated": len(matched) > preview_limit,
    }


@router.post("/files/enqueue", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_ingest_files(
    project_id: str,
    body: EnqueueFilesRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    if project.project_type == "case_library":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Wiki compilation is not available for case_library projects. Use case index rebuild instead.",
        )
    items = await _build_file_items(project, db)
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
    source_dir = preferred_source_root(project.disk_path)
    job_ids = []
    for item in eligible:
        staged = stage_source_for_ingest(project.disk_path, item.source_path, str(source_dir))
        job_ids.append(
            await ingest_queue.enqueue(
                project_id,
                project.disk_path,
                str(staged),
                user.id,
            )
        )
    return {"jobs": job_ids, "count": len(job_ids)}


@router.post("/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_ingest(
    project_id: str,
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retry a failed ingest job, then hide the old failure."""
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
    if not job or job.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status != "failed":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only failed jobs can be retried")

    # Mark old job as superseded so it won't show in history
    job.status = "superseded"
    await db.commit()

    new_jid = await ingest_queue.enqueue(project_id, project.disk_path, job.source_path, user.id)
    return {"job_id": new_jid}


@router.get("/status", response_model=list[IngestJobResponse])
async def ingest_status(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    stmt = (
        select(IngestJob)
        .where(IngestJob.project_id == project_id)
        .where(IngestJob.status.in_(["pending", "processing", "paused"]))
        .order_by(IngestJob.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


@router.post("/pause", status_code=status.HTTP_200_OK)
async def pause_all_ingest(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    await get_project_or_404(db, project_id)
    result = await ingest_queue.pause_project(project_id)
    return result


@router.post("/resume", status_code=status.HTTP_200_OK)
async def resume_all_ingest(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    count = await ingest_queue.resume_project(project_id, project.disk_path)
    resumed = await ingest_queue.resume_all(project_id, project.disk_path)
    return {"scheduled": count, "resumed": resumed}


@router.post("/{job_id}/pause", status_code=status.HTTP_200_OK)
async def pause_ingest_job(
    project_id: str,
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    await get_project_or_404(db, project_id)

    job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
    if not job or job.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status not in ("pending", "processing"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only active jobs can be paused")

    if not await ingest_queue.pause_job(job_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Job cannot be paused")
    return {"status": "paused"}


@router.post("/{job_id}/resume", status_code=status.HTTP_200_OK)
async def resume_ingest_job(
    project_id: str,
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
    if not job or job.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status != "paused":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only paused jobs can be resumed")

    if not await ingest_queue.resume_job(job_id, project_id, project.disk_path):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Job cannot be resumed")
    return {"status": "pending"}


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ingest_job(
    project_id: str,
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user, require="owner")
    job = (await db.execute(select(IngestJob).where(IngestJob.id == job_id))).scalar_one_or_none()
    if not job or job.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status in ("pending", "processing"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot delete a running job")
    await db.delete(job)
    await db.commit()


@router.get("/history", response_model=list[IngestJobResponse])
async def ingest_history(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    await check_membership(db, project_id, user)
    stmt = (
        select(IngestJob)
        .where(IngestJob.project_id == project_id)
        .where(IngestJob.status != "superseded")
        .order_by(IngestJob.created_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())
