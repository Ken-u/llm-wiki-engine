"""Ingest trigger / status / history endpoints."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime
import mimetypes
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
    SourceKind,
    apply_selection,
    browser_source_root,
    build_ingest_record_page,
    filter_and_sort_items,
    is_ingest_records_request,
    is_project_source_dir,
    is_project_source_file,
    list_project_source_files,
    list_source_files_at_root,
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
    source_kind: SourceKind | None = None
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
    file_size: int | None = None
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
    directories: list[dict[str, str]] = Field(default_factory=list)
    dir: str = ""
    recursive: bool = False
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


class IngestSourcePreviewResponse(BaseModel):
    source_file: str
    preview_type: str = "text"
    mime_type: str = "text/plain"
    content: str
    truncated: bool


def _empty_status_counts() -> dict[str, int]:
    return {key: 0 for key in ["processing", "queued", "failed", "updated", "not_queued", "compiled"]}


def _resolve_source_dir(source_root: Path, rel_dir: str) -> tuple[Path, str]:
    normalized = rel_dir.strip().replace("\\", "/").strip("/")
    if normalized in ("", "."):
        return source_root, ""
    candidate = (source_root / normalized).resolve()
    try:
        rel = candidate.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid directory") from None
    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Directory not found")
    if not is_project_source_dir(candidate, source_root):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid directory")
    return candidate, rel


def _list_source_dir(source_root: Path, rel_dir: str) -> tuple[str, list[dict[str, str]], list[Path]]:
    current_dir, normalized = _resolve_source_dir(source_root, rel_dir)
    directories: list[dict[str, str]] = []
    files: list[Path] = []
    for child in sorted(current_dir.iterdir(), key=lambda path: (path.is_file(), path.name.lower())):
        if child.is_dir():
            if is_project_source_dir(child, source_root):
                rel = child.resolve().relative_to(source_root.resolve()).as_posix()
                directories.append({"name": child.name, "path": rel})
        elif is_project_source_file(child, source_root):
            files.append(child)
    return normalized, directories, files


async def _build_file_items(project, db: AsyncSession, *, source_dir: Path | None = None, sources: list[Path] | None = None) -> list:
    if source_dir is None or sources is None:
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
    source_kind: SourceKind = Query(default="remote"),
    q: str = Query(default=""),
    include_globs: list[str] | None = Query(default=None),
    exclude_globs: list[str] | None = Query(default=None),
    dir: str = Query(default=""),
    recursive: bool = Query(default=False),
    sort_dir: SortDirection = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    directories: list[dict[str, str]] = []
    normalized_dir = ""
    filter_selection = IngestSelection(
        include_globs=include_globs or [],
        exclude_globs=exclude_globs or [],
        search=q,
    )
    has_filters = bool(q.strip() or filter_selection.include_globs or filter_selection.exclude_globs)
    if is_ingest_records_request(
        status_filter=status_filter,
        recursive=recursive,
        has_filters=has_filters,
        dir=dir,
    ):
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
        cache = await load_cache(project.disk_path)
        record_page = build_ingest_record_page(
            project_dir=project.disk_path,
            jobs=jobs,
            cache=cache,
            status_filter=status_filter,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size,
        )
        return {
            "items": [asdict(item) for item in record_page.items],
            "directories": [],
            "dir": "",
            "recursive": False,
            "total": record_page.total,
            "page": record_page.page,
            "page_size": record_page.page_size,
            "project_paused": bool(project.ingest_paused),
            "status_counts": record_page.counts,
        }

    source_dir = browser_source_root(project.disk_path, source_kind)
    if recursive or has_filters:
        sources = list_source_files_at_root(source_dir)
        items = await _build_file_items(project, db, source_dir=source_dir, sources=sources)
    else:
        normalized_dir, directories, sources = _list_source_dir(source_dir, dir)
        items = await _build_file_items(project, db, source_dir=source_dir, sources=sources)

    if has_filters:
        try:
            items = apply_selection(items, filter_selection)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    counts = _empty_status_counts()
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    if status_filter:
        items = [item for item in items if item.status == status_filter]
    items = filter_and_sort_items(items, search="", sort_dir=sort_dir)
    page_data = paginate_items(items, page=page, page_size=page_size)
    return {
        "items": [asdict(item) for item in page_data.items],
        "directories": directories,
        "dir": normalized_dir,
        "recursive": bool(recursive or has_filters),
        "total": page_data.total,
        "page": page_data.page,
        "page_size": page_data.page_size,
        "project_paused": bool(project.ingest_paused),
        "status_counts": counts,
    }


@router.get("/files/content", response_model=IngestSourcePreviewResponse)
async def preview_source_file(
    project_id: str,
    source_file: str = Query(..., min_length=1),
    source_kind: SourceKind = Query(default="remote"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    source_dir = browser_source_root(project.disk_path, source_kind)
    target = (source_dir / source_file).resolve()
    try:
        rel = target.relative_to(source_dir.resolve()).as_posix()
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid source file") from None
    if not target.exists() or not target.is_file() or not is_project_source_file(target, source_dir):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source file not found")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if mime_type.startswith("image/"):
        max_image_bytes = 5 * 1024 * 1024
        size = target.stat().st_size
        if size > max_image_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large to preview")
        data = base64.b64encode(target.read_bytes()).decode("ascii")
        return {
            "source_file": rel,
            "preview_type": "image",
            "mime_type": mime_type,
            "content": f"data:{mime_type};base64,{data}",
            "truncated": False,
        }

    try:
        content = parse_document(target)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Cannot preview source file: {exc}") from exc

    limit = 200_000
    truncated = len(content) > limit
    return {
        "source_file": rel,
        "preview_type": "text",
        "mime_type": mime_type,
        "content": content[:limit],
        "truncated": truncated,
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
        if body.source_kind is not None:
            source_dir = browser_source_root(project.disk_path, body.source_kind)
            scoped_sources = list_source_files_at_root(source_dir)
            scoped_items = await _build_file_items(
                project,
                db,
                source_dir=source_dir,
                sources=scoped_sources,
            )
            by_name = {item.source_file: item for item in scoped_items}
        missing = [source_file for source_file in body.source_files if source_file not in by_name]
        if missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source file not found: {missing[0]}")
        selected = [by_name[source_file] for source_file in body.source_files]

    eligible = [item for item in selected if item.status in ("failed", "updated", "not_queued")]
    enqueue_source_dir = (
        browser_source_root(project.disk_path, body.source_kind)
        if body.source_kind is not None
        else preferred_source_root(project.disk_path)
    )
    job_ids = []
    for item in eligible:
        staged = stage_source_for_ingest(project.disk_path, item.source_path, str(enqueue_source_dir))
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
