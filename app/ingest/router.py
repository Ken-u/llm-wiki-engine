"""Ingest trigger / status / history endpoints."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict
from datetime import datetime
import mimetypes
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.documents.parser import parse_document
from app.ingest.cache import content_hash, load_cache
from app.ingest.enqueue_service import execute_ingest_enqueue, run_enqueue_task
from app.ingest.enqueue_tasks import (
    EnqueueTaskResponse,
    create_enqueue_task,
    get_active_enqueue_task,
    get_enqueue_task,
)
from app.ingest.files import (
    IngestFileStatus,
    IngestSelection,
    SourceRepoMetadata,
    SortDirection,
    apply_selection,
    build_ingest_record_page,
    filter_and_sort_items,
    is_ingest_records_request,
    is_project_source_file,
    list_project_source_files,
    list_source_tree,
    paginate_items,
    preferred_source_root,
    raw_source_root,
    resolve_file_statuses,
    stage_source_for_ingest,
)
from app.ingest.models import IngestJob
from app.ingest.queue import ingest_queue
from app.projects import source_repositories as source_repo_service
from app.projects.service import check_membership, get_project_or_404

router = APIRouter(prefix="/api/projects/{project_id}/ingest", tags=["ingest"])
SourceKind = Literal["local", "remote"]


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
    source_repo_id: str | None = None
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
    source_repo_id: str | None = None
    source_repo_key: str | None = None
    source_repo_name: str | None = None
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
    directories: list[dict] = Field(default_factory=list)
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


def _repo_metadata(repo) -> SourceRepoMetadata:
    return SourceRepoMetadata(id=repo.id, key=repo.key, name=repo.name)


async def _resolve_source_context(
    project,
    db: AsyncSession,
    *,
    source_kind: SourceKind | None,
    source_repo_id: str | None,
) -> tuple[Path, SourceRepoMetadata | None]:
    if source_kind == "local":
        return raw_source_root(project.disk_path), None
    if source_kind == "remote":
        if not source_repo_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "source_repo_id is required for remote source files")
        repo = await source_repo_service.get_source_repository_or_404(db, project, source_repo_id)
        return source_repo_service.source_repo_checkout_root(project, repo), _repo_metadata(repo)
    return preferred_source_root(project.disk_path), None


async def _build_file_items(
    project,
    db: AsyncSession,
    *,
    source_root: Path | None = None,
    source_repo: SourceRepoMetadata | None = None,
) -> list:
    if source_root is None:
        source_dir, sources = list_project_source_files(project.disk_path)
    else:
        source_dir = source_root
        if not source_dir.exists():
            sources = []
        else:
            sources = sorted(
                [
                    path for path in source_dir.rglob("*")
                    if is_project_source_file(path, source_dir)
                ],
                key=lambda path: path.resolve().relative_to(source_dir.resolve()).as_posix().lower(),
            )
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
    items = resolve_file_statuses(
        source_paths=[str(src) for src in sources],
        jobs=jobs,
        changed_identities=changed_identities,
        cached_identities=cached_identities,
        source_root=str(source_dir),
    )
    for item in items:
        try:
            item.file_size = Path(item.source_path).stat().st_size
        except OSError:
            item.file_size = None
        if source_repo:
            item.source_repo_id = source_repo.id
            item.source_repo_key = source_repo.key
            item.source_repo_name = source_repo.name
    return items


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


def _resolve_source_file(root: Path, source_file: str) -> Path:
    normalized = source_file.strip().replace("\\", "/")
    if not normalized:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "source_file is required")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid source file: {source_file}")
    source_root = root.resolve()
    src = (source_root / normalized).resolve()
    try:
        src.relative_to(source_root)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid source file: {source_file}") from None
    if not src.exists() or not src.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source file not found: {source_file}")
    if not is_project_source_file(src, source_root):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid source file: {source_file}")
    return src


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
    source_kind: SourceKind | None = Query(default=None),
    source_repo_id: str | None = Query(default=None),
    dir: str = Query(default=""),
    recursive: bool = Query(default=False),
    include_globs: list[str] = Query(default_factory=list),
    exclude_globs: list[str] = Query(default_factory=list),
    q: str = Query(default=""),
    sort_dir: SortDirection = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    filter_selection = IngestSelection(
        include_globs=include_globs or [],
        exclude_globs=exclude_globs or [],
        search=q,
    )
    has_browser_filters = bool(q.strip() or filter_selection.include_globs or filter_selection.exclude_globs)
    if is_ingest_records_request(
        status_filter=status_filter,
        recursive=recursive,
        has_filters=has_browser_filters,
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

    directories = []
    if source_kind == "remote" and not source_repo_id and not has_browser_filters and not recursive and not status_filter:
        repos = await source_repo_service.list_source_repositories(db, project)
        directories = [
            {
                "name": repo.name,
                "path": repo.key,
                "source_repo_id": repo.id,
                "source_repo_key": repo.key,
                "source_repo_name": repo.name,
                "kind": "source_repository",
            }
            for repo in sorted(repos, key=lambda item: item.name.lower())
        ]
        return {
            "items": [],
            "directories": directories,
            "dir": dir,
            "recursive": False,
            "total": 0,
            "page": page,
            "page_size": page_size,
            "project_paused": bool(project.ingest_paused),
            "status_counts": {key: 0 for key in ["processing", "queued", "failed", "updated", "not_queued", "compiled"]},
        }

    source_root = None
    source_repo = None
    if source_kind is not None:
        source_root, source_repo = await _resolve_source_context(
            project,
            db,
            source_kind=source_kind,
            source_repo_id=source_repo_id,
        )
    items = await _build_file_items(project, db, source_root=source_root, source_repo=source_repo)
    counts = _empty_status_counts()
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    if source_kind is not None:
        try:
            tree_directories, tree_items = list_source_tree(
                source_root or preferred_source_root(project.disk_path),
                dir_path=dir,
                recursive=recursive,
                source_repo=source_repo,
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
        directories = [asdict(entry) for entry in tree_directories]
        item_status = {item.source_file: item for item in items}
        items = []
        for item in tree_items:
            status_item = item_status.get(item.source_file)
            if status_item is not None:
                status_item.file_size = item.file_size
                items.append(status_item)
            else:
                items.append(item)
    if status_filter:
        items = [item for item in items if item.status == status_filter]
    if include_globs or exclude_globs:
        try:
            items = apply_selection(items, filter_selection)
            q = ""
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    items = filter_and_sort_items(items, search=q, sort_dir=sort_dir)
    page_data = paginate_items(items, page=page, page_size=page_size)
    return {
        "items": [asdict(item) for item in page_data.items],
        "directories": directories,
        "dir": dir,
        "recursive": bool(recursive or has_browser_filters),
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
    source_repo_id: str | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    source_root, _source_repo = await _resolve_source_context(
        project,
        db,
        source_kind=source_kind,
        source_repo_id=source_repo_id,
    )
    target = _resolve_source_file(source_root, source_file)

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    rel = target.resolve().relative_to(source_root.resolve()).as_posix()
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
    return await execute_ingest_enqueue(project_id, body, user.id, db=db)


@router.post("/files/enqueue/tasks", response_model=EnqueueTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_enqueue_ingest_task(
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
    if not body.source_files and body.selection is None and not body.all:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No source files selected")

    file_count = len(body.source_files) if body.source_files else 0
    task = create_enqueue_task(
        project_id,
        file_count=file_count,
        stage=f"已提交 {file_count} 个文件" if file_count else "已提交入队请求",
    )
    asyncio.create_task(run_enqueue_task(task.task_id, project_id, body, user.id))
    return task


@router.get("/files/enqueue/tasks/current", response_model=EnqueueTaskResponse | None)
async def get_current_enqueue_task(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    return get_active_enqueue_task(project_id)


@router.get("/files/enqueue/tasks/{task_id}", response_model=EnqueueTaskResponse)
async def get_enqueue_task_status(
    project_id: str,
    task_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    task = get_enqueue_task(task_id)
    if task is None or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Enqueue task not found")
    return task


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
