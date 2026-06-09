"""Per-source ingest file status helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

IngestFileStatus = Literal["processing", "queued", "failed", "updated", "not_queued", "compiled"]
SortDirection = Literal["asc", "desc"]

_STATUS_PRIORITY: dict[IngestFileStatus, int] = {
    "processing": 0,
    "queued": 1,
    "failed": 2,
    "updated": 3,
    "not_queued": 4,
    "compiled": 5,
}


@dataclass
class IngestFileItem:
    source_file: str
    source_path: str
    status: IngestFileStatus
    job_id: str | None = None
    job_status: str | None = None
    progress: str = ""
    step: int = 0
    files_written: list[str] | None = None
    error: str | None = None
    retry_count: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class IngestFilePage:
    items: list[IngestFileItem] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 10


def source_identity(path: str, source_root: str | None = None) -> str:
    source_path = Path(path)
    if source_root:
        try:
            return source_path.resolve().relative_to(Path(source_root).resolve()).as_posix()
        except ValueError:
            pass
    return source_path.name


def _job_file_status(job_status: str) -> IngestFileStatus | None:
    if job_status == "processing":
        return "processing"
    if job_status in ("pending", "paused"):
        return "queued"
    if job_status == "failed":
        return "failed"
    return None


def _job_sort_key(job) -> datetime:
    return job.created_at or job.completed_at or datetime.min


def resolve_file_statuses(
    *,
    source_paths: list[str],
    jobs: list,
    changed_identities: set[str],
    cached_identities: set[str],
    source_root: str | None = None,
) -> list[IngestFileItem]:
    """Build one display status per source file.

    Active job state wins over content state. A file whose cached content hash
    differs from the current source is "updated"; a file never seen in cache is
    "not_queued"; an unchanged cached file is "compiled".
    """
    latest_job_by_identity: dict[str, object] = {}
    for job in sorted(jobs, key=_job_sort_key):
        identity = source_identity(job.source_path, source_root)
        current = latest_job_by_identity.get(identity)
        if current is None:
            latest_job_by_identity[identity] = job
            continue
        current_status = _job_file_status(current.status)
        next_status = _job_file_status(job.status)
        if next_status is not None and (
            current_status is None or _STATUS_PRIORITY[next_status] < _STATUS_PRIORITY[current_status]
        ):
            latest_job_by_identity[identity] = job
        elif current_status is None and _job_sort_key(job) >= _job_sort_key(current):
            latest_job_by_identity[identity] = job

    items: list[IngestFileItem] = []
    seen: set[str] = set()
    for source_path in sorted(source_paths, key=lambda p: source_identity(p, source_root).lower()):
        identity = source_identity(source_path, source_root)
        if identity in seen:
            continue
        seen.add(identity)

        job = latest_job_by_identity.get(identity)
        job_status = _job_file_status(job.status) if job else None
        if job_status:
            status = job_status
        elif identity in changed_identities:
            status = "updated"
        elif identity in cached_identities:
            status = "compiled"
        else:
            status = "not_queued"

        items.append(
            IngestFileItem(
                source_file=identity,
                source_path=source_path,
                status=status,
                job_id=getattr(job, "id", None) if job else None,
                job_status=getattr(job, "status", None) if job else None,
                progress=getattr(job, "progress", "") if job else "",
                step=getattr(job, "step", 0) if job else 0,
                files_written=getattr(job, "files_written", None) if job else None,
                error=getattr(job, "error", None) if job else None,
                retry_count=getattr(job, "retry_count", 0) if job else 0,
                created_at=getattr(job, "created_at", None) if job else None,
                completed_at=getattr(job, "completed_at", None) if job else None,
            )
        )

    return items


def paginate_items(items: list[IngestFileItem], *, page: int = 1, page_size: int = 10) -> IngestFilePage:
    safe_page = max(1, page)
    safe_page_size = min(100, max(1, page_size))
    start = (safe_page - 1) * safe_page_size
    return IngestFilePage(
        items=items[start:start + safe_page_size],
        total=len(items),
        page=safe_page,
        page_size=safe_page_size,
    )


def filter_and_sort_items(
    items: list[IngestFileItem],
    *,
    search: str = "",
    sort_dir: SortDirection = "asc",
) -> list[IngestFileItem]:
    needle = search.strip().lower()
    filtered = [
        item for item in items
        if not needle or needle in item.source_file.lower()
    ]
    return sorted(
        filtered,
        key=lambda item: item.source_file.lower(),
        reverse=sort_dir == "desc",
    )
