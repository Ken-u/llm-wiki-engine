"""Per-source ingest file status helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
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

_ACTIVE_JOB_STATUSES = {"processing", "pending", "paused"}
_EXCLUDED_REPO_SOURCE_DIRS = {
    ".git",
    ".llm-wiki",
    "__pycache__",
    "node_modules",
    "wiki",
}


@dataclass
class IngestFileItem:
    source_file: str
    source_path: str
    status: IngestFileStatus
    file_size: int | None = None
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


@dataclass
class IngestFilePage:
    items: list[IngestFileItem] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 10


@dataclass
class SourceDirectoryEntry:
    name: str
    path: str
    source_repo_id: str | None = None
    source_repo_key: str | None = None
    source_repo_name: str | None = None
    kind: str = "directory"


@dataclass
class SourceRepoMetadata:
    id: str
    key: str
    name: str


@dataclass
class IngestSelection:
    statuses: list[IngestFileStatus] = field(default_factory=list)
    include_globs: list[str] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    search: str = ""
    limit: int | None = None


def source_identity(path: str, source_root: str | None = None) -> str:
    source_path = Path(path)
    if source_root:
        try:
            return source_path.resolve().relative_to(Path(source_root).resolve()).as_posix()
        except ValueError:
            pass
    parts = source_path.parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "raw" and parts[idx + 1] == "sources":
            return Path(*parts[idx + 2:]).as_posix()
    return source_path.name


def provider_source_root(project_dir: str) -> Path:
    return Path(project_dir) / ".llm-wiki" / "source-repo"


def preferred_source_root(project_dir: str) -> Path:
    """Return provider checkout when populated, then raw/sources, then project root.

    Older projects store compile inputs under raw/sources. Git-synced
    repositories are cloned under .llm-wiki/source-repo and should be the file
    tree users select from before selected files are copied into raw/sources.
    """
    project_root = Path(project_dir)
    provider_root = provider_source_root(project_dir)
    if provider_root.exists() and any(
        path.is_file() and not path.name.startswith(".")
        for path in provider_root.rglob("*")
    ):
        return provider_root
    raw_sources = project_root / "raw" / "sources"
    if raw_sources.exists() and any(
        path.is_file() and not path.name.startswith(".")
        for path in raw_sources.rglob("*")
    ):
        return raw_sources
    return project_root


def is_project_source_file(path: Path, source_root: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    try:
        rel = path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return False
    return not any(part in _EXCLUDED_REPO_SOURCE_DIRS for part in rel.parts)


def list_project_source_files(project_dir: str) -> tuple[Path, list[Path]]:
    source_root = preferred_source_root(project_dir)
    if not source_root.exists():
        return source_root, []
    files = [
        path for path in source_root.rglob("*")
        if is_project_source_file(path, source_root)
    ]
    return source_root, sorted(
        files,
        key=lambda path: path.resolve().relative_to(source_root.resolve()).as_posix().lower(),
    )


def raw_source_root(project_dir: str) -> Path:
    return Path(project_dir) / "raw" / "sources"


def list_source_tree(
    source_root: Path,
    *,
    dir_path: str = "",
    recursive: bool = False,
    source_repo: SourceRepoMetadata | None = None,
) -> tuple[list[SourceDirectoryEntry], list[IngestFileItem]]:
    root = source_root.resolve()
    normalized_dir = dir_path.strip().replace("\\", "/")
    if normalized_dir in (".", "/"):
        normalized_dir = ""
    if normalized_dir.startswith("/") or normalized_dir.startswith("../") or "/../" in normalized_dir:
        raise ValueError(f"Invalid directory: {dir_path}")

    current = (root / normalized_dir).resolve()
    try:
        current.relative_to(root)
    except ValueError:
        raise ValueError(f"Invalid directory: {dir_path}") from None
    if not current.exists():
        return [], []
    if not current.is_dir():
        raise ValueError(f"Invalid directory: {dir_path}")

    directories: list[SourceDirectoryEntry] = []
    if not recursive:
        for child in current.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            rel = child.resolve().relative_to(root).as_posix()
            if any(part in _EXCLUDED_REPO_SOURCE_DIRS for part in Path(rel).parts):
                continue
            directories.append(
                SourceDirectoryEntry(
                    name=child.name,
                    path=rel,
                    source_repo_id=source_repo.id if source_repo else None,
                    source_repo_key=source_repo.key if source_repo else None,
                    source_repo_name=source_repo.name if source_repo else None,
                )
            )

    candidates = current.rglob("*") if recursive else current.iterdir()
    files: list[IngestFileItem] = []
    for path in candidates:
        if not is_project_source_file(path, root):
            continue
        rel = path.resolve().relative_to(root).as_posix()
        files.append(
            IngestFileItem(
                source_file=rel,
                source_path=str(path),
                status="not_queued",
                file_size=path.stat().st_size,
                source_repo_id=source_repo.id if source_repo else None,
                source_repo_key=source_repo.key if source_repo else None,
                source_repo_name=source_repo.name if source_repo else None,
            )
        )

    directories.sort(key=lambda entry: entry.name.lower())
    files.sort(key=lambda item: item.source_file.lower())
    return directories, files


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_map_path(project_dir: str) -> Path:
    return Path(project_dir) / ".llm-wiki" / "source-map.json"


def load_source_map(project_dir: str) -> dict[str, dict[str, str]]:
    path = _source_map_path(project_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_source_map(project_dir: str, source_map: dict[str, dict[str, str]]) -> None:
    path = _source_map_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_source_for_ingest(
    project_dir: str,
    source_path: str,
    source_root: str,
    source_repo: SourceRepoMetadata | None = None,
) -> Path:
    """Copy a selected provider file into raw/sources and record its origin."""
    src = Path(source_path)
    root = Path(source_root)
    rel = src.resolve().relative_to(root.resolve()).as_posix()
    map_key = f"{source_repo.key}/{rel}" if source_repo else rel
    raw_root = raw_source_root(project_dir)
    dest = raw_root / map_key
    if src.resolve() == dest.resolve():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    source_map = load_source_map(project_dir)
    source_map[map_key] = {
        "source_path": rel,
        "raw_source_path": f"raw/sources/{map_key}",
        "sha256": _file_sha256(src),
        "copied_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_repo:
        source_map[map_key]["source_repo_id"] = source_repo.id
        source_map[map_key]["source_repo_key"] = source_repo.key
        source_map[map_key]["source_repo_name"] = source_repo.name
    save_source_map(project_dir, source_map)
    return dest


def validate_source_pattern(pattern: str) -> str:
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Empty source pattern")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise ValueError(f"Invalid source pattern: {pattern}")
    return normalized


def _matches_glob(identity: str, pattern: str) -> bool:
    normalized = validate_source_pattern(pattern)
    if fnmatchcase(identity, normalized):
        return True
    if "/" not in normalized and fnmatchcase(Path(identity).name, normalized):
        return True
    if normalized.endswith("/"):
        return identity.startswith(normalized)
    return False


def apply_selection(items: list[IngestFileItem], selection: IngestSelection) -> list[IngestFileItem]:
    """Filter source items with status/search/glob rules."""
    selected = items
    if selection.statuses:
        allowed = set(selection.statuses)
        selected = [item for item in selected if item.status in allowed]

    needle = selection.search.strip().lower()
    if needle:
        selected = [item for item in selected if needle in item.source_file.lower()]

    includes = [validate_source_pattern(p) for p in selection.include_globs if p.strip()]
    excludes = [validate_source_pattern(p) for p in selection.exclude_globs if p.strip()]

    if includes:
        selected = [
            item for item in selected
            if any(_matches_glob(item.source_file, pattern) for pattern in includes)
        ]
    if excludes:
        selected = [
            item for item in selected
            if not any(_matches_glob(item.source_file, pattern) for pattern in excludes)
        ]

    selected = sorted(selected, key=lambda item: item.source_file.lower())
    if selection.limit is not None:
        selected = selected[:max(0, selection.limit)]
    return selected


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

        current_active = current.status in _ACTIVE_JOB_STATUSES
        next_active = job.status in _ACTIVE_JOB_STATUSES
        if not current_active and not next_active:
            latest_job_by_identity[identity] = job
            continue
        if current_active != next_active:
            if next_active:
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
