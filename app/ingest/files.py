"""Per-source ingest file status helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Literal

from app.documents.parser import parse_document
from app.ingest.cache import content_hash

IngestFileStatus = Literal["processing", "queued", "failed", "updated", "not_queued", "compiled"]
INGEST_RECORD_STATUSES = frozenset({"processing", "queued", "failed", "updated", "compiled"})
SortDirection = Literal["asc", "desc"]
SourceKind = Literal["remote", "local"]

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
class IngestRecordPageResult:
    items: list[IngestFileItem]
    counts: dict[str, int]
    total: int
    page: int
    page_size: int


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


def local_source_root(project_dir: str) -> Path:
    return Path(project_dir) / "raw" / "sources"


def browser_source_root(project_dir: str, source_kind: SourceKind) -> Path:
    if source_kind == "remote":
        return provider_source_root(project_dir)
    return local_source_root(project_dir)


def list_source_files_at_root(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    files = [
        path for path in source_root.rglob("*")
        if is_project_source_file(path, source_root)
    ]
    return sorted(
        files,
        key=lambda path: path.resolve().relative_to(source_root.resolve()).as_posix().lower(),
    )


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


def is_project_source_dir(path: Path, source_root: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
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


def stage_sources_for_ingest(
    project_dir: str,
    pairs: list[tuple[str, str]],
    *,
    on_item: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Copy selected files into raw/sources, persisting source-map once."""
    if not pairs:
        return []
    source_map = load_source_map(project_dir)
    raw_root = Path(project_dir) / "raw" / "sources"
    results: list[Path] = []
    dirty = False
    total = len(pairs)
    for index, (source_path, source_root) in enumerate(pairs, start=1):
        src = Path(source_path)
        root = Path(source_root)
        rel = src.resolve().relative_to(root.resolve()).as_posix()
        if on_item:
            on_item(index, total, rel)
        dest = raw_root / rel
        if src.resolve() == dest.resolve():
            results.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        source_map[rel] = {
            "source_path": rel,
            "raw_source_path": f"raw/sources/{rel}",
            "sha256": _file_sha256(src),
            "copied_at": datetime.now(timezone.utc).isoformat(),
        }
        dirty = True
        results.append(dest)
    if dirty:
        save_source_map(project_dir, source_map)
    return results


def stage_source_for_ingest(project_dir: str, source_path: str, source_root: str) -> Path:
    """Copy a selected provider file into raw/sources and record its origin."""
    return stage_sources_for_ingest(project_dir, [(source_path, source_root)])[0]


def resolve_browser_source_files(source_root: Path, source_files: list[str]) -> list[Path]:
    """Resolve explicit browser selections without scanning the whole tree."""
    resolved: list[Path] = []
    for source_file in source_files:
        src = (source_root / source_file).resolve()
        try:
            src.relative_to(source_root.resolve())
        except ValueError:
            raise ValueError(f"Invalid source file: {source_file}") from None
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(source_file)
        if not is_project_source_file(src, source_root):
            raise ValueError(f"Invalid source file: {source_file}")
        resolved.append(src)
    return resolved


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


def is_ingest_records_request(
    *,
    status_filter: IngestFileStatus | None,
    recursive: bool,
    has_filters: bool,
    dir: str,
) -> bool:
    """Compile-page tabs: list from jobs/cache only, never scan the full source tree."""
    return (
        status_filter is not None
        and status_filter in INGEST_RECORD_STATUSES
        and not recursive
        and not has_filters
        and not dir.strip()
    )


def detect_changed_identities(source_dir: Path, cache: dict[str, str]) -> set[str]:
    """Compare cached hashes against staged raw/sources files (not the whole repo)."""
    changed: set[str] = set()
    for identity in cache:
        src = source_dir / identity
        if not src.is_file():
            continue
        cache_key = identity if identity in cache else src.name
        try:
            if content_hash(parse_document(src)) != cache[cache_key]:
                changed.add(identity)
        except Exception:
            changed.add(identity)
    return changed


def _job_source_paths(jobs: list) -> list[str]:
    return list({job.source_path for job in jobs})


def build_ingest_record_page(
    *,
    project_dir: str,
    jobs: list,
    cache: dict[str, str],
    status_filter: IngestFileStatus,
    sort_dir: SortDirection,
    page: int,
    page_size: int,
) -> IngestRecordPageResult:
    """Fast path for compile-record tabs: jobs + ingest cache, no provider checkout scan."""
    source_dir = local_source_root(project_dir)
    cached = set(cache.keys())
    job_paths = _job_source_paths(jobs)
    changed = detect_changed_identities(source_dir, cache)

    count_paths = list(
        set(job_paths)
        | {
            str((source_dir / identity).resolve())
            for identity in cached
            if (source_dir / identity).is_file()
        }
    )
    all_items = resolve_file_statuses(
        source_paths=count_paths,
        jobs=jobs,
        changed_identities=changed,
        cached_identities=cached,
        source_root=str(source_dir),
    )
    counts = {key: 0 for key in ["processing", "queued", "failed", "updated", "not_queued", "compiled"]}
    for item in all_items:
        counts[item.status] = counts.get(item.status, 0) + 1

    if status_filter in ("processing", "queued", "failed"):
        page_source_paths = job_paths
        page_changed: set[str] = set()
    elif status_filter == "updated":
        page_source_paths = [
            str((source_dir / identity).resolve())
            for identity in sorted(changed)
            if (source_dir / identity).is_file()
        ]
        page_changed = changed
    else:
        page_source_paths = [
            str((source_dir / identity).resolve())
            for identity in sorted(cached)
            if identity not in changed and (source_dir / identity).is_file()
        ]
        page_changed = changed

    page_items = resolve_file_statuses(
        source_paths=page_source_paths,
        jobs=jobs,
        changed_identities=page_changed,
        cached_identities=cached,
        source_root=str(source_dir),
    )
    page_items = [item for item in page_items if item.status == status_filter]
    page_items = filter_and_sort_items(page_items, sort_dir=sort_dir)
    page_data = paginate_items(page_items, page=page, page_size=page_size)
    return IngestRecordPageResult(
        items=page_data.items,
        counts=counts,
        total=page_data.total,
        page=page_data.page,
        page_size=page_data.page_size,
    )


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

        source = Path(source_path)
        file_size = source.stat().st_size if source.is_file() else None
        items.append(
            IngestFileItem(
                source_file=identity,
                source_path=source_path,
                file_size=file_size,
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
