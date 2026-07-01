"""Document upload and listing service."""

from __future__ import annotations

from pathlib import Path

import aiofiles

from app.projects.models import Project


async def save_uploaded_file(project: Project, filename: str, content: bytes) -> Path:
    target_dir = Path(project.disk_path) / "raw" / "sources"
    dest = target_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)
    return dest


def read_document_content(project: Project, doc_path: str) -> str | None:
    """Read document text content by relative path (under raw/sources/)."""
    source_dir = Path(project.disk_path) / "raw" / "sources"
    target = (source_dir / doc_path).resolve()
    # Path traversal guard
    if not str(target).startswith(str(source_dir.resolve())):
        return None
    if not target.is_file():
        return None
    from app.documents.parser import parse_document
    return parse_document(target)


def list_documents(project: Project) -> list[dict]:
    source_dir = Path(project.disk_path) / "raw" / "sources"
    if not source_dir.exists():
        return []
    docs = []
    for f in sorted(source_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = f.relative_to(source_dir).as_posix()
            docs.append({
                "name": rel,
                "path": f.relative_to(project.disk_path).as_posix(),
                "size": f.stat().st_size,
            })
    return docs
