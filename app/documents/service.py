"""Document upload and listing service."""

from __future__ import annotations

import shutil
from pathlib import Path

import aiofiles

from app.projects.models import Project


async def save_uploaded_file(project: Project, filename: str, content: bytes) -> Path:
    target_dir = Path(project.disk_path) / "raw" / "sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)
    return dest


def list_documents(project: Project) -> list[dict]:
    source_dir = Path(project.disk_path) / "raw" / "sources"
    if not source_dir.exists():
        return []
    docs = []
    for f in sorted(source_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            docs.append({
                "name": f.name,
                "path": str(f.relative_to(project.disk_path)),
                "size": f.stat().st_size,
            })
    return docs
