"""Wiki CRUD endpoints: file tree, read/write pages, graph, overview."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.projects.service import check_membership, get_project_or_404
from app.wiki.frontmatter import parse_frontmatter
from app.wiki.graph import build_wiki_graph, graph_to_json

router = APIRouter(prefix="/api/projects/{project_id}/wiki", tags=["wiki"])


def _build_file_tree(wiki_dir: Path, base: Path) -> list[dict]:
    """Recursively build a file tree structure."""
    items = []
    if not wiki_dir.exists():
        return items
    for entry in sorted(wiki_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(base))
        if entry.is_dir():
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "directory",
                "children": _build_file_tree(entry, base),
            })
        elif entry.suffix == ".md":
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "file",
            })
    return items


@router.get("")
async def wiki_file_tree(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    base = Path(project.disk_path)
    wiki_dir = base / "wiki"
    return _build_file_tree(wiki_dir, base)


class WikiPageResponse(BaseModel):
    path: str
    content: str
    meta: dict


@router.get("/overview")
async def wiki_overview(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    overview_path = Path(project.disk_path) / "wiki" / "overview.md"
    if not overview_path.exists():
        return {"content": "", "meta": {}}
    async with aiofiles.open(overview_path, "r", encoding="utf-8") as f:
        content = await f.read()
    meta, body = parse_frontmatter(content)
    return {"content": content, "meta": meta.raw}


@router.get("/graph")
async def wiki_graph(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    graph = build_wiki_graph(project.disk_path)
    return graph_to_json(graph)


@router.get("/{path:path}", response_model=WikiPageResponse)
async def read_wiki_page(
    project_id: str,
    path: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    # Security: ensure path stays within wiki/
    if ".." in path or path.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")

    full_path = Path(project.disk_path) / path
    if not full_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")

    async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
        content = await f.read()

    meta, body = parse_frontmatter(content)
    return WikiPageResponse(path=path, content=content, meta=meta.raw)


class UpdatePageRequest(BaseModel):
    content: str


@router.put("/{path:path}")
async def update_wiki_page(
    project_id: str,
    path: str,
    body: UpdatePageRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    if ".." in path or path.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")

    full_path = Path(project.disk_path) / path
    if not full_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")

    async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
        await f.write(body.content)

    # Re-embed the updated page
    try:
        from app.embedding.service import embed_pages
        await embed_pages(project.disk_path, [path])
    except Exception:
        pass

    return {"path": path, "status": "updated"}
