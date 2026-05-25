"""Project CRUD and permission helpers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import User
from app.config import get_config
from app.projects.models import Project, ProjectMember

_WIKI_DIRS = [
    "wiki/entities",
    "wiki/concepts",
    "wiki/sources",
    "wiki/queries",
    "wiki/comparisons",
    "wiki/synthesis",
    "raw/sources",
    ".llm-wiki",
]


async def create_project(
    db: AsyncSession,
    *,
    name: str,
    slug: str,
    description: str,
    user: User,
) -> Project:
    exists = (await db.execute(select(Project).where(Project.slug == slug))).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Slug already taken")

    project_id = str(uuid.uuid4())
    base_dir = Path(get_config().server.projects_dir) / project_id
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in _WIKI_DIRS:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)

    (base_dir / "purpose.md").write_text(f"# {name}\n\n{description}\n", encoding="utf-8")
    (base_dir / "wiki" / "index.md").write_text(f"# {name} - Index\n", encoding="utf-8")
    (base_dir / "wiki" / "overview.md").write_text(f"# {name} - Overview\n", encoding="utf-8")
    (base_dir / "wiki" / "log.md").write_text("# Compilation Log\n", encoding="utf-8")
    (base_dir / ".llm-wiki" / "project.json").write_text(
        json.dumps({"id": project_id, "name": name}), encoding="utf-8"
    )
    (base_dir / ".llm-wiki" / "ingest-cache.json").write_text("{}", encoding="utf-8")
    (base_dir / ".llm-wiki" / "ingest-queue.json").write_text("[]", encoding="utf-8")

    proj = Project(
        id=project_id,
        name=name,
        slug=slug,
        description=description,
        disk_path=str(base_dir),
        created_by=user.id,
    )
    db.add(proj)
    member = ProjectMember(project_id=project_id, user_id=user.id, role="owner")
    db.add(member)
    await db.commit()
    await db.refresh(proj)
    return proj


async def list_user_projects(db: AsyncSession, user: User) -> list[Project]:
    stmt = (
        select(Project)
        .join(ProjectMember, Project.id == ProjectMember.project_id)
        .where(ProjectMember.user_id == user.id)
        .order_by(Project.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_project_or_404(db: AsyncSession, project_id: str) -> Project:
    proj = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if proj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return proj


async def check_membership(db: AsyncSession, project_id: str, user: User, *, require: str | None = None) -> ProjectMember:
    stmt = select(ProjectMember).where(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user.id,
    )
    member = (await db.execute(stmt)).scalar_one_or_none()
    if member is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not a project member")
    if require and member.role != require and member.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires {require} role")
    return member


async def add_member(db: AsyncSession, project_id: str, user_id: int, role: str = "editor") -> ProjectMember:
    existing = (
        await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.role = role
        await db.commit()
        return existing
    member = ProjectMember(project_id=project_id, user_id=user_id, role=role)
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return member


async def delete_project(db: AsyncSession, project: Project) -> None:
    import shutil

    disk = Path(project.disk_path)
    if disk.exists():
        shutil.rmtree(disk, ignore_errors=True)
    await db.delete(project)
    await db.commit()
