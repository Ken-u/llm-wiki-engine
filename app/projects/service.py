"""Project CRUD and permission helpers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataclasses import dataclass

from app.auth.models import User
from app.config import get_config
from app.projects.models import Project, ProjectMember


@dataclass
class _AdminMember:
    """Lightweight stand-in for ProjectMember when admin bypasses membership."""
    project_id: str
    user_id: int
    role: str

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


def project_dir(project_id: str) -> Path:
    return Path(get_config().server.projects_dir) / project_id


async def create_project(
    db: AsyncSession,
    *,
    name: str,
    slug: str,
    description: str,
    user: User,
    case_library_main_project_id: str | None = None,
) -> Project:
    exists = (await db.execute(select(Project).where(Project.slug == slug))).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Slug already taken")

    main_project: Project | None = None
    if case_library_main_project_id:
        await check_membership(db, case_library_main_project_id, user, require="owner")
        main_project = (
            await db.execute(select(Project).where(Project.id == case_library_main_project_id))
        ).scalar_one_or_none()
        if main_project is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Main project not found")
        if main_project.ticket_project_id is not None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Main project already has a case library binding")

    project_id = str(uuid.uuid4())
    base_dir = project_dir(project_id)
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
        _disk_path="",
        created_by=user.id,
    )
    db.add(proj)
    member = ProjectMember(project_id=project_id, user_id=user.id, role="owner")
    db.add(member)
    if main_project:
        main_project.ticket_project_id = project_id
    await db.commit()
    await db.refresh(proj)
    return proj


async def list_user_projects(db: AsyncSession, user: User) -> list[Project]:
    if user.role == "admin":
        stmt = select(Project).order_by(Project.created_at.desc())
    else:
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


async def check_membership(db: AsyncSession, project_id: str, user: User, *, require: str | None = None) -> ProjectMember | _AdminMember:
    if user.role == "admin":
        return _AdminMember(project_id=project_id, user_id=user.id, role="owner")

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


async def update_project(
    db: AsyncSession,
    project: Project,
    *,
    name: str | None = None,
    description: str | None = None,
    ticket_project_id: str | None = ...,
    feedback_enabled: bool | None = None,
    git_repo_url: str | None = None,
    git_branch: str | None = None,
    git_username: str | None = None,
    git_auth_token: str | None = None,
    git_author_name: str | None = None,
    git_author_email: str | None = None,
    git_sync_enabled: bool | None = None,
    git_sync_time: str | None = None,
) -> Project:
    """Update project fields. Pass ticket_project_id=None to clear binding."""
    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if feedback_enabled is not None:
        project.feedback_enabled = feedback_enabled

    if ticket_project_id is not ...:
        if ticket_project_id is not None:
            if ticket_project_id == project.id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot bind project to itself as ticket project")
            target = (await db.execute(select(Project).where(Project.id == ticket_project_id))).scalar_one_or_none()
            if target is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket project not found")
            if target.ticket_project_id is not None:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ticket project cannot itself have a ticket binding (no recursive chains)")
        project.ticket_project_id = ticket_project_id

    git_simple_fields = {
        "git_repo_url": git_repo_url,
        "git_branch": git_branch,
        "git_username": git_username,
        "git_auth_token": git_auth_token,
        "git_author_name": git_author_name,
        "git_author_email": git_author_email,
        "git_sync_enabled": git_sync_enabled,
        "git_sync_time": git_sync_time,
    }
    for field_name, value in git_simple_fields.items():
        if value is not None:
            setattr(project, field_name, value)

    await db.commit()
    await db.refresh(project)
    return project


async def delete_project(db: AsyncSession, project: Project) -> None:
    import shutil

    disk = Path(project.disk_path)
    if disk.exists():
        shutil.rmtree(disk, ignore_errors=True)
    await db.delete(project)
    await db.commit()
