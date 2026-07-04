"""Source repository model helpers."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.projects.models import Project, ProjectSourceRepository

DEFAULT_SOURCE_REPO_KEY = "default"
DEFAULT_SOURCE_REPO_NAME = "默认源仓库"
SOURCE_REPO_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def normalize_source_repo_key(value: str) -> str:
    key = value.strip().lower()
    if not SOURCE_REPO_KEY_RE.fullmatch(key):
        raise ValueError("Invalid source repository key")
    return key


def infer_repo_name(repo_url: str) -> str:
    value = repo_url.strip()
    if not value:
        return DEFAULT_SOURCE_REPO_NAME

    parsed = urlparse(value)
    path = parsed.path if parsed.path else value.rsplit(":", 1)[-1]
    name = Path(path.rstrip("/")).name
    if name.endswith(".git"):
        name = name[:-4]
    return name or DEFAULT_SOURCE_REPO_NAME


def source_repo_checkout_root(project: Project, repo: ProjectSourceRepository) -> Path:
    return Path(project.disk_path) / ".llm-wiki" / "source-repos" / repo.key


def legacy_source_repo_checkout_root(project: Project) -> Path:
    return Path(project.disk_path) / ".llm-wiki" / "source-repo"


async def list_source_repositories(
    db: AsyncSession,
    project: Project,
) -> list[ProjectSourceRepository]:
    repos = list(
        (
            await db.execute(
                select(ProjectSourceRepository)
                .where(ProjectSourceRepository.project_id == project.id)
                .order_by(ProjectSourceRepository.created_at.asc(), ProjectSourceRepository.key.asc())
            )
        ).scalars().all()
    )
    if repos:
        return repos
    if not (project.git_repo_url or "").strip():
        return []
    return [await create_default_source_repository(db, project)]


async def create_source_repository(
    db: AsyncSession,
    project: Project,
    *,
    key: str,
    name: str,
    repo_url: str = "",
    branch: str = "main",
    username: str = "",
    auth_token: str = "",
    author_name: str = "",
    author_email: str = "",
    sync_enabled: bool = False,
    auto_compile: bool = False,
    sync_time: str = "02:00",
) -> ProjectSourceRepository:
    try:
        normalized_key = normalize_source_repo_key(key)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid source repository key") from exc

    existing = (
        await db.execute(
            select(ProjectSourceRepository).where(
                ProjectSourceRepository.project_id == project.id,
                ProjectSourceRepository.key == normalized_key,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Source repository key already exists")

    repo = ProjectSourceRepository(
        id=str(uuid.uuid4()),
        project_id=project.id,
        key=normalized_key,
        name=name,
        repo_url=repo_url,
        branch=branch,
        username=username,
        auth_token=auth_token,
        author_name=author_name,
        author_email=author_email,
        sync_enabled=sync_enabled,
        auto_compile=auto_compile,
        sync_time=sync_time,
    )
    db.add(repo)
    await db.commit()
    await db.refresh(repo)
    return repo


async def update_source_repository(
    db: AsyncSession,
    repo: ProjectSourceRepository,
    **fields,
) -> ProjectSourceRepository:
    fields.pop("key", None)
    for field_name, value in fields.items():
        if value is not None:
            setattr(repo, field_name, value)

    await db.commit()
    await db.refresh(repo)
    return repo


async def delete_source_repository(
    db: AsyncSession,
    project: Project,
    repo_id: str,
) -> None:
    repo = await get_source_repository_or_404(db, project, repo_id)
    await db.delete(repo)
    await db.commit()


async def create_default_source_repository(
    db: AsyncSession,
    project: Project,
) -> ProjectSourceRepository:
    existing = (
        await db.execute(
            select(ProjectSourceRepository).where(
                ProjectSourceRepository.project_id == project.id,
                ProjectSourceRepository.key == DEFAULT_SOURCE_REPO_KEY,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    repo = ProjectSourceRepository(
        id=str(uuid.uuid4()),
        project_id=project.id,
        key=DEFAULT_SOURCE_REPO_KEY,
        name=infer_repo_name(project.git_repo_url or ""),
        repo_url=project.git_repo_url or "",
        branch=project.git_branch or "main",
        username=project.git_username or "",
        auth_token=project.git_auth_token or "",
        author_name=project.git_author_name or "",
        author_email=project.git_author_email or "",
        sync_enabled=bool(project.git_sync_enabled),
        auto_compile=bool(project.git_sync_auto_compile),
        sync_time=project.git_sync_time or "02:00",
        last_sync_at=project.last_git_sync_at,
        last_sync_status=project.last_git_sync_status or "idle",
        last_sync_error=project.last_git_sync_error or "",
    )
    db.add(repo)
    await db.commit()
    await db.refresh(repo)
    return repo


async def sync_default_source_repository_from_project(
    db: AsyncSession,
    project: Project,
) -> ProjectSourceRepository:
    """Create or update the legacy default source repo from project Git fields."""
    repo = (
        await db.execute(
            select(ProjectSourceRepository).where(
                ProjectSourceRepository.project_id == project.id,
                ProjectSourceRepository.key == DEFAULT_SOURCE_REPO_KEY,
            )
        )
    ).scalar_one_or_none()
    if repo is None:
        return await create_default_source_repository(db, project)

    repo.name = infer_repo_name(project.git_repo_url or "")
    repo.repo_url = project.git_repo_url or ""
    repo.branch = project.git_branch or "main"
    repo.username = project.git_username or ""
    repo.auth_token = project.git_auth_token or ""
    repo.author_name = project.git_author_name or ""
    repo.author_email = project.git_author_email or ""
    repo.sync_enabled = bool(project.git_sync_enabled)
    repo.auto_compile = bool(project.git_sync_auto_compile)
    repo.sync_time = project.git_sync_time or "02:00"
    await db.commit()
    await db.refresh(repo)
    return repo


async def get_source_repository_or_404(
    db: AsyncSession,
    project: Project,
    repo_id: str,
) -> ProjectSourceRepository:
    repo = (
        await db.execute(
            select(ProjectSourceRepository).where(
                ProjectSourceRepository.project_id == project.id,
                ProjectSourceRepository.id == repo_id,
            )
        )
    ).scalar_one_or_none()
    if repo is not None:
        return repo
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Source repository not found")
