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
    return [await create_default_source_repository(db, project)]


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
        name=DEFAULT_SOURCE_REPO_NAME,
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


async def get_source_repository_or_404(
    db: AsyncSession,
    project: Project,
    repo_key: str,
) -> ProjectSourceRepository:
    key = normalize_source_repo_key(repo_key)
    repo = (
        await db.execute(
            select(ProjectSourceRepository).where(
                ProjectSourceRepository.project_id == project.id,
                ProjectSourceRepository.key == key,
            )
        )
    ).scalar_one_or_none()
    if repo is not None:
        return repo
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Source repository not found")
