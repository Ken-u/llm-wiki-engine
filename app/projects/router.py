"""Project management endpoints."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.projects import service
from app.projects.models import Project, ProjectMember
from app.projects import source_repositories as source_repo_service

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    slug: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    as_case_library: bool = False
    main_project_id: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str
    created_by: int
    created_at: datetime
    ticket_project_id: str | None = None
    ticket_project_name: str | None = None
    feedback_enabled: bool = True
    git_repo_url: str = ""
    git_branch: str = "main"
    git_username: str = ""
    git_author_name: str = ""
    git_author_email: str = ""
    git_sync_enabled: bool = False
    git_sync_auto_compile: bool = False
    git_sync_time: str = "02:00"
    git_auth_configured: bool = False
    last_git_sync_at: datetime | None = None
    last_git_sync_status: str = "idle"
    last_git_sync_error: str = ""
    publish_repo_url: str = ""
    publish_branch: str = "main"
    publish_username: str = ""
    publish_author_name: str = ""
    publish_author_email: str = ""
    publish_enabled: bool = False
    publish_auth_configured: bool = False
    last_publish_at: datetime | None = None
    last_publish_status: str = "idle"
    last_publish_error: str = ""
    ingest_paused: bool = False
    project_type: str = "knowledge_base"
    case_index_auto_rebuild: bool = False

    class Config:
        from_attributes = True


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    ticket_project_id: str | None = None
    feedback_enabled: bool | None = None
    git_repo_url: str | None = None
    git_branch: str | None = None
    git_username: str | None = None
    git_auth_token: str | None = None
    clear_git_auth_token: bool | None = None
    git_author_name: str | None = None
    git_author_email: str | None = None
    git_sync_enabled: bool | None = None
    git_sync_auto_compile: bool | None = None
    git_sync_time: str | None = None
    publish_repo_url: str | None = None
    publish_branch: str | None = None
    publish_username: str | None = None
    publish_auth_token: str | None = None
    clear_publish_auth_token: bool | None = None
    publish_author_name: str | None = None
    publish_author_email: str | None = None
    publish_enabled: bool | None = None
    case_index_auto_rebuild: bool | None = None
    ingest_paused: bool | None = None


class SourceRepositoryCreateRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    repo_url: str = ""
    branch: str = "main"
    username: str = ""
    auth_token: str = ""
    author_name: str = ""
    author_email: str = ""
    sync_enabled: bool = False
    auto_compile: bool = False
    sync_time: str = "02:00"


class SourceRepositoryUpdateRequest(BaseModel):
    name: str | None = None
    repo_url: str | None = None
    branch: str | None = None
    username: str | None = None
    auth_token: str | None = None
    clear_auth_token: bool | None = None
    author_name: str | None = None
    author_email: str | None = None
    sync_enabled: bool | None = None
    auto_compile: bool | None = None
    sync_time: str | None = None


class SourceRepositoryResponse(BaseModel):
    id: str
    project_id: str
    key: str
    name: str
    repo_url: str = ""
    branch: str = "main"
    username: str = ""
    author_name: str = ""
    author_email: str = ""
    sync_enabled: bool = False
    auto_compile: bool = False
    sync_time: str = "02:00"
    last_sync_at: datetime | None = None
    last_sync_status: str = "idle"
    last_sync_error: str = ""
    auth_configured: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TestGitConnectionRequest(BaseModel):
    git_repo_url: str | None = None
    git_branch: str | None = None
    git_username: str | None = None
    git_auth_token: str | None = None


class TestPublishConnectionRequest(BaseModel):
    publish_repo_url: str | None = None
    publish_branch: str | None = None
    publish_username: str | None = None
    publish_auth_token: str | None = None


class AddMemberRequest(BaseModel):
    user_id: int
    role: str = Field(default="editor", pattern=r"^(owner|editor|viewer)$")


class MemberResponse(BaseModel):
    user_id: int
    role: str

    class Config:
        from_attributes = True


async def _build_project_response(db: AsyncSession, proj: Project) -> ProjectResponse:
    """Build ProjectResponse with optional ticket project name."""
    ticket_name = None
    if proj.ticket_project_id:
        ticket = (await db.execute(
            select(Project.name).where(Project.id == proj.ticket_project_id)
        )).scalar_one_or_none()
        ticket_name = ticket
    return ProjectResponse(
        id=proj.id,
        name=proj.name,
        slug=proj.slug,
        description=proj.description,
        created_by=proj.created_by,
        created_at=proj.created_at,
        ticket_project_id=proj.ticket_project_id,
        ticket_project_name=ticket_name,
        feedback_enabled=proj.feedback_enabled,
        git_repo_url=proj.git_repo_url,
        git_branch=proj.git_branch,
        git_username=proj.git_username,
        git_author_name=proj.git_author_name,
        git_author_email=proj.git_author_email,
        git_sync_enabled=proj.git_sync_enabled,
        git_sync_auto_compile=proj.git_sync_auto_compile,
        git_sync_time=proj.git_sync_time,
        git_auth_configured=bool(proj.git_auth_token),
        last_git_sync_at=proj.last_git_sync_at,
        last_git_sync_status=proj.last_git_sync_status,
        last_git_sync_error=proj.last_git_sync_error,
        publish_repo_url=proj.publish_repo_url,
        publish_branch=proj.publish_branch,
        publish_username=proj.publish_username,
        publish_author_name=proj.publish_author_name,
        publish_author_email=proj.publish_author_email,
        publish_enabled=proj.publish_enabled,
        publish_auth_configured=bool(proj.publish_auth_token),
        last_publish_at=proj.last_publish_at,
        last_publish_status=proj.last_publish_status,
        last_publish_error=proj.last_publish_error,
        ingest_paused=proj.ingest_paused,
        project_type=proj.project_type,
        case_index_auto_rebuild=proj.case_index_auto_rebuild,
    )


def _build_source_repository_response(repo) -> SourceRepositoryResponse:
    return SourceRepositoryResponse(
        id=repo.id,
        project_id=repo.project_id,
        key=repo.key,
        name=repo.name,
        repo_url=repo.repo_url,
        branch=repo.branch,
        username=repo.username,
        author_name=repo.author_name,
        author_email=repo.author_email,
        sync_enabled=repo.sync_enabled,
        auto_compile=repo.auto_compile,
        sync_time=repo.sync_time,
        last_sync_at=repo.last_sync_at,
        last_sync_status=repo.last_sync_status,
        last_sync_error=repo.last_sync_error,
        auth_configured=bool(repo.auth_token),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )


def _project_for_git_test(project: Project, body: TestGitConnectionRequest | None):
    return SimpleNamespace(
        git_repo_url=body.git_repo_url if body and body.git_repo_url is not None else project.git_repo_url,
        git_branch=body.git_branch if body and body.git_branch is not None else project.git_branch,
        git_username=body.git_username if body and body.git_username is not None else project.git_username,
        git_auth_token=body.git_auth_token if body and body.git_auth_token is not None else project.git_auth_token,
    )


def _project_for_publish_test(project: Project, body: TestPublishConnectionRequest | None):
    return SimpleNamespace(
        publish_repo_url=(
            body.publish_repo_url if body and body.publish_repo_url is not None else project.publish_repo_url
        ),
        publish_branch=(
            body.publish_branch if body and body.publish_branch is not None else project.publish_branch
        ),
        publish_username=(
            body.publish_username if body and body.publish_username is not None else project.publish_username
        ),
        publish_auth_token=(
            body.publish_auth_token if body and body.publish_auth_token is not None else project.publish_auth_token
        ),
    )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: CreateProjectRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proj = await service.create_project(
        db,
        name=body.name,
        slug=body.slug,
        description=body.description,
        user=user,
        as_case_library=body.as_case_library,
        case_library_main_project_id=body.main_project_id if body.as_case_library else None,
    )
    return await _build_project_response(db, proj)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    projects = await service.list_user_projects(db, user)
    return [await _build_project_response(db, p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    return await _build_project_response(db, proj)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)

    kwargs = {}
    if "name" in body.model_fields_set:
        kwargs["name"] = body.name
    if "description" in body.model_fields_set:
        kwargs["description"] = body.description
    if "ticket_project_id" in body.model_fields_set:
        if body.ticket_project_id is not None:
            await service.check_membership(db, body.ticket_project_id, user)
        kwargs["ticket_project_id"] = body.ticket_project_id
    if "feedback_enabled" in body.model_fields_set:
        kwargs["feedback_enabled"] = body.feedback_enabled
    if "case_index_auto_rebuild" in body.model_fields_set:
        kwargs["case_index_auto_rebuild"] = body.case_index_auto_rebuild
    if "ingest_paused" in body.model_fields_set:
        kwargs["ingest_paused"] = body.ingest_paused

    git_fields = [
        "git_repo_url", "git_branch", "git_username", "git_author_name",
        "git_author_email", "git_sync_enabled", "git_sync_auto_compile",
        "git_sync_time",
    ]
    for field in git_fields:
        if field in body.model_fields_set:
            kwargs[field] = getattr(body, field)
    if "git_auth_token" in body.model_fields_set and body.git_auth_token:
        kwargs["git_auth_token"] = body.git_auth_token
    if "clear_git_auth_token" in body.model_fields_set and body.clear_git_auth_token:
        kwargs["git_auth_token"] = ""

    publish_fields = [
        "publish_repo_url", "publish_branch", "publish_username",
        "publish_author_name", "publish_author_email", "publish_enabled",
    ]
    for field in publish_fields:
        if field in body.model_fields_set:
            kwargs[field] = getattr(body, field)
    if "publish_auth_token" in body.model_fields_set and body.publish_auth_token:
        kwargs["publish_auth_token"] = body.publish_auth_token
    if "clear_publish_auth_token" in body.model_fields_set and body.clear_publish_auth_token:
        kwargs["publish_auth_token"] = ""

    proj = await service.update_project(db, proj, **kwargs)

    git_schedule_fields = {"git_sync_enabled", "git_sync_time"}
    if git_schedule_fields & body.model_fields_set:
        from app.projects.git_sync import register_sync_jobs
        import asyncio
        asyncio.create_task(register_sync_jobs())

    return await _build_project_response(db, proj)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    await service.delete_project(db, proj)


@router.post("/{project_id}/members", response_model=MemberResponse, status_code=status.HTTP_201_CREATED)
async def add_member(
    project_id: str,
    body: AddMemberRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    member = await service.add_member(db, project_id, body.user_id, body.role)
    return member


@router.get("/{project_id}/members", response_model=list[MemberResponse])
async def list_members(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    stmt = select(ProjectMember).where(ProjectMember.project_id == project_id)
    return list((await db.execute(stmt)).scalars().all())


@router.get("/{project_id}/source-repositories", response_model=list[SourceRepositoryResponse])
async def list_project_source_repositories(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    repos = await source_repo_service.list_source_repositories(db, proj)
    return [_build_source_repository_response(repo) for repo in repos]


@router.post(
    "/{project_id}/source-repositories",
    response_model=SourceRepositoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project_source_repository(
    project_id: str,
    body: SourceRepositoryCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    repo = await source_repo_service.create_source_repository(
        db,
        proj,
        key=body.key,
        name=body.name,
        repo_url=body.repo_url,
        branch=body.branch,
        username=body.username,
        auth_token=body.auth_token,
        author_name=body.author_name,
        author_email=body.author_email,
        sync_enabled=body.sync_enabled,
        auto_compile=body.auto_compile,
        sync_time=body.sync_time,
    )
    return _build_source_repository_response(repo)


@router.patch("/{project_id}/source-repositories/{repo_id}", response_model=SourceRepositoryResponse)
async def update_project_source_repository(
    project_id: str,
    repo_id: str,
    body: SourceRepositoryUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    repo = await source_repo_service.get_source_repository_or_404(db, proj, repo_id)
    kwargs = body.model_dump(exclude_unset=True, exclude={"clear_auth_token", "auth_token"})
    if "auth_token" in body.model_fields_set and body.auth_token:
        kwargs["auth_token"] = body.auth_token
    if "clear_auth_token" in body.model_fields_set and body.clear_auth_token:
        kwargs["auth_token"] = ""
    repo = await source_repo_service.update_source_repository(db, repo, **kwargs)
    return _build_source_repository_response(repo)


@router.delete("/{project_id}/source-repositories/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_source_repository(
    project_id: str,
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    await source_repo_service.delete_source_repository(db, proj, repo_id)


@router.post("/{project_id}/git/test")
async def test_git_connection(
    project_id: str,
    body: TestGitConnectionRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    from app.projects.git_sync import test_project_git_connection
    return await test_project_git_connection(_project_for_git_test(proj, body))


@router.post("/{project_id}/git/sync")
async def trigger_git_sync(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import asyncio
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    if not proj.git_repo_url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "未配置 Git 仓库")
    from app.projects.git_sync import sync_project_from_git, _get_sync_lock
    lock = _get_sync_lock(project_id)
    if lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "该项目已有同步正在执行")
    asyncio.create_task(sync_project_from_git(project_id, triggered_by=user.id, source="manual"))
    return {"status": "started"}


@router.get("/{project_id}/git/status")
async def get_git_sync_status(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    return {
        "git_sync_enabled": proj.git_sync_enabled,
        "last_git_sync_at": proj.last_git_sync_at.isoformat() if proj.last_git_sync_at else None,
        "last_git_sync_status": proj.last_git_sync_status,
        "last_git_sync_error": proj.last_git_sync_error,
    }


@router.post("/{project_id}/git/publish/test")
async def test_publish_connection(
    project_id: str,
    body: TestPublishConnectionRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)
    from app.projects.git_sync import test_project_publish_connection
    return await test_project_publish_connection(_project_for_publish_test(proj, body))


@router.post("/{project_id}/git/publish")
async def trigger_git_publish(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import asyncio
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    if not proj.publish_repo_url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "未配置发布仓库")
    from app.projects.git_sync import publish_project_to_git, _get_sync_lock
    lock = _get_sync_lock(f"{project_id}:publish")
    if lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "该项目已有发布正在执行")
    asyncio.create_task(publish_project_to_git(project_id, triggered_by=user.id))
    return {"status": "started"}


@router.get("/{project_id}/git/publish/status")
async def get_git_publish_status(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)
    return {
        "publish_enabled": proj.publish_enabled,
        "last_publish_at": proj.last_publish_at.isoformat() if proj.last_publish_at else None,
        "last_publish_status": proj.last_publish_status,
        "last_publish_error": proj.last_publish_error,
    }
