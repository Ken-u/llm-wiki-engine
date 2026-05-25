"""Project management endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.projects import service
from app.projects.models import ProjectMember

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    slug: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""


class ProjectResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str
    created_by: int
    created_at: datetime

    class Config:
        from_attributes = True


class AddMemberRequest(BaseModel):
    user_id: int
    role: str = Field(default="editor", pattern=r"^(owner|editor|viewer)$")


class MemberResponse(BaseModel):
    user_id: int
    role: str

    class Config:
        from_attributes = True


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: CreateProjectRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proj = await service.create_project(db, name=body.name, slug=body.slug, description=body.description, user=user)
    return proj


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_user_projects(db, user)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    return await service.get_project_or_404(db, project_id)


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
