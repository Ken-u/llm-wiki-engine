"""Document upload/list/read endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.documents import service
from app.projects.service import check_membership, get_project_or_404

router = APIRouter(prefix="/api/projects/{project_id}/documents", tags=["documents"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large (max 50MB)")

    dest = await service.save_uploaded_file(project, file.filename, content)
    return {"filename": file.filename, "path": str(dest), "size": len(content)}


@router.get("")
async def list_documents(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    return service.list_documents(project)


@router.get("/content/{doc_path:path}")
async def read_document_content(
    project_id: str,
    doc_path: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read a document's text content for preview."""
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    content = service.read_document_content(project, doc_path)
    if content is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return PlainTextResponse(content)
