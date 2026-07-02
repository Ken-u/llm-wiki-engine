"""Search API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.index_tasks import (
    IndexRebuildTaskResponse,
    create_index_task,
    get_active_index_task,
    get_index_task,
    has_active_index_task,
    set_index_task,
)
from app.projects.service import check_membership, get_project_or_404
from app.search.bm25 import search_bm25
from app.search.fusion import FusedResult, rrf_fusion

router = APIRouter(prefix="/api/projects/{project_id}/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern=r"^(hybrid|keyword|vector)$")


class SearchResultItem(BaseModel):
    path: str
    page_id: str
    title: str
    score: float
    snippet: str
    sources: list[str]


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    mode: str
    keyword_hits: int
    vector_hits: int


class ReindexResponse(BaseModel):
    pages: int
    chunks: int


async def _run_vector_reindex_task(task_id: str, disk_path: str) -> None:
    try:
        set_index_task(task_id, status="running", progress=10, stage="重建知识库向量索引")
        from app.embedding.service import rebuild_project_embeddings

        result = await rebuild_project_embeddings(disk_path)
        set_index_task(
            task_id,
            status="succeeded",
            progress=100,
            stage="知识库向量索引已重建",
            result=result,
        )
    except Exception as exc:
        set_index_task(
            task_id,
            status="failed",
            progress=100,
            stage="知识库向量索引重建失败",
            error=str(exc),
        )


@router.post("", response_model=SearchResponse)
async def search(
    project_id: str,
    body: SearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    kw_results = []
    vec_results = []

    if body.mode in ("hybrid", "keyword"):
        kw_results = search_bm25(project.disk_path, body.query, top_k=body.top_k * 2)

    if body.mode in ("hybrid", "vector"):
        from app.search.vector import search_vector

        vec_results = await search_vector(project.disk_path, body.query, top_k=body.top_k * 2)

    if body.mode == "keyword":
        fused = [
            FusedResult(
                path=r.path, page_id=r.page_id, title=r.title,
                score=r.score, snippet=r.snippet, sources=["keyword"],
            )
            for r in kw_results[:body.top_k]
        ]
    elif body.mode == "vector":
        fused = [
            FusedResult(
                path=r.path, page_id=r.page_id, title=r.page_id,
                score=r.score, snippet=r.chunk_text[:200], sources=["vector"],
            )
            for r in vec_results[:body.top_k]
        ]
    else:
        fused = rrf_fusion(kw_results, vec_results, body.query)[:body.top_k]

    return SearchResponse(
        results=[SearchResultItem(**f.__dict__) for f in fused],
        mode=body.mode,
        keyword_hits=len(kw_results),
        vector_hits=len(vec_results),
    )


@router.post("/reindex", response_model=ReindexResponse)
async def rebuild_vector_index(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    from app.embedding.service import rebuild_project_embeddings

    return await rebuild_project_embeddings(project.disk_path)


@router.post("/reindex/tasks", response_model=IndexRebuildTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_vector_reindex_task(
    project_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    if has_active_index_task(target="knowledge", project_id=project_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Rebuild already in progress")
    task = create_index_task("knowledge", project_id=project_id)
    background_tasks.add_task(_run_vector_reindex_task, task.task_id, project.disk_path)
    return task


@router.get("/reindex/tasks/current", response_model=IndexRebuildTaskResponse | None)
async def get_current_vector_reindex_task(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    return get_active_index_task(target="knowledge", project_id=project_id)


@router.get("/reindex/tasks/{task_id}", response_model=IndexRebuildTaskResponse)
async def get_vector_reindex_status(
    project_id: str,
    task_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    task = get_index_task(task_id)
    if task is None or task.project_id != project_id or task.target != "knowledge":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rebuild task not found")
    return task
