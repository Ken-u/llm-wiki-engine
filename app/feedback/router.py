"""Feedback REST API — list, detail, review actions."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.feedback import service
from app.feedback.queue import trigger_recompile, trigger_reevaluate

router = APIRouter(prefix="/api/projects/{project_id}/feedback", tags=["feedback"])


class FeedbackTaskResponse(BaseModel):
    id: str
    conversation_id: str
    agent_id: str | None
    target_page_path: str | None
    status: str
    evaluator_confidence: str | None
    revision_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FeedbackTaskDetailResponse(FeedbackTaskResponse):
    user_message: str
    assistant_answer: str
    evaluator_result: dict | None = None
    repair_candidate: dict | None = None
    existing_page_content: str | None = None
    review_guidance: str | None = None
    reject_reason: str | None = None
    error: str | None = None


class FeedbackCountsResponse(BaseModel):
    counts: dict[str, int]
    total: int


class ReviewRequest(BaseModel):
    action: str = Field(description="approve | reject | revise")
    reason: str = ""
    guidance: str = ""


@router.get("", response_model=list[FeedbackTaskResponse])
async def list_feedback_tasks(
    project_id: str,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tasks = await service.list_tasks(
        db, project_id, status_filter=status_filter, limit=limit, offset=offset,
    )
    return [FeedbackTaskResponse.model_validate(t) for t in tasks]


@router.get("/counts", response_model=FeedbackCountsResponse)
async def feedback_counts(
    project_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    counts = await service.count_tasks(db, project_id)
    return FeedbackCountsResponse(counts=counts, total=sum(counts.values()))


@router.get("/{task_id}", response_model=FeedbackTaskDetailResponse)
async def get_feedback_task(
    project_id: str,
    task_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")

    existing_content = await _read_existing_page(db, task)

    return FeedbackTaskDetailResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        agent_id=task.agent_id,
        target_page_path=task.target_page_path,
        status=task.status,
        evaluator_confidence=task.evaluator_confidence,
        revision_count=task.revision_count,
        created_at=task.created_at,
        updated_at=task.updated_at,
        user_message=task.user_message,
        assistant_answer=task.assistant_answer,
        evaluator_result=_safe_json(task.evaluator_result_json),
        repair_candidate=_safe_json(task.repair_candidate_json),
        existing_page_content=existing_content,
        review_guidance=task.review_guidance,
        reject_reason=task.reject_reason,
        error=task.error,
    )


@router.post("/{task_id}/review")
async def review_feedback_task(
    project_id: str,
    task_id: str,
    body: ReviewRequest,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")

    if body.action == "approve":
        await service.approve_task(db, task)
        return {"status": "approved"}
    elif body.action == "reject":
        await service.reject_task(db, task, body.reason)
        return {"status": "rejected"}
    elif body.action == "revise":
        try:
            await service.request_revision(db, task, body.guidance)
        except ValueError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
        asyncio.create_task(trigger_recompile(task.id))
        return {"status": "pending_recompile"}
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown action: {body.action}")


@router.delete("/{task_id}")
async def delete_feedback_task(
    project_id: str,
    task_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")
    await service.delete_task(db, task)
    return {"status": "deleted"}


@router.post("/{task_id}/reevaluate")
async def reevaluate_feedback_task(
    project_id: str,
    task_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-run the evaluator from scratch."""
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")

    reevaluable = {
        "evaluation_done", "pending_review", "pending_recompile",
        "compile_failed", "rejected",
    }
    if task.status not in reevaluable:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cannot reevaluate from status '{task.status}'. Allowed: {reevaluable}",
        )

    service.transition_status(task, "pending_evaluation")
    task.evaluator_result_json = None
    task.evaluator_confidence = None
    task.repair_candidate_json = None
    task.reject_reason = None
    task.review_guidance = None
    task.error = None
    await db.commit()

    asyncio.create_task(trigger_reevaluate(task.id))
    return {"status": "pending_evaluation"}


@router.post("/{task_id}/recompile")
async def force_recompile(
    project_id: str,
    task_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Force trigger recompilation regardless of current status."""
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")

    recompilable = {"evaluation_done", "pending_review", "compile_failed", "rejected"}
    if task.status not in recompilable:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Cannot recompile from status '{task.status}'. Allowed: {recompilable}",
        )

    if task.status != "pending_recompile":
        service.transition_status(task, "pending_recompile")
        await db.commit()

    asyncio.create_task(trigger_recompile(task.id))
    return {"status": "pending_recompile"}


@router.post("/{task_id}/apply")
async def apply_feedback_task(
    project_id: str,
    task_id: str,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    task = await service.get_task(db, task_id)
    if not task or task.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback task not found")

    if task.status != "approved":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Task must be approved before applying")

    candidate = _safe_json(task.repair_candidate_json)
    if not candidate:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No repair candidate")

    changes = candidate.get("changes")
    if not changes and candidate.get("proposed_content"):
        changes = [{
            "path": task.target_page_path,
            "action": "modify",
            "new_content": candidate["proposed_content"],
        }]
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file changes in repair candidate")

    try:
        await _apply_wiki_changes(db, task.project_id, changes, task.id)
        await service.mark_applied(db, task)
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to apply: {e}")

    return {"status": "applied", "files_changed": len(changes)}


async def _read_existing_page(db: AsyncSession, task) -> str | None:
    """Read the current wiki page content for diff display."""
    if not task.target_page_path:
        return None
    try:
        import os
        from sqlalchemy import select as _sel
        from app.projects.models import Project

        proj = (await db.execute(
            _sel(Project).where(Project.id == task.project_id)
        )).scalar_one_or_none()
        if not proj:
            return None
        full_path = os.path.join(proj.disk_path, "wiki", task.target_page_path)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return None


def _safe_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def _apply_wiki_changes(
    db: AsyncSession,
    project_id: str,
    changes: list[dict],
    task_id: str,
) -> None:
    """Apply multi-file changes to wiki with git safety."""
    import os
    import subprocess
    from sqlalchemy import select
    from app.projects.models import Project

    proj = (await db.execute(
        select(Project).where(Project.id == project_id)
    )).scalar_one_or_none()
    if not proj:
        raise ValueError("Project not found")

    disk_path = proj.disk_path
    _ensure_git(disk_path)
    _git_commit_snapshot(disk_path, f"pre-feedback-apply: task {task_id[:8]}")

    wiki_dir = os.path.join(disk_path, "wiki")
    try:
        for ch in changes:
            path = ch.get("path")
            if not path:
                continue
            full_path = os.path.join(wiki_dir, os.path.normpath(path))
            if os.path.normpath(path).startswith(".."):
                raise ValueError(f"Path traversal not allowed: {path}")

            action = ch.get("action", "modify")
            content = ch.get("new_content", "")

            if action in ("modify", "create"):
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

        _git_commit_snapshot(disk_path, f"feedback-apply: task {task_id[:8]}")
    except Exception:
        _git_rollback(disk_path)
        raise


def _ensure_git(path: str) -> None:
    """Initialize git repo if not already present."""
    import subprocess
    git_dir = os.path.join(path, ".git")
    if not os.path.isdir(git_dir):
        subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "feedback@llm-wiki"],
            cwd=path, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "LLM-Wiki Feedback"],
            cwd=path, capture_output=True,
        )


def _git_commit_snapshot(path: str, message: str) -> None:
    """Stage all and commit. Silently succeeds if nothing to commit."""
    import subprocess
    subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty-message"],
        cwd=path, capture_output=True,
    )


def _git_rollback(path: str) -> None:
    """Reset to the last commit (undo uncommitted changes)."""
    import subprocess
    subprocess.run(["git", "checkout", "--", "."], cwd=path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=path, capture_output=True)
