"""Feedback task business logic: create, transition, dedup, guidance handling."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_config
from app.feedback.models import FeedbackTask, VALID_STATUSES

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"approved", "rejected", "applied"}


async def create_feedback_task(
    db: AsyncSession,
    *,
    project_id: str,
    conversation_id: str,
    agent_id: str | None,
    user_message: str,
    assistant_answer: str,
    tool_traces: list[dict],
    wiki_reads: list[dict],
    raw_reads: list[dict],
) -> FeedbackTask:
    """Create a new feedback task in pending_evaluation status."""
    task = FeedbackTask(
        id=str(uuid.uuid4()),
        project_id=project_id,
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_message=user_message,
        assistant_answer=assistant_answer,
        tool_traces_json=json.dumps(tool_traces, ensure_ascii=False),
        wiki_reads_json=json.dumps(wiki_reads, ensure_ascii=False),
        raw_reads_json=json.dumps(raw_reads, ensure_ascii=False),
        status="pending_evaluation",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task


async def get_task(db: AsyncSession, task_id: str) -> FeedbackTask | None:
    return (await db.execute(
        select(FeedbackTask).where(FeedbackTask.id == task_id)
    )).scalar_one_or_none()


async def list_tasks(
    db: AsyncSession,
    project_id: str,
    *,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[FeedbackTask]:
    stmt = (
        select(FeedbackTask)
        .where(FeedbackTask.project_id == project_id)
        .order_by(FeedbackTask.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status_filter and status_filter in VALID_STATUSES:
        stmt = stmt.where(FeedbackTask.status == status_filter)
    return list((await db.execute(stmt)).scalars().all())


async def count_tasks(db: AsyncSession, project_id: str) -> dict[str, int]:
    tasks = (await db.execute(
        select(FeedbackTask.status)
        .where(FeedbackTask.project_id == project_id)
    )).scalars().all()
    counts: dict[str, int] = {}
    for s in tasks:
        counts[s] = counts.get(s, 0) + 1
    return counts


def transition_status(task: FeedbackTask, new_status: str) -> None:
    """Validate and apply a status transition. Raises ValueError on illegal transition."""
    allowed = _TRANSITIONS.get(task.status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Cannot transition from '{task.status}' to '{new_status}'. "
            f"Allowed: {allowed}"
        )
    task.status = new_status


_TRANSITIONS: dict[str, set[str]] = {
    "pending_evaluation": {"evaluation_done", "rejected", "compile_failed"},
    "evaluation_done": {"pending_evaluation", "pending_review", "pending_recompile"},
    "pending_review": {"pending_evaluation", "approved", "rejected", "pending_recompile"},
    "pending_recompile": {"pending_evaluation", "pending_review", "compile_failed"},
    "approved": {"applied"},
    "rejected": {"pending_evaluation", "pending_recompile"},
    "compile_failed": {"pending_evaluation", "pending_recompile", "rejected"},
}


async def set_evaluator_result(
    db: AsyncSession,
    task: FeedbackTask,
    *,
    needs_repair: bool,
    evaluator_output: dict,
) -> None:
    task.evaluator_result_json = json.dumps(evaluator_output, ensure_ascii=False)
    task.evaluator_confidence = evaluator_output.get("confidence", "low")

    if not needs_repair:
        transition_status(task, "rejected")
        task.reject_reason = "evaluator: no repair needed"
    else:
        transition_status(task, "evaluation_done")

    await db.commit()


async def set_repair_candidate(
    db: AsyncSession,
    task: FeedbackTask,
    candidate: dict,
) -> None:
    task.repair_candidate_json = json.dumps(candidate, ensure_ascii=False)
    transition_status(task, "pending_review")
    await db.commit()


async def approve_task(db: AsyncSession, task: FeedbackTask) -> None:
    transition_status(task, "approved")
    await db.commit()


async def reject_task(db: AsyncSession, task: FeedbackTask, reason: str) -> None:
    task.reject_reason = reason
    transition_status(task, "rejected")
    await db.commit()


async def request_revision(
    db: AsyncSession,
    task: FeedbackTask,
    guidance: str,
) -> None:
    cfg = get_config().feedback
    if task.revision_count >= cfg.max_revisions:
        raise ValueError(
            f"Max revisions ({cfg.max_revisions}) reached for task {task.id}"
        )
    task.review_guidance = guidance
    task.revision_count += 1
    transition_status(task, "pending_recompile")
    await db.commit()


async def mark_applied(db: AsyncSession, task: FeedbackTask) -> None:
    transition_status(task, "applied")
    await db.commit()


async def mark_compile_failed(
    db: AsyncSession, task: FeedbackTask, error: str
) -> None:
    task.error = error
    transition_status(task, "compile_failed")
    await db.commit()


async def delete_task(db: AsyncSession, task: FeedbackTask) -> None:
    await db.delete(task)
    await db.commit()


async def find_duplicate(
    db: AsyncSession,
    project_id: str,
    target_page_path: str,
) -> FeedbackTask | None:
    cfg = get_config().feedback
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=cfg.dedup_window_minutes)

    stmt = (
        select(FeedbackTask)
        .where(and_(
            FeedbackTask.project_id == project_id,
            FeedbackTask.target_page_path == target_page_path,
            FeedbackTask.status.notin_(list(TERMINAL_STATUSES)),
            FeedbackTask.created_at >= cutoff,
        ))
        .order_by(FeedbackTask.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()
