"""Feedback async processing queue.

Orchestrates the feedback pipeline: extract reads from tool traces,
trigger evaluation, and (if needed) run repair compilation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from app.config import get_config
from app.database import async_session
from app.feedback import service
from app.feedback.evaluator import EvaluatorInput, run_evaluator
from app.feedback.compiler import CompilerInput, run_compiler

logger = logging.getLogger(__name__)

WIKI_TOOL_NAMES = {"search_wiki", "read_wiki_page", "get_wiki_index"}
RAW_TOOL_NAMES = {"read_raw", "grep_raw"}


def _extract_reads(tool_traces: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split tool traces into wiki reads and raw reads."""
    wiki_reads: list[dict] = []
    raw_reads: list[dict] = []
    for trace in tool_traces:
        name = trace.get("name", "")
        if name in WIKI_TOOL_NAMES:
            wiki_reads.append(trace)
        elif name in RAW_TOOL_NAMES:
            raw_reads.append(trace)
    return wiki_reads, raw_reads


def _has_raw_usage(tool_traces: list[dict]) -> bool:
    """Return True if any tool trace uses a raw read tool."""
    return any(t.get("name") in RAW_TOOL_NAMES for t in tool_traces)


async def maybe_trigger_feedback(
    *,
    project_id: str,
    conversation_id: str,
    agent_id: str | None,
    user_message: str,
    assistant_answer: str,
    tool_traces: list[dict],
) -> None:
    """Decide whether to create a feedback task and start the pipeline.

    Called asynchronously after an agent conversation completes.
    """
    cfg = get_config().feedback
    if not cfg.enabled:
        return

    if not tool_traces:
        return

    if not _has_raw_usage(tool_traces):
        return

    if not await _is_project_feedback_enabled(project_id):
        return

    wiki_reads, raw_reads = _extract_reads(tool_traces)

    async with async_session() as db:
        task = await service.create_feedback_task(
            db,
            project_id=project_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_message=user_message,
            assistant_answer=assistant_answer,
            tool_traces=tool_traces,
            wiki_reads=wiki_reads,
            raw_reads=raw_reads,
        )

    logger.info("Created feedback task %s for conversation %s", task.id, conversation_id)
    await _run_evaluation(task.id)


async def _run_evaluation(task_id: str) -> None:
    """Run evaluator on a feedback task."""
    cfg = get_config().feedback

    async with async_session() as db:
        task = await service.get_task(db, task_id)
        if not task:
            return

        inp = EvaluatorInput(
            user_message=task.user_message,
            assistant_answer=task.assistant_answer,
            tool_traces=json.loads(task.tool_traces_json),
            wiki_reads=json.loads(task.wiki_reads_json),
            raw_reads=json.loads(task.raw_reads_json),
        )

        try:
            result = await run_evaluator(inp, cfg.evaluator)
        except Exception:
            logger.exception("Evaluator failed for task %s", task_id)
            await service.mark_compile_failed(db, task, "evaluator exception")
            return

        await service.set_evaluator_result(
            db, task,
            needs_repair=result.needs_repair,
            evaluator_output=result.raw,
        )

        if not result.needs_repair:
            logger.info("Task %s: no repair needed (reason: %s)", task_id, result.reason)
            return

    logger.info("Task %s: repair needed, starting compilation", task_id)
    await _run_repair(task_id)


async def _run_repair(task_id: str) -> None:
    """Run compiler on a feedback task that needs repair."""
    cfg = get_config().feedback

    async with async_session() as db:
        task = await service.get_task(db, task_id)
        if not task:
            return

        if task.status not in ("pending_recompile",):
            service.transition_status(task, "pending_recompile")
            await db.commit()
            await db.refresh(task)

        evaluator_result = json.loads(task.evaluator_result_json or "{}")

        existing_content: str | None = None
        if task.target_page_path:
            existing_content = await _fetch_wiki_page(task.project_id, task.target_page_path)

        inp = CompilerInput(
            user_message=task.user_message,
            assistant_answer=task.assistant_answer,
            evaluator_result=evaluator_result,
            target_page_path=task.target_page_path or "",
            existing_page_content=existing_content,
            review_guidance=task.review_guidance,
            wiki_reads=json.loads(task.wiki_reads_json),
            raw_reads=json.loads(task.raw_reads_json),
        )

        try:
            result = await run_compiler(inp, cfg.compiler)
        except Exception:
            logger.exception("Compiler failed for task %s", task_id)
            await service.mark_compile_failed(db, task, "compiler exception")
            return

        await service.set_repair_candidate(db, task, asdict(result))
        logger.info("Task %s: repair candidate ready for review", task_id)


async def trigger_recompile(task_id: str) -> None:
    """Re-run the compiler after a revision request."""
    await _run_repair(task_id)


async def _fetch_wiki_page(project_id: str, page_path: str) -> str | None:
    """Try to read an existing wiki page's content. Returns None if not found."""
    try:
        from app.projects.models import Project
        from sqlalchemy import select

        async with async_session() as db:
            proj = (await db.execute(
                select(Project).where(Project.id == project_id)
            )).scalar_one_or_none()
            if not proj:
                return None

        import os
        full_path = os.path.join(proj.disk_path, "wiki", page_path)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        logger.debug("Could not fetch wiki page %s for project %s", page_path, project_id)
    return None


async def _is_project_feedback_enabled(project_id: str) -> bool:
    """Check the per-project feedback_enabled flag."""
    try:
        from app.projects.models import Project
        from sqlalchemy import select

        async with async_session() as db:
            val = (await db.execute(
                select(Project.feedback_enabled).where(Project.id == project_id)
            )).scalar_one_or_none()
            return val is not False
    except Exception:
        return True
