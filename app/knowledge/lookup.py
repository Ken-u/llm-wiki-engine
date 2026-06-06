"""Knowledge lookup orchestration — routes between fast and slow paths."""

from __future__ import annotations

import json
import logging

from app.agents.orchestrator import run_agent_turn
from app.agents.tools import ToolContext
from app.knowledge.fast_lookup import (
    FastLookupResult,
    extract_term_from_definition_query,
    fast_lookup,
    is_definition_query,
)
from app.projects.models import Project

logger = logging.getLogger(__name__)


async def run_agent_turn_collect(
    message: str,
    history: list[dict],
    system_prompt: str,
    ctx: ToolContext,
    *,
    max_tool_calls: int = 10,
) -> str:
    """Run an agent turn and collect all token events into a single string."""
    parts: list[str] = []
    async for event_str in run_agent_turn(
        message, history, system_prompt, ctx,
        max_tool_calls=max_tool_calls,
    ):
        try:
            event = json.loads(event_str)
        except json.JSONDecodeError:
            continue
        if "token" in event:
            parts.append(event["token"])
    return "".join(parts)


async def knowledge_lookup(
    project: Project,
    message: str,
    system_prompt: str,
    max_tool_calls: int = 10,
) -> str:
    """Main entry point for knowledge lookup.

    Routes between fast path (definition) and slow path (Agent tool loop).
    Slow path never includes ticket project tools.
    """
    # Fast path: only for definition-style queries
    if is_definition_query(message):
        term = extract_term_from_definition_query(message)
        result = fast_lookup(project.disk_path, term)
        if result:
            logger.info("knowledge fast-path hit: term=%r matched_by=%s", term, result.matched_by)
            return _format_fast_result(result)

    # Slow path: full Agent tool-calling loop, no ticket project
    logger.info("knowledge slow-path: message=%r", message[:80])
    ctx = ToolContext(
        main_projects=[project],
        ticket_project=None,
    )
    text = await run_agent_turn_collect(
        message, [], system_prompt, ctx,
        max_tool_calls=max_tool_calls,
    )
    return text or "知识库中未找到相关信息。"


def _format_fast_result(result: FastLookupResult) -> str:
    """Format a fast lookup result for the completion response."""
    parts = []
    if result.title:
        parts.append(f"# {result.title}")
        parts.append("")
    parts.append(result.content)
    parts.append("")
    parts.append(f"---\n来源：[[{result.path}]]")
    return "\n".join(parts)
