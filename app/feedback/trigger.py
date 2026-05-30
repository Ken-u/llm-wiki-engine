"""Thin SSE wrapper that intercepts agent conversation events.

This is the ONLY integration point with existing agent modules. It
wraps the SSE generator at the router level without modifying any
agent internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def wrap_agent_sse(
    sse_generator: AsyncGenerator[str, None],
    *,
    project_id: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Transparently pass-through all SSE events while collecting data for feedback.

    After the stream completes, asynchronously fires feedback evaluation
    if tool traces indicate potential wiki deficiencies.
    """
    collected_tokens: list[str] = []
    collected_traces: list[dict] = []

    async for event in sse_generator:
        yield event

        try:
            raw = event.strip()
            if not raw.startswith("data: "):
                continue
            payload = json.loads(raw[6:])

            if "token" in payload:
                collected_tokens.append(payload["token"])
            elif "done" in payload and payload["done"]:
                collected_traces = payload.get("tool_traces", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    if not collected_traces:
        return

    full_answer = "".join(collected_tokens)
    conv_id = conversation_id or str(uuid.uuid4())

    from app.feedback.queue import maybe_trigger_feedback
    asyncio.create_task(
        maybe_trigger_feedback(
            project_id=project_id,
            conversation_id=conv_id,
            agent_id=agent_id,
            user_message=user_message,
            assistant_answer=full_answer,
            tool_traces=collected_traces,
        )
    )
