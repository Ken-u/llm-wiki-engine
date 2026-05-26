"""Public Agent API — accessible with optional API key, no JWT required."""

from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.agents.models import Agent
from app.agents import service
from app.database import get_db

router = APIRouter(prefix="/api/public/agents", tags=["public-agents"])


class PublicChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


@router.post("/{agent_id}/chat")
async def public_agent_chat(
    agent_id: str,
    body: PublicChatRequest,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if not agent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    if not agent.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Agent is not public")

    # Auth check
    if agent.require_api_key:
        if not authorization:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key required")
        raw_key = authorization.replace("Bearer ", "").strip()
        if not await service.verify_api_key(db, agent, raw_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

    projects = await service.get_agent_projects(db, agent.id)
    if not projects:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Agent has no knowledge bases")

    async def sse_stream():
        async for token in service.cross_project_rag(
            projects, body.message, [], agent.system_prompt,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
