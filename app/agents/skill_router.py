"""Public Agent Skill endpoints.

These routes resolve the target Agent from a Skill Token instead of exposing
agent IDs to installed external Skills.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import service
from app.agents.models import Agent
from app.database import get_db

router = APIRouter(prefix="/api/public/skills", tags=["public-skills"])


class SkillChatRequest(BaseModel):
    message: str = Field(min_length=1)


class SkillSource(BaseModel):
    name: str
    source_ref: str


class SkillChatResponse(BaseModel):
    answer: str
    sources: list[SkillSource] = Field(default_factory=list)


@dataclass
class SkillAccess:
    agent: Agent


def _hash_token(raw_token: str) -> str:
    return service._hash_key(raw_token)  # Reuse existing Agent key hashing.


async def _agent_for_skill_token(db: AsyncSession, raw_token: str) -> Agent:
    agent = (
        await db.execute(
            select(Agent).where(Agent.skill_token_hash == _hash_token(raw_token))
        )
    ).scalar_one_or_none()
    if agent is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Skill Token")
    if not agent.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Agent is not public")
    projects = await service.get_agent_projects(db, agent.id)
    if not projects:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Agent has no knowledge bases")
    return agent


async def require_skill_access(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> SkillAccess:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Skill Token required")
    raw_token = authorization.replace("Bearer ", "").strip()
    return SkillAccess(agent=await _agent_for_skill_token(db, raw_token))


def build_skill_markdown(agent: Agent, base_url: str, raw_token: str) -> str:
    skill_name = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in agent.name.strip()
    ).strip("-") or "llm-wiki-knowledge"
    chat_url = f"{base_url}/api/public/skills/chat"
    source_url = f"{base_url}/api/public/skills/documents/content?ref=<source_ref>"
    return f"""---
name: {skill_name}-knowledge
description: Use when the user asks to query or verify information against the {agent.name} knowledge base.
---

# {agent.name} Knowledge

When the user asks to use this knowledge base, call the remote Skill chat endpoint before answering.

Do not answer from memory for in-scope questions.

Use:

- POST {chat_url}
- GET {source_url}

Authorization: Bearer {raw_token}

Only read source documents through `source_ref` values returned by chat. Do not guess filenames or paths. Do not list raw documents.
If the knowledge base does not contain an answer, say so clearly.
"""


async def collect_skill_answer(
    db: AsyncSession,
    agent: Agent,
    message: str,
) -> tuple[str, list[SkillSource]]:
    projects = await service.get_agent_projects(db, agent.id)
    chunks: list[str] = []
    async for event in service.agent_toolcall_chat(
        db,
        projects,
        message,
        [],
        agent.system_prompt,
        system_prompt_override=agent.system_prompt_override or "",
        max_tool_calls=agent.max_tool_calls,
        debug_result_limit=agent.debug_result_limit,
    ):
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        token = payload.get("token")
        if token:
            chunks.append(token)
    return "".join(chunks), []


@router.post("/chat", response_model=SkillChatResponse)
async def skill_chat(
    body: SkillChatRequest,
    access: SkillAccess = Depends(require_skill_access),
    db: AsyncSession = Depends(get_db),
):
    answer, sources = await collect_skill_answer(db, access.agent, body.message)
    return SkillChatResponse(answer=answer, sources=sources)


@router.get("/{install_token}")
async def download_skill(
    install_token: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    agent = await _agent_for_skill_token(db, install_token)
    markdown = build_skill_markdown(agent, str(request.base_url).rstrip("/"), install_token)
    return PlainTextResponse(markdown, media_type="text/markdown")
