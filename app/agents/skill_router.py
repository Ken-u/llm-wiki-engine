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
from app.agents.skill_refs import sign_source_ref, verify_source_ref
from app.llm.model_select import fast_virtual_model_id, parse_virtual_model
from app.database import get_db
from app.documents.service import read_document_content
from pathlib import Path

router = APIRouter(prefix="/api/public/skills", tags=["public-skills"])


class SkillChatRequest(BaseModel):
    message: str = Field(min_length=1)
    model: str | None = None
    use_fast_model: bool = False


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


def build_skill_markdown(
    agent: Agent,
    base_url: str,
    raw_token: str,
    *,
    knowledge_model_name: str = "",
) -> str:
    skill_name = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in agent.name.strip()
    ).strip("-") or "llm-wiki-knowledge"
    chat_url = f"{base_url}/api/public/skills/chat"
    source_url = f"{base_url}/api/public/skills/documents/content?ref=<source_ref>"
    fast_model_lines = [
        "",
        "Fast model (optional, when configured by the knowledge base admin):",
        '- Default request uses the standard model: `{"message": "..."}`',
        '- Use fast model: `{"message": "...", "use_fast_model": true}`',
    ]
    if knowledge_model_name:
        fast_model_lines.append(
            f'- Or set model to `{fast_virtual_model_id(knowledge_model_name)}`'
        )
    fast_model_section = "\n".join(fast_model_lines)
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
{fast_model_section}

Only read source documents through `source_ref` values returned by chat. Do not guess filenames or paths. Do not list raw documents.
If the knowledge base does not contain an answer, say so clearly.
"""


async def collect_skill_answer(
    db: AsyncSession,
    agent: Agent,
    message: str,
    *,
    use_fast_model: bool = False,
) -> tuple[str, list[SkillSource]]:
    projects = await service.get_agent_projects(db, agent.id)
    chunks: list[str] = []
    sources_by_name: dict[str, SkillSource] = {}

    def add_source(doc_name: str) -> None:
        if not doc_name or doc_name in sources_by_name:
            return
        for project in projects:
            candidate = Path(project.disk_path) / "raw" / "sources" / doc_name
            if candidate.is_file():
                sources_by_name[doc_name] = SkillSource(
                    name=doc_name,
                    source_ref=sign_source_ref(
                        agent_id=agent.id,
                        project_id=project.id,
                        doc_name=doc_name,
                        display_name=doc_name,
                    ),
                )
                return

    async for event in service.agent_toolcall_chat(
        db,
        projects,
        message,
        [],
        agent.system_prompt,
        system_prompt_override=agent.system_prompt_override or "",
        max_tool_calls=agent.max_tool_calls,
        debug_result_limit=agent.debug_result_limit,
        use_fast_model=use_fast_model,
    ):
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        token = payload.get("token")
        if token:
            chunks.append(token)
        tool_result = payload.get("tool_result")
        if isinstance(tool_result, dict):
            result = tool_result.get("result")
            if tool_result.get("name") == "read_raw" and isinstance(result, dict):
                add_source(str(result.get("path") or ""))
            if tool_result.get("name") == "grep_raw" and isinstance(result, dict):
                for match in result.get("matches") or []:
                    if isinstance(match, dict):
                        add_source(str(match.get("file") or ""))
    return "".join(chunks), list(sources_by_name.values())


@router.post("/chat", response_model=SkillChatResponse)
async def skill_chat(
    body: SkillChatRequest,
    access: SkillAccess = Depends(require_skill_access),
    db: AsyncSession = Depends(get_db),
):
    _, fast_from_model = parse_virtual_model(body.model or "")
    use_fast_model = body.use_fast_model or fast_from_model
    answer, sources = await collect_skill_answer(
        db, access.agent, body.message, use_fast_model=use_fast_model,
    )
    return SkillChatResponse(answer=answer, sources=sources)


@router.get("/documents/content")
async def read_skill_document_content(
    ref: str,
    access: SkillAccess = Depends(require_skill_access),
    db: AsyncSession = Depends(get_db),
):
    payload = verify_source_ref(ref)
    if payload is None or payload.agent_id != access.agent.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found")

    projects = await service.get_agent_projects(db, access.agent.id)
    project = next((p for p in projects if p.id == payload.project_id), None)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found")

    content = read_document_content(project, payload.doc_name)
    if content is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found")
    return PlainTextResponse(content)


@router.get("/{install_token}")
async def download_skill(
    install_token: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    agent = await _agent_for_skill_token(db, install_token)
    projects = await service.get_agent_projects(db, agent.id)
    knowledge_model_name = next(
        (p.knowledge_api_model_name for p in projects if p.knowledge_api_model_name),
        "",
    )
    markdown = build_skill_markdown(
        agent,
        str(request.base_url).rstrip("/"),
        install_token,
        knowledge_model_name=knowledge_model_name,
    )
    return PlainTextResponse(markdown, media_type="text/markdown")
