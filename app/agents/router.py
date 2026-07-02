"""Agent CRUD API (JWT-authenticated)."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.agents import conversations
from app.agents.models import Agent, AgentProject
from app.agents import service
from app.agents.orchestrator import _build_system_prompt
from app.auth.deps import get_current_user
from app.auth.models import User
from app.config import get_config
from app.database import get_db
from app.projects.models import Project

router = APIRouter(prefix="/api/agents", tags=["agents"])


class CreateAgentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    system_prompt: str = ""
    system_prompt_override: str = ""
    project_ids: list[str] = []
    is_public: bool = False
    require_api_key: bool = True
    max_tool_calls: int = Field(default=20, ge=1, le=200)
    debug_result_limit: int = Field(default=2000, ge=500, le=50000)
    tool_labels: dict[str, str] = Field(default_factory=dict)


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    system_prompt_override: str | None = None
    project_ids: list[str] | None = None
    is_public: bool | None = None
    require_api_key: bool | None = None
    max_tool_calls: int | None = Field(default=None, ge=1, le=200)
    debug_result_limit: int | None = Field(default=None, ge=500, le=50000)
    tool_labels: dict[str, str] | None = None


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    system_prompt_override: str
    is_public: bool
    require_api_key: bool
    max_tool_calls: int
    debug_result_limit: int
    tool_labels: dict[str, str]
    project_ids: list[str]
    created_by: int
    created_at: datetime

    class Config:
        from_attributes = True


class CreateAgentResponse(BaseModel):
    agent: AgentResponse
    api_key: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str | None = None
    message_count: int


class DefaultSystemPromptResponse(BaseModel):
    system_prompt: str


async def _get_agent_or_404(db: AsyncSession, agent_id: str, user: User) -> Agent:
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if not agent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    if agent.created_by != user.id and user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not the agent owner")
    return agent


def _parse_tool_labels(agent: Agent) -> dict[str, str]:
    try:
        return json.loads(agent.tool_labels or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _agent_response(agent: Agent, project_ids: list[str]) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        system_prompt_override=agent.system_prompt_override or "",
        is_public=agent.is_public,
        require_api_key=agent.require_api_key,
        max_tool_calls=agent.max_tool_calls,
        debug_result_limit=agent.debug_result_limit,
        tool_labels=_parse_tool_labels(agent),
        project_ids=project_ids,
        created_by=agent.created_by,
        created_at=agent.created_at,
    )


async def _agent_project_ids(db: AsyncSession, agent_id: str) -> list[str]:
    stmt = (
        select(AgentProject.project_id)
        .join(Project, Project.id == AgentProject.project_id)
        .where(AgentProject.agent_id == agent_id)
        .where(Project.project_type != "case_library")
    )
    return list((await db.execute(stmt)).scalars().all())


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(
    body: CreateAgentRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent, raw_key = await service.create_agent(
        db,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        project_ids=body.project_ids,
        is_public=body.is_public,
        require_api_key=body.require_api_key,
        system_prompt_override=body.system_prompt_override,
        max_tool_calls=body.max_tool_calls,
        debug_result_limit=body.debug_result_limit,
        tool_labels=body.tool_labels,
        user_id=user.id,
    )
    pids = await _agent_project_ids(db, agent.id)
    return CreateAgentResponse(
        agent=_agent_response(agent, pids),
        api_key=raw_key,
    )


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user.role == "admin":
        stmt = select(Agent).order_by(Agent.created_at.desc())
    else:
        stmt = select(Agent).where(Agent.created_by == user.id).order_by(Agent.created_at.desc())
    agents = list((await db.execute(stmt)).scalars().all())
    result = []
    for a in agents:
        pids = await _agent_project_ids(db, a.id)
        result.append(_agent_response(a, pids))
    return result


@router.get("/default-system-prompt", response_model=DefaultSystemPromptResponse)
async def get_default_system_prompt(has_ticket: bool = True):
    return DefaultSystemPromptResponse(
        system_prompt=_build_system_prompt("", has_ticket=has_ticket)
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent = await _get_agent_or_404(db, agent_id, user)
    pids = await _agent_project_ids(db, agent.id)
    return _agent_response(agent, pids)


@router.get("/{agent_id}/conversations", response_model=list[ConversationResponse])
async def list_agent_conversations(
    agent_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_agent_or_404(db, agent_id, user)
    base_dir = get_config().server.projects_dir
    return conversations.list_conversations(base_dir, agent_id=agent_id, user_id=user.id)


@router.get("/{agent_id}/conversations/{conversation_id}")
async def get_agent_conversation(
    agent_id: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_agent_or_404(db, agent_id, user)
    base_dir = get_config().server.projects_dir
    conv = conversations.get_conversation(
        base_dir,
        agent_id=agent_id,
        user_id=user.id,
        conversation_id=conversation_id,
    )
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


@router.delete("/{agent_id}/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_conversation(
    agent_id: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_agent_or_404(db, agent_id, user)
    base_dir = get_config().server.projects_dir
    conversations.delete_conversation(
        base_dir,
        agent_id=agent_id,
        user_id=user.id,
        conversation_id=conversation_id,
    )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    body: UpdateAgentRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent = await _get_agent_or_404(db, agent_id, user)
    agent = await service.update_agent(db, agent, **body.model_dump(exclude_none=True))
    pids = await _agent_project_ids(db, agent.id)
    return _agent_response(agent, pids)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent = await _get_agent_or_404(db, agent_id, user)
    await db.delete(agent)
    await db.commit()


@router.post("/{agent_id}/regenerate-key")
async def regenerate_key(
    agent_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent = await _get_agent_or_404(db, agent_id, user)
    raw_key = await service.regenerate_key(db, agent)
    return {"api_key": raw_key}


@router.post("/{agent_id}/chat")
async def agent_chat(
    agent_id: str,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent = await _get_agent_or_404(db, agent_id, user)
    projects = await service.get_agent_projects(db, agent.id)
    if not projects:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Agent has no projects")

    conv_id = body.conversation_id
    history = []
    if conv_id:
        stored = conversations.get_conversation(
            get_config().server.projects_dir,
            agent_id=agent.id,
            user_id=user.id,
            conversation_id=conv_id,
        )
        if stored:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in stored["messages"]
                if m.get("role") in ("user", "assistant")
            ]

    async def sse_stream():
        collected_tokens: list[str] = []
        persisted_id: str | None = conv_id
        async for event in service.agent_toolcall_chat(
            db, projects, body.message, history, agent.system_prompt,
            system_prompt_override=agent.system_prompt_override or "",
            max_tool_calls=agent.max_tool_calls,
            debug_result_limit=agent.debug_result_limit,
        ):
            payload = json.loads(event)
            if "token" in payload:
                collected_tokens.append(payload["token"])
            if payload.get("done"):
                conv = conversations.append_turn(
                    get_config().server.projects_dir,
                    agent_id=agent.id,
                    user_id=user.id,
                    conversation_id=persisted_id,
                    user_message=body.message,
                    assistant_answer="".join(collected_tokens),
                    compressed_history=payload.get("compressed_history"),
                )
                persisted_id = conv["id"]
                payload["conversation_id"] = persisted_id
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    from app.feedback.trigger import wrap_agent_sse
    wrapped = wrap_agent_sse(
        sse_stream(),
        project_id=projects[0].id if projects else "",
        agent_id=agent.id,
        conversation_id=body.conversation_id,
        user_message=body.message,
    )
    return StreamingResponse(wrapped, media_type="text/event-stream")
