"""Project knowledge API endpoints — configure and manage the OpenAI-compatible knowledge service."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import Agent, AgentProject
from app.auth.deps import generate_api_token, get_current_user
from app.auth.models import User, UserApiToken
from app.database import get_db
from app.projects import service
from app.projects.models import Project

router = APIRouter(prefix="/api/projects/{project_id}/knowledge-api", tags=["knowledge-api"])

DEFAULT_KNOWLEDGE_SYSTEM_PROMPT = (
    "你是一个术语/概念查询助手。你的任务是从知识库中查找用户询问的专业术语或概念的定义与解释。\n\n"
    "策略：\n"
    "- 使用 search_wiki 搜索相关知识页面\n"
    "- 必要时用 read_wiki_page 读取页面全文\n"
    "- 只基于知识库内容回答，不要猜测\n"
    "- 引用信息时使用 [[page-name]] 格式\n"
    "- 如果知识库中没有找到相关信息，请如实说明"
)


class KnowledgeApiResponse(BaseModel):
    enabled: bool
    model_name: str
    base_url: str
    api_token_configured: bool
    agent_id: str | None = None
    agent_system_prompt: str = ""
    agent_max_tool_calls: int = 10


class KnowledgeApiUpdate(BaseModel):
    enabled: bool | None = None
    model_name: str | None = Field(default=None, max_length=128)
    system_prompt: str | None = None
    max_tool_calls: int | None = Field(default=None, ge=1, le=50)


class RegenerateTokenResponse(BaseModel):
    api_token: str


@router.get("", response_model=KnowledgeApiResponse)
async def get_knowledge_api(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user)
    proj = await service.get_project_or_404(db, project_id)

    agent_prompt = ""
    agent_max_calls = 10
    if proj.knowledge_agent_id:
        agent = (await db.execute(
            select(Agent).where(Agent.id == proj.knowledge_agent_id)
        )).scalar_one_or_none()
        if agent:
            agent_prompt = agent.system_prompt
            agent_max_calls = agent.max_tool_calls

    token_exists = (await db.execute(
        select(UserApiToken).where(UserApiToken.user_id == user.id)
    )).scalar_one_or_none() is not None

    return KnowledgeApiResponse(
        enabled=proj.knowledge_api_enabled,
        model_name=proj.knowledge_api_model_name or "",
        base_url="/v1",
        api_token_configured=token_exists,
        agent_id=proj.knowledge_agent_id,
        agent_system_prompt=agent_prompt,
        agent_max_tool_calls=agent_max_calls,
    )


@router.patch("", response_model=KnowledgeApiResponse)
async def update_knowledge_api(
    project_id: str,
    body: KnowledgeApiUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")
    proj = await service.get_project_or_404(db, project_id)

    if body.model_name is not None:
        model_name = body.model_name.strip()
        if model_name:
            existing = (await db.execute(
                select(Project).where(
                    Project.knowledge_api_model_name == model_name,
                    Project.id != project_id,
                )
            )).scalar_one_or_none()
            if existing:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"模型名 '{model_name}' 已被其他项目使用",
                )
        proj.knowledge_api_model_name = model_name

    if body.enabled is not None:
        proj.knowledge_api_enabled = body.enabled
        if body.enabled and not proj.knowledge_agent_id:
            agent = await _ensure_knowledge_agent(db, proj, user)
            proj.knowledge_agent_id = agent.id

    # Update agent settings if provided
    if proj.knowledge_agent_id and (body.system_prompt is not None or body.max_tool_calls is not None):
        agent = (await db.execute(
            select(Agent).where(Agent.id == proj.knowledge_agent_id)
        )).scalar_one_or_none()
        if agent:
            if body.system_prompt is not None:
                agent.system_prompt = body.system_prompt
            if body.max_tool_calls is not None:
                agent.max_tool_calls = body.max_tool_calls

    await db.commit()
    await db.refresh(proj)

    return await _build_response(db, proj, user)


@router.post("/regenerate-token", response_model=RegenerateTokenResponse)
async def regenerate_knowledge_token(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.check_membership(db, project_id, user, require="owner")

    from app.auth.deps import hash_api_token
    raw_token = generate_api_token()

    existing = (await db.execute(
        select(UserApiToken).where(UserApiToken.user_id == user.id)
    )).scalar_one_or_none()

    if existing:
        existing.token_hash = hash_api_token(raw_token)
    else:
        db.add(UserApiToken(user_id=user.id, token_hash=hash_api_token(raw_token)))

    await db.commit()
    return RegenerateTokenResponse(api_token=raw_token)


async def _ensure_knowledge_agent(db: AsyncSession, proj: Project, user: User) -> Agent:
    """Create a knowledge Agent for the project if it doesn't exist."""
    agent_id = str(uuid.uuid4())
    agent = Agent(
        id=agent_id,
        name=f"{proj.name} 知识检索",
        description=f"项目「{proj.name}」的知识检索 Agent（自动创建）",
        system_prompt=DEFAULT_KNOWLEDGE_SYSTEM_PROMPT,
        is_public=False,
        require_api_key=False,
        max_tool_calls=10,
        debug_result_limit=2000,
        tool_labels="{}",
        api_key_hash=None,
        created_by=user.id,
    )
    db.add(agent)
    db.add(AgentProject(agent_id=agent_id, project_id=proj.id))
    await db.flush()
    return agent


async def _build_response(db: AsyncSession, proj: Project, user: User) -> KnowledgeApiResponse:
    agent_prompt = ""
    agent_max_calls = 10
    if proj.knowledge_agent_id:
        agent = (await db.execute(
            select(Agent).where(Agent.id == proj.knowledge_agent_id)
        )).scalar_one_or_none()
        if agent:
            agent_prompt = agent.system_prompt
            agent_max_calls = agent.max_tool_calls

    token_exists = (await db.execute(
        select(UserApiToken).where(UserApiToken.user_id == user.id)
    )).scalar_one_or_none() is not None

    return KnowledgeApiResponse(
        enabled=proj.knowledge_api_enabled,
        model_name=proj.knowledge_api_model_name or "",
        base_url="/v1",
        api_token_configured=token_exists,
        agent_id=proj.knowledge_agent_id,
        agent_system_prompt=agent_prompt,
        agent_max_tool_calls=agent_max_calls,
    )
