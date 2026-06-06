"""OpenAI-compatible /v1/chat/completions and /v1/models endpoints.

Routes requests by virtual model_name to the correct wiki project, then
uses fast-path or Agent-based lookup. Does NOT support client-side tool calls.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import Agent
from app.auth.deps import get_current_user
from app.auth.models import User
from app.database import get_db
from app.knowledge.lookup import knowledge_lookup
from app.projects.models import Project
from app.projects.service import check_membership

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# --- Request / Response models ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # Explicitly typed so we can detect and reject
    tools: list | None = None
    tool_choice: str | dict | None = None
    functions: list | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars count as 2 tokens, others as ~0.25 per char."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    rest = len(text) - cjk
    return cjk * 2 + max(1, rest // 4)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = UsageInfo()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llm-wiki-engine"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# --- Endpoints ---

@router.get("/models", response_model=ModelsResponse)
async def list_models(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all enabled virtual knowledge models visible to the current user."""
    projects = (await db.execute(
        select(Project).where(
            Project.knowledge_api_enabled == True,  # noqa: E712
            Project.knowledge_api_model_name != "",
        )
    )).scalars().all()

    models = []
    for proj in projects:
        models.append(ModelInfo(
            id=proj.knowledge_api_model_name,
            created=int(proj.created_at.timestamp()) if proj.created_at else 0,
        ))

    return ModelsResponse(data=models)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """OpenAI-compatible chat completion backed by wiki knowledge lookup."""
    # Reject external tool calls
    if body.tools or body.tool_choice or body.functions:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "本接口不支持外部 tool call。请移除 tools/tool_choice/functions 参数。",
        )

    if body.stream:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "暂不支持流式响应，请设置 stream=false。",
        )

    # Resolve model_name -> project
    project = (await db.execute(
        select(Project).where(
            Project.knowledge_api_model_name == body.model,
            Project.knowledge_api_enabled == True,  # noqa: E712
        )
    )).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"未找到模型 '{body.model}'，请确认模型名称正确且已启用。",
        )

    # Verify membership
    await check_membership(db, project.id, user)

    # Extract user message (last message with role=user)
    user_message = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages 中缺少 user 角色消息。")

    # Get agent config
    system_prompt = ""
    max_tool_calls = 10
    if project.knowledge_agent_id:
        agent = (await db.execute(
            select(Agent).where(Agent.id == project.knowledge_agent_id)
        )).scalar_one_or_none()
        if agent:
            system_prompt = agent.system_prompt
            max_tool_calls = agent.max_tool_calls

    # Run knowledge lookup
    content = await knowledge_lookup(
        project, user_message, system_prompt, max_tool_calls=max_tool_calls,
    )

    # Estimate token usage
    prompt_text = " ".join(msg.content for msg in body.messages)
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(content)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
