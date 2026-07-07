"""Chat API with SSE streaming + conversation persistence."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.agents import service as agent_service
from app.auth.deps import get_current_user
from app.auth.models import User
from app.config import get_config
from app.database import get_db
from app.projects.service import check_membership, get_project_or_404

router = APIRouter(prefix="/api/projects/{project_id}", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    use_fast_model: bool = False


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


def _conv_dir(project_dir: str) -> Path:
    p = Path(project_dir) / ".llm-wiki" / "chats"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _conv_list_path(project_dir: str) -> Path:
    return Path(project_dir) / ".llm-wiki" / "conversations.json"


async def _load_conversations(project_dir: str) -> list[dict]:
    p = _conv_list_path(project_dir)
    if not p.exists():
        return []
    async with aiofiles.open(p, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def _save_conversations(project_dir: str, convs: list[dict]) -> None:
    p = _conv_list_path(project_dir)
    async with aiofiles.open(p, "w", encoding="utf-8") as f:
        await f.write(json.dumps(convs, ensure_ascii=False, indent=2))


async def _load_messages(project_dir: str, conv_id: str) -> list[dict]:
    p = _conv_dir(project_dir) / f"{conv_id}.json"
    if not p.exists():
        return []
    async with aiofiles.open(p, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def _save_messages(project_dir: str, conv_id: str, messages: list[dict]) -> None:
    p = _conv_dir(project_dir) / f"{conv_id}.json"
    async with aiofiles.open(p, "w", encoding="utf-8") as f:
        await f.write(json.dumps(messages, ensure_ascii=False, indent=2))


@router.post("/chat")
async def chat(
    project_id: str,
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    conv_id = body.conversation_id or str(uuid.uuid4())
    history = await _load_messages(project.disk_path, conv_id)

    # Convert to LLM format
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]

    async def should_cancel() -> bool:
        return await request.is_disconnected()

    async def sse_stream():
        collected_tokens: list[str] = []
        collected_traces: list[dict] = []
        persisted = False

        async def persist_messages(*, compressed_history: list[dict] | None = None) -> None:
            nonlocal persisted
            if persisted:
                return
            persisted = True
            assistant_response = "".join(collected_tokens)

            if compressed_history is not None:
                llm_base = [
                    {"role": m["role"], "content": m["content"]}
                    for m in compressed_history
                    if m.get("role") in ("user", "assistant")
                ]
            else:
                llm_base = [
                    {"role": m["role"], "content": m["content"]}
                    for m in history
                    if m.get("role") in ("user", "assistant")
                ]

            now = datetime.now(timezone.utc).isoformat()
            new_messages = llm_base + [
                {"role": "user", "content": body.message, "timestamp": now},
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "rawContent": assistant_response,
                    "timestamp": now,
                },
            ]
            await _save_messages(project.disk_path, conv_id, new_messages)

            convs = await _load_conversations(project.disk_path)
            existing = next((c for c in convs if c["id"] == conv_id), None)
            if existing:
                existing["message_count"] = len(new_messages)
            else:
                convs.append({
                    "id": conv_id,
                    "title": body.message[:80],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message_count": len(new_messages),
                    "user_id": user.id,
                })
            await _save_conversations(project.disk_path, convs)

        try:
            async for event in agent_service.agent_toolcall_chat(
                db,
                [project],
                body.message,
                llm_history,
                "",
                should_cancel=should_cancel,
                use_fast_model=body.use_fast_model,
            ):
                payload = json.loads(event)
                if "token" in payload:
                    collected_tokens.append(payload["token"])
                if payload.get("done"):
                    collected_traces[:] = payload.get("tool_traces", [])
                    await persist_messages(
                        compressed_history=payload.get("compressed_history"),
                    )
                    payload["conversation_id"] = conv_id
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if await should_cancel():
                return

            await persist_messages()
            if collected_traces:
                from app.feedback.queue import maybe_trigger_feedback
                asyncio.create_task(
                    maybe_trigger_feedback(
                        project_id=project.id,
                        conversation_id=conv_id,
                        agent_id=None,
                        user_message=body.message,
                        assistant_answer="".join(collected_tokens),
                        tool_traces=collected_traces,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@router.get("/chat/options")
async def chat_options(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    await get_project_or_404(db, project_id)
    cfg = get_config().llm
    return {"fast_model_enabled": bool(cfg.fast_model)}


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    convs = await _load_conversations(project.disk_path)
    return [
        ConversationResponse(
            id=c["id"],
            title=c.get("title", ""),
            created_at=c.get("created_at", ""),
            message_count=c.get("message_count", 0),
        )
        for c in convs
    ]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    project_id: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)

    msg_file = _conv_dir(project.disk_path) / f"{conversation_id}.json"
    if msg_file.exists():
        msg_file.unlink()

    convs = await _load_conversations(project.disk_path)
    convs = [c for c in convs if c["id"] != conversation_id]
    await _save_conversations(project.disk_path, convs)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    project_id: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_membership(db, project_id, user)
    project = await get_project_or_404(db, project_id)
    messages = await _load_messages(project.disk_path, conversation_id)
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return {"id": conversation_id, "messages": messages}
