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
from app.auth.deps import verify_password, create_access_token
from app.auth.models import User
from app.database import get_db

router = APIRouter(prefix="/api/public/agents", tags=["public-agents"])


class PublicChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class PublicAgentInfo(BaseModel):
    id: str
    name: str
    description: str
    require_api_key: bool


class PublicAuthRequest(BaseModel):
    username: str
    password: str


@router.get("/{agent_id}/info")
async def get_public_agent_info(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get public agent info without authentication."""
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if not agent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    if not agent.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Agent is not public")
    return PublicAgentInfo(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        require_api_key=agent.require_api_key,
    )


@router.post("/{agent_id}/auth")
async def public_agent_auth(
    agent_id: str,
    body: PublicAuthRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate for public agent chat using system account credentials."""
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if not agent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    if not agent.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Agent is not public")

    user = (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


async def _verify_public_agent_access(
    db: AsyncSession, agent_id: str, authorization: str | None,
) -> Agent:
    """Verify access to a public agent. Returns the agent if authorized."""
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if not agent:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    if not agent.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Agent is not public")

    if agent.require_api_key:
        if not authorization:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key required")
        raw_key = authorization.replace("Bearer ", "").strip()
        # Try API key first, then JWT token
        if not await service.verify_api_key(db, agent, raw_key):
            # Try as JWT token
            from app.auth.deps import _decode_token
            try:
                user_id = _decode_token(raw_key)
                user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
                if user is None:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
            except HTTPException:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key or token")

    return agent


@router.post("/{agent_id}/chat")
async def public_agent_chat(
    agent_id: str,
    body: PublicChatRequest,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    agent = await _verify_public_agent_access(db, agent_id, authorization)

    projects = await service.get_agent_projects(db, agent.id)
    if not projects:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Agent has no knowledge bases")

    async def sse_stream():
        async for event in service.agent_toolcall_chat(
            db, projects, body.message, [], agent.system_prompt,
            max_tool_calls=agent.max_tool_calls,
        ):
            yield f"data: {event}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@router.get("/{agent_id}/wiki")
async def public_agent_wiki_tree(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    """Get wiki file tree for a public agent's projects."""
    agent = await _verify_public_agent_access(db, agent_id, authorization)
    projects = await service.get_agent_projects(db, agent.id)
    if not projects:
        return []

    from pathlib import Path as FsPath
    from app.wiki.frontmatter import parse_frontmatter

    def build_tree(wiki_dir: FsPath, base: FsPath) -> list[dict]:
        items = []
        if not wiki_dir.exists():
            return items
        for entry in sorted(wiki_dir.iterdir()):
            if entry.name.startswith("."):
                continue
            rel = str(entry.relative_to(base))
            if entry.is_dir():
                items.append({"name": entry.name, "path": rel, "type": "directory", "children": build_tree(entry, base)})
            elif entry.suffix == ".md":
                title = ""
                try:
                    content = entry.read_text(encoding="utf-8", errors="replace")
                    meta, _ = parse_frontmatter(content)
                    title = meta.title
                except Exception:
                    pass
                items.append({"name": entry.name, "path": rel, "type": "file", "title": title})
        return items

    all_tree: list[dict] = []
    for proj in projects:
        base = FsPath(proj.disk_path)
        wiki_dir = base / "wiki"
        all_tree.extend(build_tree(wiki_dir, base))
    return all_tree


@router.get("/{agent_id}/wiki/{path:path}")
async def public_agent_wiki_page(
    agent_id: str,
    path: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    """Read a wiki page for a public agent's projects."""
    agent = await _verify_public_agent_access(db, agent_id, authorization)
    projects = await service.get_agent_projects(db, agent.id)

    if ".." in path or path.startswith("/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")

    from pathlib import Path as FsPath
    import aiofiles
    from app.wiki.frontmatter import parse_frontmatter

    for proj in projects:
        full_path = FsPath(proj.disk_path) / path
        if full_path.exists():
            async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
                content = await f.read()
            meta, _ = parse_frontmatter(content)
            return {"path": path, "content": content, "meta": meta.raw}

    raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
