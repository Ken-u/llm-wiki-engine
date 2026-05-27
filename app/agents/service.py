"""Agent service: CRUD + cross-project RAG."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import uuid
from typing import AsyncGenerator

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import Agent, AgentProject
from app.agents.orchestrator import run_agent_turn
from app.agents.tools import ToolContext
from app.chat.context import compute_context_budget, truncate_to_budget
from app.config import get_config
from app.llm import client as llm_client
from app.projects.models import Project
from app.search.bm25 import search_bm25
from app.search.fusion import rrf_fusion, FusedResult
from app.search.vector import search_vector
from app.wiki.graph import build_wiki_graph, graph_expand

import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return (raw_key, hashed_key)."""
    raw = f"lwk_{secrets.token_urlsafe(32)}"
    return raw, _hash_key(raw)


async def create_agent(
    db: AsyncSession,
    *,
    name: str,
    description: str,
    system_prompt: str,
    project_ids: list[str],
    is_public: bool,
    require_api_key: bool,
    max_tool_calls: int = 20,
    debug_result_limit: int = 2000,
    user_id: int,
) -> tuple[Agent, str | None]:
    """Create an agent. Returns (agent, raw_api_key_or_None)."""
    agent_id = str(uuid.uuid4())
    raw_key = None
    key_hash = None

    if is_public and require_api_key:
        raw_key, key_hash = generate_api_key()

    agent = Agent(
        id=agent_id,
        name=name,
        description=description,
        system_prompt=system_prompt,
        is_public=is_public,
        require_api_key=require_api_key,
        max_tool_calls=max_tool_calls,
        debug_result_limit=debug_result_limit,
        api_key_hash=key_hash,
        created_by=user_id,
    )
    db.add(agent)

    for pid in project_ids:
        db.add(AgentProject(agent_id=agent_id, project_id=pid))

    await db.commit()
    await db.refresh(agent)
    return agent, raw_key


async def update_agent(
    db: AsyncSession,
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    project_ids: list[str] | None = None,
    is_public: bool | None = None,
    require_api_key: bool | None = None,
    max_tool_calls: int | None = None,
    debug_result_limit: int | None = None,
) -> Agent:
    if name is not None:
        agent.name = name
    if description is not None:
        agent.description = description
    if system_prompt is not None:
        agent.system_prompt = system_prompt
    if is_public is not None:
        agent.is_public = is_public
    if require_api_key is not None:
        agent.require_api_key = require_api_key
    if max_tool_calls is not None:
        agent.max_tool_calls = max_tool_calls
    if debug_result_limit is not None:
        agent.debug_result_limit = debug_result_limit

    if project_ids is not None:
        await db.execute(delete(AgentProject).where(AgentProject.agent_id == agent.id))
        for pid in project_ids:
            db.add(AgentProject(agent_id=agent.id, project_id=pid))

    await db.commit()
    await db.refresh(agent)
    return agent


async def regenerate_key(db: AsyncSession, agent: Agent) -> str:
    raw_key, key_hash = generate_api_key()
    agent.api_key_hash = key_hash
    await db.commit()
    return raw_key


async def get_agent_projects(db: AsyncSession, agent_id: str) -> list[Project]:
    stmt = (
        select(Project)
        .join(AgentProject, Project.id == AgentProject.project_id)
        .where(AgentProject.agent_id == agent_id)
    )
    return list((await db.execute(stmt)).scalars().all())


async def verify_api_key(db: AsyncSession, agent: Agent, raw_key: str) -> bool:
    if not agent.require_api_key:
        return True
    if not agent.api_key_hash:
        return False
    return _hash_key(raw_key) == agent.api_key_hash


async def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def cross_project_rag(
    projects: list[Project],
    message: str,
    history: list[dict],
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """RAG across multiple projects, then stream LLM response."""
    cfg = get_config()

    # Phase 1+2: parallel hybrid search across all projects
    async def search_one(proj: Project):
        kw = search_bm25(proj.disk_path, message, top_k=10)
        vec = await search_vector(proj.disk_path, message, top_k=10)
        fused = rrf_fusion(kw, vec, message)[:5]
        return proj, fused

    results = await asyncio.gather(*[search_one(p) for p in projects])

    # Phase 3: Cross-project merge — simple interleave by score
    all_results: list[tuple[Project, FusedResult]] = []
    for proj, fused in results:
        for r in fused:
            all_results.append((proj, r))
    all_results.sort(key=lambda x: -x[1].score)

    # Phase 4: Graph expansion per project
    expanded_ids: dict[str, list[str]] = {}
    for proj, r in all_results[:10]:
        if proj.id not in expanded_ids:
            graph = build_wiki_graph(proj.disk_path)
            seed = [rr[1].page_id for rr in all_results if rr[0].id == proj.id]
            expanded_ids[proj.id] = graph_expand(seed, graph, max_related=2)

    # Phase 5: Load pages within budget
    budget = compute_context_budget(cfg.llm.max_context_size)
    max_chars = budget.page_tokens * 4
    pages_content: list[str] = []
    total_chars = 0

    seen_pages: set[str] = set()
    for proj, r in all_results[:15]:
        base = Path(proj.disk_path)
        page_path = base / r.path
        if r.path in seen_pages or not page_path.exists():
            continue
        content = await _read_file(page_path)
        if content and total_chars + len(content) <= max_chars:
            pages_content.append(f"### [{proj.name}] {r.title or r.page_id}\n\n{content}")
            total_chars += len(content)
            seen_pages.add(r.path)

    references = "\n\n---\n\n".join(pages_content) if pages_content else "没有找到相关知识。"

    # Phase 6: Assemble system prompt with agent's custom prompt
    full_system = "\n".join([
        system_prompt or "你是一个知识问答助手。",
        "",
        "基于以下知识库内容回答用户问题。如果知识库中没有相关信息，请如实说明。",
        "引用信息时使用 [[page-name]] 格式。",
        "",
        f"## 相关知识\n\n{references}",
    ])
    full_system = truncate_to_budget(full_system, budget.system_tokens + budget.page_tokens)

    # Truncate history
    history_truncated = history[-20:]

    messages = [{"role": "system", "content": full_system}]
    messages.extend(history_truncated)
    messages.append({"role": "user", "content": message})

    async for token in llm_client.stream(
        messages,
        temperature=cfg.llm.chat_temperature,
        max_tokens=4096,
    ):
        yield token


async def get_ticket_project(db, projects: list[Project]) -> Project | None:
    """If any main project has a ticket binding, load the ticket project."""
    from sqlalchemy import select as sa_select
    for proj in projects:
        if proj.ticket_project_id:
            ticket = (await db.execute(
                sa_select(Project).where(Project.id == proj.ticket_project_id)
            )).scalar_one_or_none()
            if ticket:
                return ticket
    return None


async def agent_toolcall_chat(
    db,
    projects: list[Project],
    message: str,
    history: list[dict],
    system_prompt: str,
    max_tool_calls: int = 20,
    debug_result_limit: int = 2000,
):
    """Agent chat via tool-calling orchestrator. Returns an async generator of SSE events."""
    ticket_project = await get_ticket_project(db, projects)

    ctx = ToolContext(
        main_projects=projects,
        ticket_project=ticket_project,
    )

    async for event in run_agent_turn(
        message, history, system_prompt, ctx,
        max_tool_calls=max_tool_calls,
        debug_result_limit=debug_result_limit,
    ):
        yield event
