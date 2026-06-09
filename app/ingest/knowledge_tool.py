"""Main knowledge lookup tool support for case-library ingest."""

from __future__ import annotations

import json
import logging

from sqlalchemy import select

from app.database import async_session
from app.knowledge.lookup import knowledge_lookup
from app.llm import client as llm_client
from app.projects.knowledge_router import DEFAULT_KNOWLEDGE_SYSTEM_PROMPT
from app.projects.models import Project

logger = logging.getLogger(__name__)

MAIN_KNOWLEDGE_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_main_knowledge_term",
        "description": "查询关联主知识库中术语、业务概念或缩写的含义。仅在案例库编译时遇到不确定名词时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "需要查询的术语、概念或缩写。",
                },
                "question": {
                    "type": "string",
                    "description": "可选。围绕该术语的具体问题，例如它在当前案例中的含义。",
                },
            },
            "required": ["term"],
        },
    },
}

_TOOL_SYSTEM_HINT = (
    "\n\n案例库编译工具：如果源文档中出现不清楚的业务术语、系统名、缩写或概念，"
    "可以调用 lookup_main_knowledge_term 查询关联主知识库。只在需要确认含义时调用；"
    "不要猜测术语定义，查询结果只能作为理解上下文，最终输出仍必须遵守本阶段的格式要求。"
)


async def get_case_library_main_projects(case_project_id: str) -> list[Project]:
    """Return enabled main knowledge projects that use this project as a case library."""
    async with async_session() as db:
        rows = await db.execute(
            select(Project)
            .where(
                Project.ticket_project_id == case_project_id,
                Project.knowledge_api_enabled.is_(True),
            )
            .order_by(Project.name.asc(), Project.id.asc())
        )
        return list(rows.scalars().all())


async def collect_with_main_knowledge_tools(
    system: str,
    user: str,
    main_projects: list[Project],
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
    max_tool_rounds: int = 4,
) -> str:
    """Collect an ingest LLM response, allowing term lookup against linked main projects."""
    if not main_projects:
        return await llm_client.stream_collect(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    messages: list[dict] = [
        {"role": "system", "content": f"{system}{_TOOL_SYSTEM_HINT}"},
        {"role": "user", "content": user},
    ]
    tools = [MAIN_KNOWLEDGE_LOOKUP_TOOL]

    for _ in range(max_tool_rounds + 1):
        resp = await llm_client.complete_with_tools(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not resp.tool_calls:
            return resp.content or ""

        assistant_message: dict = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in resp.tool_calls
            ],
        }
        messages.append(assistant_message)

        for call in resp.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": await _execute_tool_call(call.name, call.arguments, main_projects),
            })

    logger.warning("Main knowledge lookup tool rounds exhausted during ingest")
    return ""


async def _execute_tool_call(name: str, raw_arguments: str, main_projects: list[Project]) -> str:
    if name != "lookup_main_knowledge_term":
        return f"Unsupported tool: {name}"

    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return "工具参数不是有效 JSON，无法查询。"

    term = str(args.get("term") or "").strip()
    if not term:
        return "缺少 term 参数，无法查询。"

    question = str(args.get("question") or "").strip()
    message = question or f"{term} 是什么？"
    return await _lookup_across_main_projects(term, message, main_projects)


async def _lookup_across_main_projects(term: str, message: str, main_projects: list[Project]) -> str:
    results: list[str] = []
    for project in main_projects:
        try:
            answer = await knowledge_lookup(
                project,
                message,
                DEFAULT_KNOWLEDGE_SYSTEM_PROMPT,
                max_tool_calls=10,
            )
        except Exception as exc:
            logger.warning(
                "Main knowledge lookup failed during ingest: project=%s term=%r error=%s",
                project.id,
                term,
                exc,
            )
            answer = f"查询失败：{exc}"
        results.append(f"## {project.name}\n{answer}")

    return "\n\n".join(results) if results else f"未找到可查询的主知识库：{term}"
