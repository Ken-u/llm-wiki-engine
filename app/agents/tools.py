"""Agent tool implementations.

Each tool function takes validated arguments and returns a dict suitable for
JSON serialization back to the LLM as a tool result.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles

from app.projects.models import Project
from app.search.bm25 import search_bm25
from app.search.fusion import rrf_fusion
from app.search.vector import search_vector

logger = logging.getLogger(__name__)

MAX_PAGE_CHARS = 12000
MAX_RAW_READ_CHARS = 8000
MAX_GREP_MATCHES = 20


@dataclass
class ToolContext:
    """Runtime context passed to every tool invocation."""
    main_projects: list[Project]
    ticket_project: Project | None


def get_tool_definitions(ctx: ToolContext) -> list[dict]:
    """Return OpenAI-format tool definitions based on available context."""
    defs = [
        {
            "type": "function",
            "function": {
                "name": "search_wiki",
                "description": "在主知识库中进行语义搜索，返回最相关的 wiki 页面列表。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询词"},
                        "limit": {"type": "integer", "description": "返回结果数量上限", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_wiki_page",
                "description": "读取主知识库中指定路径的 wiki 页面全文内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "页面路径，如 wiki/concepts/gms.md"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wiki_index",
                "description": "获取主知识库的目录索引，帮助了解知识库的整体结构。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_purpose",
                "description": "获取主知识库的用途说明，帮助理解知识库的范围和目的。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_raw",
                "description": "直接读取知识库中指定文件的原始内容（带偏移和长度限制）。当索引搜索不到时，可用此工具查看原始文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径，如 wiki/concepts/gms.md"},
                        "offset": {"type": "integer", "description": "起始字符偏移量", "default": 0},
                        "limit": {"type": "integer", "description": "最大读取字符数（上限 8000）", "default": 4000},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_raw",
                "description": "在知识库原始文件中搜索包含指定关键词的行。当索引搜索找不到结果时，可用此工具直接全文搜索。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "搜索关键词或正则表达式"},
                        "glob": {"type": "string", "description": "文件匹配模式，如 *.md 或 wiki/**/*.md", "default": "**/*.md"},
                    },
                    "required": ["pattern"],
                },
            },
        },
    ]

    if ctx.ticket_project is not None:
        defs.extend([
            {
                "type": "function",
                "function": {
                    "name": "search_ticket_cases",
                    "description": "在绑定的案例库（ticket wiki）中搜索历史案例、故障经验、处理先例。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询词"},
                            "limit": {"type": "integer", "description": "返回结果数量上限", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_ticket_page",
                    "description": "读取案例库中指定路径的页面全文内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "页面路径，如 wiki/issues/boot-failure.md"},
                        },
                        "required": ["path"],
                    },
                },
            },
        ])

    return defs


async def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def _do_search(project: Project, query: str, limit: int, source_type: str) -> dict:
    """Run hybrid search (BM25 + vector + RRF) against a single project."""
    kw = search_bm25(project.disk_path, query, top_k=limit * 2)
    vec = await search_vector(project.disk_path, query, top_k=limit * 2)
    fused = rrf_fusion(kw, vec, query)[:limit]

    results = []
    for r in fused:
        results.append({
            "path": r.path,
            "title": r.title,
            "snippet": r.snippet[:300],
            "score": round(r.score, 4),
        })

    payload: dict[str, Any] = {
        "source_type": source_type,
        "results": results,
    }
    if source_type == "ticket":
        payload["project_id"] = project.id
        payload["project_name"] = project.name

    return payload


async def _do_read(project: Project, page_path: str, source_type: str) -> dict:
    """Read a wiki page from a project, with path traversal protection."""
    if ".." in page_path or page_path.startswith("/"):
        return {"error": "Invalid path"}

    full = Path(project.disk_path) / page_path
    if not full.exists():
        return {"error": f"Page not found: {page_path}"}

    content = await _read_file(full)
    content = content[:MAX_PAGE_CHARS]

    title = ""
    for line in content.split("\n")[:10]:
        if line.startswith("# "):
            title = line[2:].strip()
            break
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip().strip('"').strip("'")
            break

    return {
        "source_type": source_type,
        "path": page_path,
        "title": title,
        "content": content,
    }


async def _do_read_raw(ctx: ToolContext, arguments: dict) -> dict:
    """Read raw file content with offset/limit from any main project."""
    file_path = arguments.get("path", "")
    offset = max(0, arguments.get("offset", 0))
    limit = min(arguments.get("limit", 4000), MAX_RAW_READ_CHARS)

    if ".." in file_path or file_path.startswith("/"):
        return {"error": "Invalid path"}

    for proj in ctx.main_projects:
        full = Path(proj.disk_path) / file_path
        if full.exists() and full.is_file():
            content = await _read_file(full)
            total_len = len(content)
            chunk = content[offset : offset + limit]
            return {
                "path": file_path,
                "offset": offset,
                "length": len(chunk),
                "total_length": total_len,
                "has_more": (offset + limit) < total_len,
                "content": chunk,
            }

    return {"error": f"File not found: {file_path}"}


async def _do_grep_raw(ctx: ToolContext, arguments: dict) -> dict:
    """Grep through raw files in main projects for a pattern."""
    pattern_str = arguments.get("pattern", "")
    glob_pattern = arguments.get("glob", "**/*.md")

    if not pattern_str:
        return {"error": "Empty pattern"}

    try:
        regex = re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern_str), re.IGNORECASE)

    matches: list[dict] = []

    for proj in ctx.main_projects:
        base = Path(proj.disk_path)
        for file_path in sorted(base.glob(glob_pattern)):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_no, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    rel = str(file_path.relative_to(base))
                    matches.append({
                        "file": rel,
                        "line": line_no,
                        "text": line.strip()[:200],
                    })
                    if len(matches) >= MAX_GREP_MATCHES:
                        return {
                            "pattern": pattern_str,
                            "matches": matches,
                            "truncated": True,
                        }

    return {
        "pattern": pattern_str,
        "matches": matches,
        "truncated": False,
    }


async def execute_tool(name: str, arguments: dict, ctx: ToolContext) -> dict:
    """Execute a named tool with parsed arguments. Returns JSON-serializable result."""
    if name == "search_wiki":
        query = arguments.get("query", "")
        limit = min(arguments.get("limit", 5), 10)
        all_results: list[dict] = []
        for proj in ctx.main_projects:
            r = await _do_search(proj, query, limit, "wiki")
            all_results.extend(r["results"])
        all_results.sort(key=lambda x: -x.get("score", 0))
        return {"source_type": "wiki", "results": all_results[:limit]}

    if name == "read_wiki_page":
        page_path = arguments.get("path", "")
        for proj in ctx.main_projects:
            full = Path(proj.disk_path) / page_path
            if full.exists():
                return await _do_read(proj, page_path, "wiki")
        return {"error": f"Page not found: {page_path}"}

    if name == "get_wiki_index":
        parts = []
        for proj in ctx.main_projects:
            idx = await _read_file(Path(proj.disk_path) / "wiki" / "index.md")
            if idx:
                parts.append(f"## [{proj.name}]\n\n{idx[:4000]}")
        return {"source_type": "wiki", "content": "\n\n---\n\n".join(parts) or "No index available."}

    if name == "get_project_purpose":
        parts = []
        for proj in ctx.main_projects:
            purpose = await _read_file(Path(proj.disk_path) / "purpose.md")
            if purpose:
                parts.append(f"## [{proj.name}]\n\n{purpose[:2000]}")
        return {"source_type": "wiki", "content": "\n\n---\n\n".join(parts) or "No purpose file."}

    if name == "search_ticket_cases":
        if ctx.ticket_project is None:
            return {"error": "Ticket wiki not configured for this project."}
        query = arguments.get("query", "")
        limit = min(arguments.get("limit", 5), 10)
        return await _do_search(ctx.ticket_project, query, limit, "ticket")

    if name == "read_ticket_page":
        if ctx.ticket_project is None:
            return {"error": "Ticket wiki not configured for this project."}
        page_path = arguments.get("path", "")
        return await _do_read(ctx.ticket_project, page_path, "ticket")

    if name == "read_raw":
        return await _do_read_raw(ctx, arguments)

    if name == "grep_raw":
        return await _do_grep_raw(ctx, arguments)

    return {"error": f"Unknown tool: {name}"}
