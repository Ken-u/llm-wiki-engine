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

from app.case_index.builder import load_manifest
from app.case_index.search import search_cases, read_case
from app.projects.models import Project
from app.search.bm25 import search_bm25
from app.search.fusion import rrf_fusion
from app.search.vector import search_vector

logger = logging.getLogger(__name__)

MAX_PAGE_CHARS = 12000
MAX_RAW_READ_LINES = 200
MAX_GREP_MATCHES = 20
RAW_SUBDIR = "raw/sources"


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
                "description": "在主知识库中进行语义搜索。仅当对话历史中尚无相关信息时使用，勿对追问/澄清重复搜索。",
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
                "description": "读取主知识库 wiki 页面全文。仅当历史中未包含该页内容且 search 摘要不足时使用，勿重复读取已在对话中出现过的页面。",
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
                "description": "按行读取项目原始源文件内容（raw/sources 目录下）。路径相对于 raw/sources，如 Android_GMS_Developer_Guide_CN.md。可配合 grep_raw 返回的行号定位读取。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径，相对于 raw/sources，如 Android_GMS_Developer_Guide_CN.md"},
                        "start_line": {"type": "integer", "description": "起始行号（1-based）", "default": 1},
                        "line_count": {"type": "integer", "description": "读取行数（上限 200）", "default": 100},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_raw",
                "description": "在项目原始源文件（raw/sources 目录下）中搜索包含指定关键词的行，返回文件名和行号。当索引搜索找不到结果时，可用此工具直接全文搜索。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "搜索关键词或正则表达式"},
                        "glob": {"type": "string", "description": "文件匹配模式", "default": "**/*.md"},
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
                    "description": "搜索案例库。仅当用户问新案例/新故障主题且对话历史未涵盖时使用，勿对追问已讨论案例重复搜索。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询词"},
                            "limit": {"type": "integer", "description": "返回结果数量上限", "default": 3},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_ticket_case",
                    "description": "读取案例详情。仅当 search 摘要不足且该案例未在对话历史中完整出现时使用，勿重复读取已讨论过的章节。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string", "description": "案例 ID"},
                            "section": {
                                "type": "string",
                                "description": "章节名，如'处理过程'、'最终处理方案'、'原因分析'。不传则返回全部章节摘要。",
                            },
                        },
                        "required": ["case_id"],
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
    """Read raw source file by line range from raw/sources/ directory."""
    file_path = arguments.get("path", "")
    start_line = max(1, arguments.get("start_line", 1))
    line_count = min(max(1, arguments.get("line_count", 100)), MAX_RAW_READ_LINES)

    if ".." in file_path or file_path.startswith("/"):
        return {"error": "Invalid path"}

    for proj in ctx.main_projects:
        full = Path(proj.disk_path) / RAW_SUBDIR / file_path
        if full.exists() and full.is_file():
            content = await _read_file(full)
            all_lines = content.split("\n")
            total_lines = len(all_lines)
            start_idx = start_line - 1
            selected = all_lines[start_idx : start_idx + line_count]
            return {
                "path": file_path,
                "start_line": start_line,
                "lines_returned": len(selected),
                "total_lines": total_lines,
                "has_more": (start_idx + line_count) < total_lines,
                "content": "\n".join(selected),
            }

    return {"error": f"File not found in raw/sources: {file_path}"}


async def _do_grep_raw(ctx: ToolContext, arguments: dict) -> dict:
    """Grep through raw/sources/ files in main projects for a pattern."""
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
        raw_base = Path(proj.disk_path) / RAW_SUBDIR
        if not raw_base.exists():
            continue
        for file_path in sorted(raw_base.glob(glob_pattern)):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_no, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    rel = str(file_path.relative_to(raw_base))
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
        limit = min(arguments.get("limit", 3), 5)

        manifest = load_manifest(ctx.ticket_project.disk_path)
        if manifest is None or not manifest.is_ready:
            return {"error": "Case index is not built or not ready. Rebuild the case index first."}

        results = await search_cases(ctx.ticket_project.disk_path, query, limit=limit)
        return {
            "source_type": "ticket_case_index",
            "results": [
                {
                    "case_id": r.case_id,
                    "title": r.title,
                    "domain": r.domain,
                    "problem_summary": r.problem_summary,
                    "root_cause": r.root_cause,
                    "resolution": r.resolution,
                    "matched_sections": [
                        {"section": s.section, "snippet": s.snippet}
                        for s in r.matched_sections
                    ],
                    "score": r.score,
                }
                for r in results
            ],
            "usage_hint": (
                "If the summaries are enough, answer directly. "
                "Use read_ticket_case(case_id, section) for section details. "
                "Do NOT use read_raw, grep_raw, or read_wiki_page for case content."
            ),
        }

    if name == "read_ticket_case":
        if ctx.ticket_project is None:
            return {"error": "Ticket wiki not configured for this project."}
        case_id = arguments.get("case_id", "")
        section = (
            arguments.get("section")
            or arguments.get("session")
            or arguments.get("section_name")
        )
        result = read_case(ctx.ticket_project.disk_path, case_id, section=section)
        if result is None:
            return {"error": f"Case not found: {case_id}"}
        return result

    if name == "read_raw":
        return await _do_read_raw(ctx, arguments)

    if name == "grep_raw":
        return await _do_grep_raw(ctx, arguments)

    return {"error": f"Unknown tool: {name}"}
