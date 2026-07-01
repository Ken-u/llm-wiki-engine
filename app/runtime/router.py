"""Read-only runtime API for local wiki inference."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

import aiofiles
import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse, StreamingResponse

from app.agents.orchestrator import run_agent_turn
from app.agents.tools import ToolContext
from app.case_index.builder import load_manifest
from app.case_index.search import read_case_source, search_cases
from app.knowledge.fast_lookup import (
    extract_term_from_definition_query,
    fast_lookup,
    is_definition_query,
)
from app.knowledge.lookup import _format_fast_result
from app.runtime.config import get_runtime_config, get_runtime_config_path, load_runtime_config
from app.runtime.hooks import run_startup_hooks
from app.runtime.projects import (
    build_wiki_tree,
    get_case_project,
    get_knowledge_project,
    safe_join,
)
from app.runtime.security import require_runtime_api_key
from app.runtime.status import build_status
from app.search.bm25 import search_bm25
from app.search.fusion import FusedResult, rrf_fusion
from app.search.vector import search_vector
from app.wiki.frontmatter import parse_frontmatter

router = APIRouter(prefix="/api", tags=["runtime"], dependencies=[Depends(require_runtime_api_key)])


class RuntimeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid"


class RuntimeCaseSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=20)


class RuntimeChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    mode: Literal["auto", "agent", "fast", "rag"] | None = None
    stream: bool = True
    debug: bool = True


def _redact_config(data: dict[str, Any]) -> dict[str, Any]:
    for section in ("llm", "embedding"):
        key = data.get(section, {}).get("api_key", "")
        if key:
            data[section]["api_key"] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
        data.get(section, {})["api_key_configured"] = bool(key)
    return data


def _preserve_redacted_keys(body: dict[str, Any]) -> dict[str, Any]:
    current = get_runtime_config()
    for section, current_key in (
        ("llm", current.llm.api_key),
        ("embedding", current.embedding.api_key),
    ):
        value = body.get(section, {}).get("api_key")
        if value in ("***", ""):
            body.setdefault(section, {})["api_key"] = current_key
        elif isinstance(value, str) and "..." in value:
            body.setdefault(section, {})["api_key"] = current_key
    return body


@router.get("/status")
async def status_response():
    return await build_status(get_runtime_config())


@router.get("/config")
async def get_config_response():
    return _redact_config(get_runtime_config().model_dump())


@router.put("/config")
async def update_config_response(body: dict[str, Any]):
    path = get_runtime_config_path()
    if path is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Runtime config path not initialized")
    body = _preserve_redacted_keys(body)
    path.write_text(
        yaml.safe_dump(body, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return _redact_config(load_runtime_config(path).model_dump())


@router.post("/hooks/run")
async def run_hooks_response():
    return {"results": [r.__dict__ for r in await run_startup_hooks(get_runtime_config())]}


@router.post("/search")
async def search_response(body: RuntimeSearchRequest):
    project = get_knowledge_project()
    kw_results = []
    vec_results = []

    if body.mode in ("hybrid", "keyword"):
        kw_results = search_bm25(project.disk_path, body.query, top_k=body.top_k * 2)

    if body.mode in ("hybrid", "vector"):
        vec_results = await search_vector(project.disk_path, body.query, top_k=body.top_k * 2)

    if body.mode == "keyword":
        fused = [
            FusedResult(
                path=r.path,
                page_id=r.page_id,
                title=r.title,
                score=r.score,
                snippet=r.snippet,
                sources=["keyword"],
            )
            for r in kw_results[:body.top_k]
        ]
    elif body.mode == "vector":
        fused = [
            FusedResult(
                path=r.path,
                page_id=r.page_id,
                title=r.page_id,
                score=r.score,
                snippet=r.chunk_text[:200],
                sources=["vector"],
            )
            for r in vec_results[:body.top_k]
        ]
    else:
        fused = rrf_fusion(kw_results, vec_results, body.query)[:body.top_k]

    return {
        "results": [f.__dict__ for f in fused],
        "mode": body.mode,
        "keyword_hits": len(kw_results),
        "vector_hits": len(vec_results),
    }


@router.post("/cases/search")
async def case_search_response(body: RuntimeCaseSearchRequest):
    case_project = get_case_project()
    if case_project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case library is disabled")
    manifest = load_manifest(case_project.disk_path)
    if manifest is None or not manifest.is_ready:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Case index is not ready")
    results = await search_cases(case_project.disk_path, body.query, limit=body.limit)
    return {
        "results": [
            {
                "case_id": r.case_id,
                "title": r.title,
                "domain": r.domain,
                "problem_summary": r.problem_summary,
                "root_cause": r.root_cause,
                "resolution": r.resolution,
                "matched_sections": [s.__dict__ for s in r.matched_sections],
                "score": r.score,
            }
            for r in results
        ]
    }


@router.get("/wiki")
async def wiki_tree_response():
    return build_wiki_tree(get_knowledge_project())


@router.get("/wiki/{path:path}")
async def wiki_page_response(path: str):
    project = get_knowledge_project()
    full_path = safe_join(project.disk_path, path)
    if full_path is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
    content = await _read_text(full_path)
    meta, _ = parse_frontmatter(content)
    return {"path": path, "content": content, "meta": meta.raw}


@router.get("/cases/{case_id}")
async def case_response(case_id: str):
    case_project = get_case_project()
    if case_project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case library is disabled")
    result = read_case_source(case_project.disk_path, case_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Case not found: {case_id}")
    return result


@router.post("/chat")
async def chat_response(body: RuntimeChatRequest):
    if body.stream:
        return StreamingResponse(_chat_sse(body), media_type="text/event-stream")

    answer_parts: list[str] = []
    tool_traces: list[dict] = []
    references: list[dict] = []
    used_case_library = False
    async for event in _chat_events(body):
        if event["event"] == "token":
            answer_parts.append(event["data"].get("text", ""))
        elif event["event"] == "tool_call":
            tool_traces.append(event["data"])
        elif event["event"] == "tool_result":
            refs, used_case = _references_from_tool_result(event["data"])
            references.extend(refs)
            used_case_library = used_case_library or used_case
        elif event["event"] == "done":
            tool_traces = event["data"].get("tool_traces", tool_traces)
            references.extend(event["data"].get("references", []))
            used_case_library = used_case_library or event["data"].get("used_case_library", False)
    return {
        "answer": "".join(answer_parts),
        "tool_traces": tool_traces,
        "references": _dedupe_references(references),
        "used_case_library": used_case_library,
        "mode": body.mode or get_runtime_config().runtime.mode,
    }


async def _read_text(path: Path) -> str:
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def _chat_sse(body: RuntimeChatRequest) -> AsyncGenerator[str, None]:
    try:
        async for event in _chat_events(body):
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
    except Exception as exc:
        yield "event: error\n"
        yield f"data: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"


async def _chat_events(body: RuntimeChatRequest) -> AsyncGenerator[dict, None]:
    settings = get_runtime_config()
    mode = body.mode or settings.runtime.mode
    project = get_knowledge_project(settings)
    case_project = get_case_project(settings)

    if mode in ("auto", "fast") and is_definition_query(body.message):
        term = extract_term_from_definition_query(body.message)
        result = fast_lookup(project.disk_path, term)
        if result is not None:
            yield {"event": "token", "data": {"text": _format_fast_result(result)}}
            yield {
                "event": "done",
                "data": {
                    "tool_traces": [],
                    "references": [{"type": "wiki", "path": result.path, "title": result.title}],
                    "used_case_library": False,
                    "mode": "fast",
                },
            }
            return
        if mode == "fast":
            yield {"event": "token", "data": {"text": "知识库中未找到相关信息。"}}
            yield {"event": "done", "data": {"tool_traces": [], "references": [], "mode": "fast"}}
            return

    ctx = ToolContext(main_projects=[project], ticket_project=case_project)
    references: list[dict] = []
    used_case_library = False
    traces: list[dict] = []

    async for raw_event in run_agent_turn(
        body.message,
        body.history,
        settings.knowledge.system_prompt,
        ctx,
        max_tool_calls=settings.runtime.max_tool_calls,
        debug_result_limit=settings.runtime.debug_result_limit,
    ):
        payload = json.loads(raw_event)
        if "token" in payload:
            yield {"event": "token", "data": {"text": payload["token"]}}
        elif "tool_call" in payload:
            traces.append(payload["tool_call"])
            yield {"event": "tool_call", "data": payload["tool_call"]}
        elif "tool_result" in payload:
            refs, used_case = _references_from_tool_result(payload["tool_result"])
            references.extend(refs)
            used_case_library = used_case_library or used_case
            yield {"event": "tool_result", "data": payload["tool_result"]}
        elif payload.get("done"):
            if payload.get("tool_traces"):
                traces = payload["tool_traces"]
            yield {
                "event": "done",
                "data": {
                    "tool_traces": traces,
                    "references": _dedupe_references(references),
                    "used_case_library": used_case_library,
                    "context_compressed": payload.get("context_compressed", False),
                    "mode": "agent",
                },
            }


def _references_from_tool_result(tool_result: dict) -> tuple[list[dict], bool]:
    name = tool_result.get("name", "")
    result = tool_result.get("result", {})
    refs: list[dict] = []
    used_case = False

    if name in ("search_wiki", "read_wiki_page"):
        if result.get("path"):
            refs.append({"type": "wiki", "path": result["path"], "title": result.get("title", "")})
        for item in result.get("results", []) or []:
            if item.get("path"):
                refs.append({"type": "wiki", "path": item["path"], "title": item.get("title", "")})

    if name in ("search_ticket_cases", "read_ticket_case"):
        used_case = True
        if result.get("case_id"):
            refs.append({"type": "case", "case_id": result["case_id"], "title": result.get("title", "")})
        for item in result.get("results", []) or []:
            if item.get("case_id"):
                refs.append({"type": "case", "case_id": item["case_id"], "title": item.get("title", "")})

    return refs, used_case


def _dedupe_references(references: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for ref in references:
        key = (ref.get("type"), ref.get("path") or ref.get("case_id"))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(ref)
    return out


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    functions: list | None = None


openai_router = APIRouter(prefix="/v1", tags=["runtime-openai"], dependencies=[Depends(require_runtime_api_key)])


@openai_router.get("/models")
async def models_response():
    settings = get_runtime_config()
    return {
        "object": "list",
        "data": [{
            "id": settings.knowledge.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "llm-wiki-runtime",
        }],
    }


@openai_router.post("/chat/completions")
async def openai_chat_response(body: OpenAIChatRequest):
    if body.tools or body.tool_choice or body.functions:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Runtime does not support external tool calls on the OpenAI-compatible endpoint.",
        )

    settings = get_runtime_config()
    if body.model != settings.knowledge.model_name:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model not found: {body.model}")

    user_message = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if not user_message:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages must include a user message")

    history = [
        {"role": m.role, "content": m.content}
        for m in body.messages[:-1]
        if m.role in ("user", "assistant")
    ]
    runtime_body = RuntimeChatRequest(message=user_message, history=history, stream=body.stream)

    if body.stream:
        return StreamingResponse(_openai_sse(runtime_body, body.model), media_type="text/event-stream")

    answer_parts: list[str] = []
    async for event in _chat_events(runtime_body):
        if event["event"] == "token":
            answer_parts.append(event["data"].get("text", ""))

    content = "".join(answer_parts)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _openai_sse(body: RuntimeChatRequest, model: str) -> AsyncGenerator[str, None]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    async for event in _chat_events(body):
        if event["event"] != "token":
            continue
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": event["data"].get("text", "")},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    done_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/raw/{path:path}")
async def raw_file_response(path: str):
    project = get_knowledge_project()
    full_path = safe_join(project.disk_path, f"raw/sources/{path}")
    if full_path is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source file not found")
    return PlainTextResponse(await _read_text(full_path))
