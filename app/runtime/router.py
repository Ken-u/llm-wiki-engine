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
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse, StreamingResponse

from app.agents.orchestrator import ShouldCancel, _build_system_prompt, run_agent_turn
from app.agents.tools import ToolContext
from app.case_index.builder import load_manifest
from app.case_index.search import read_case_source, search_cases
from app.index_tasks import (
    IndexRebuildTaskResponse,
    create_index_task,
    get_active_index_task,
    get_index_task,
    has_active_index_task,
    set_index_task,
)
from app.knowledge.fast_lookup import (
    extract_term_from_definition_query,
    fast_lookup,
    is_definition_query,
)
from app.knowledge.lookup import _format_fast_result
from app.agents.skill_refs import sign_source_ref, verify_source_ref
from app.http.external_url import external_base_url
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
from app.wiki.resolve import resolve_missing_wiki_page

router = APIRouter(prefix="/api", tags=["runtime"], dependencies=[Depends(require_runtime_api_key)])


class RuntimeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid"


class RuntimeReindexResponse(BaseModel):
    knowledge: dict[str, int] | None = None
    cases: dict[str, Any] | None = None


class RuntimeReindexRequest(BaseModel):
    target: Literal["knowledge", "cases", "all"] = "knowledge"


class RuntimeCaseSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=20)


class RuntimeChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    mode: Literal["auto", "agent", "fast", "rag"] | None = None
    stream: bool = True
    debug: bool = True


class RuntimeSkillChatRequest(BaseModel):
    message: str = Field(min_length=1)


class RuntimeSkillSource(BaseModel):
    name: str
    source_ref: str


class RuntimeSkillChatResponse(BaseModel):
    answer: str
    sources: list[RuntimeSkillSource] = Field(default_factory=list)


class RuntimeYamlConfigResponse(BaseModel):
    path: str
    content: str


class RuntimeSystemPromptConfigResponse(BaseModel):
    path: str
    system_prompt: str
    system_prompt_override: str
    default_system_prompt: str


class RuntimeSystemPromptConfigUpdate(BaseModel):
    system_prompt: str = ""
    system_prompt_override: str = ""


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


def _runtime_config_path_or_500() -> Path:
    path = get_runtime_config_path()
    if path is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Runtime config path not initialized")
    return path


def _dump_runtime_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def _build_yaml_config_response(path: Path) -> RuntimeYamlConfigResponse:
    redacted = _redact_config(get_runtime_config().model_dump())
    return RuntimeYamlConfigResponse(path=str(path), content=_dump_runtime_yaml(redacted))


def _build_system_prompt_config_response(path: Path) -> RuntimeSystemPromptConfigResponse:
    settings = get_runtime_config()
    return RuntimeSystemPromptConfigResponse(
        path=str(path),
        system_prompt=settings.knowledge.system_prompt,
        system_prompt_override=settings.knowledge.system_prompt_override,
        default_system_prompt=_build_system_prompt("", has_ticket=get_case_project(settings) is not None),
    )


def sign_runtime_source_ref(doc_name: str) -> str:
    return sign_source_ref(
        agent_id="runtime",
        project_id="knowledge",
        doc_name=doc_name,
        display_name=doc_name,
    )


def _runtime_source_from_tool_result(tool_result: dict) -> list[RuntimeSkillSource]:
    name = tool_result.get("name", "")
    result = tool_result.get("result", {})
    sources: dict[str, RuntimeSkillSource] = {}

    def add(doc_name: str) -> None:
        if not doc_name or doc_name in sources:
            return
        project = get_knowledge_project(get_runtime_config())
        full_path = safe_join(project.disk_path, f"raw/sources/{doc_name}")
        if full_path is None or not full_path.is_file():
            return
        sources[doc_name] = RuntimeSkillSource(
            name=doc_name,
            source_ref=sign_runtime_source_ref(doc_name),
        )

    if name == "read_raw" and isinstance(result, dict):
        add(str(result.get("path") or ""))
    if name == "grep_raw" and isinstance(result, dict):
        for match in result.get("matches") or []:
            if isinstance(match, dict):
                add(str(match.get("file") or ""))
    return list(sources.values())


def _build_runtime_skill_markdown(base_url: str) -> str:
    settings = get_runtime_config()
    chat_url = f"{base_url}/api/skill/chat"
    source_url = f"{base_url}/api/skill/documents/content?ref=<source_ref>"
    auth_line = (
        f"\nAuthorization: Bearer {settings.server.api_key}\n"
        if settings.server.api_key
        else "\nNo Authorization header is required for this Runtime.\n"
    )
    return f"""---
name: runtime-knowledge
description: Use when the user asks to query or verify information against the {settings.knowledge.name} Runtime knowledge base.
---

# {settings.knowledge.name} Runtime Knowledge

When the user asks to use this knowledge base, call the Runtime Skill chat endpoint before answering.

Do not answer from memory for in-scope questions.

Use:

- POST {chat_url}
- GET {source_url}
{auth_line}
Only read source documents through `source_ref` values returned by chat. Do not guess filenames or paths. Do not list raw documents.
If the knowledge base does not contain an answer, say so clearly.
"""


@router.get("/status")
async def status_response():
    return await build_status(get_runtime_config())


@router.get("/config")
async def get_config_response():
    return _redact_config(get_runtime_config().model_dump())


@router.put("/config")
async def update_config_response(body: dict[str, Any]):
    path = _runtime_config_path_or_500()
    body = _preserve_redacted_keys(body)
    path.write_text(_dump_runtime_yaml(body), encoding="utf-8")
    return _redact_config(load_runtime_config(path).model_dump())


@router.get("/config/yaml", response_model=RuntimeYamlConfigResponse)
async def get_yaml_config_response():
    return _build_yaml_config_response(_runtime_config_path_or_500())


@router.put("/config/yaml", response_model=RuntimeYamlConfigResponse)
async def update_yaml_config_response(content: str = Body(media_type="text/plain")):
    path = _runtime_config_path_or_500()
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Runtime config YAML must be a mapping")
    data = _preserve_redacted_keys(data)
    path.write_text(_dump_runtime_yaml(data), encoding="utf-8")
    load_runtime_config(path)
    return _build_yaml_config_response(path)


@router.get("/config/system-prompt", response_model=RuntimeSystemPromptConfigResponse)
async def get_system_prompt_config_response():
    return _build_system_prompt_config_response(_runtime_config_path_or_500())


@router.get("/skill")
async def runtime_skill_download(request: Request):
    return PlainTextResponse(
        _build_runtime_skill_markdown(external_base_url(request)),
        media_type="text/markdown",
    )


@router.post("/skill/chat", response_model=RuntimeSkillChatResponse)
async def runtime_skill_chat(body: RuntimeSkillChatRequest):
    runtime_body = RuntimeChatRequest(message=body.message, stream=False)
    answer_parts: list[str] = []
    sources_by_name: dict[str, RuntimeSkillSource] = {}

    async for event in _chat_events(runtime_body):
        if event["event"] == "token":
            answer_parts.append(event["data"].get("text", ""))
        elif event["event"] == "tool_result":
            for source in _runtime_source_from_tool_result(event["data"]):
                sources_by_name.setdefault(source.name, source)

    return RuntimeSkillChatResponse(
        answer="".join(answer_parts),
        sources=list(sources_by_name.values()),
    )


@router.get("/skill/documents/content")
async def runtime_skill_document_content(ref: str):
    payload = verify_source_ref(ref)
    if payload is None or payload.agent_id != "runtime" or payload.project_id != "knowledge":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found")

    project = get_knowledge_project(get_runtime_config())
    full_path = safe_join(project.disk_path, f"raw/sources/{payload.doc_name}")
    if full_path is None or not full_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source not found")
    return PlainTextResponse(await _read_text(full_path))


@router.put("/config/system-prompt", response_model=RuntimeSystemPromptConfigResponse)
async def update_system_prompt_config_response(body: RuntimeSystemPromptConfigUpdate):
    path = _runtime_config_path_or_500()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Runtime config YAML must be a mapping")
    knowledge = data.setdefault("knowledge", {})
    if not isinstance(knowledge, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Runtime config knowledge section must be a mapping")
    knowledge["system_prompt"] = body.system_prompt
    knowledge["system_prompt_override"] = body.system_prompt_override
    path.write_text(_dump_runtime_yaml(data), encoding="utf-8")
    load_runtime_config(path)
    return _build_system_prompt_config_response(path)


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


@router.post("/search/reindex")
async def rebuild_vector_index_response():
    project = get_knowledge_project()
    from app.embedding.service import rebuild_project_embeddings

    return await rebuild_project_embeddings(project.disk_path)


async def _rebuild_indexes(body: RuntimeReindexRequest, task_id: str | None = None) -> RuntimeReindexResponse:
    response = RuntimeReindexResponse()

    if body.target in ("knowledge", "all"):
        if task_id:
            set_index_task(task_id, status="running", progress=10, stage="重建知识库向量索引")
        project = get_knowledge_project()
        from app.embedding.service import rebuild_project_embeddings

        response.knowledge = await rebuild_project_embeddings(project.disk_path)
        if task_id:
            set_index_task(task_id, progress=55 if body.target == "all" else 90, stage="知识库向量索引已重建")

    if body.target in ("cases", "all"):
        if task_id:
            set_index_task(task_id, status="running", progress=60 if body.target == "all" else 10, stage="重建案例库索引")
        case_project = get_case_project()
        if case_project is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Case library is disabled")
        from app.case_index.builder import rebuild_case_index

        manifest = await rebuild_case_index(case_project.disk_path)
        response.cases = manifest.to_dict()
        if task_id:
            set_index_task(task_id, progress=90, stage="案例库索引已重建")

    return response


async def _run_rebuild_indexes_task(task_id: str, body: RuntimeReindexRequest) -> None:
    try:
        result = await _rebuild_indexes(body, task_id)
        set_index_task(
            task_id,
            status="succeeded",
            progress=100,
            stage="索引重建完成",
            result=result.model_dump(),
        )
    except HTTPException as exc:
        set_index_task(task_id, status="failed", progress=100, stage="索引重建失败", error=str(exc.detail))
    except Exception as exc:
        set_index_task(task_id, status="failed", progress=100, stage="索引重建失败", error=str(exc))


@router.post("/indexes/rebuild", response_model=IndexRebuildTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_rebuild_indexes_response(body: RuntimeReindexRequest, background_tasks: BackgroundTasks):
    target = f"runtime:{body.target}"
    if has_active_index_task(target=target):
        raise HTTPException(status.HTTP_409_CONFLICT, "Rebuild already in progress")
    task = create_index_task(target)
    background_tasks.add_task(_run_rebuild_indexes_task, task.task_id, body)
    return task


@router.get("/indexes/rebuild/current", response_model=IndexRebuildTaskResponse | None)
async def get_current_rebuild_indexes_status():
    return (
        get_active_index_task(target="runtime:all")
        or get_active_index_task(target="runtime:knowledge")
        or get_active_index_task(target="runtime:cases")
    )


@router.get("/indexes/rebuild/{task_id}", response_model=IndexRebuildTaskResponse)
async def get_rebuild_indexes_status(task_id: str):
    task = get_index_task(task_id)
    if task is None or not task.target.startswith("runtime:"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rebuild task not found")
    return task


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
    settings = get_runtime_config()
    project = get_knowledge_project(settings)
    full_path = safe_join(project.disk_path, path)
    if full_path is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path")
    if not full_path.exists() or not full_path.is_file():
        resolved = await _resolve_missing_wiki_page(
            project.disk_path,
            path,
            settings.search.wiki_fallback_vector_distance_threshold,
        )
        if resolved is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
        full_path, resolution = resolved
        content = await _read_text(full_path)
        meta, _ = parse_frontmatter(content)
        return {
            "path": full_path.relative_to(Path(project.disk_path)).as_posix(),
            "content": content,
            "meta": meta.raw,
            "resolved_from": path,
            "resolution": resolution,
        }
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


def _disconnect_checker(request: Request) -> ShouldCancel:
    async def check() -> bool:
        return await request.is_disconnected()

    return check


@router.post("/chat")
async def chat_response(body: RuntimeChatRequest, request: Request):
    should_cancel = _disconnect_checker(request)
    if body.stream:
        return StreamingResponse(_chat_sse(body, should_cancel), media_type="text/event-stream")

    answer_parts: list[str] = []
    tool_traces: list[dict] = []
    references: list[dict] = []
    used_case_library = False
    async for event in _chat_events(body, should_cancel):
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


async def _resolve_missing_wiki_page(
    project_dir: str,
    requested_path: str,
    distance_threshold: float,
) -> tuple[Path, dict] | None:
    resolved = await resolve_missing_wiki_page(project_dir, requested_path, distance_threshold)
    if resolved is None:
        return None
    return resolved.path, resolved.info


async def _chat_sse(body: RuntimeChatRequest, should_cancel: ShouldCancel = None) -> AsyncGenerator[str, None]:
    try:
        async for event in _chat_events(body, should_cancel):
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
    except Exception as exc:
        yield "event: error\n"
        yield f"data: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"


async def _chat_events(body: RuntimeChatRequest, should_cancel: ShouldCancel = None) -> AsyncGenerator[dict, None]:
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
        system_prompt_override=settings.knowledge.system_prompt_override,
        max_tool_calls=settings.runtime.max_tool_calls,
        debug_result_limit=settings.runtime.debug_result_limit,
        should_cancel=should_cancel,
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
async def openai_chat_response(body: OpenAIChatRequest, request: Request):
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
    should_cancel = _disconnect_checker(request)

    if body.stream:
        return StreamingResponse(_openai_sse(runtime_body, body.model, should_cancel), media_type="text/event-stream")

    answer_parts: list[str] = []
    async for event in _chat_events(runtime_body, should_cancel):
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


async def _openai_sse(
    body: RuntimeChatRequest,
    model: str,
    should_cancel: ShouldCancel = None,
) -> AsyncGenerator[str, None]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    async for event in _chat_events(body, should_cancel):
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
