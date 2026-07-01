"""Admin-only endpoints for LLM / Embedding configuration."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, text

from app.auth.deps import require_admin
from app.auth.models import User, UserApiToken
from app.config import _reload_config, get_config, load_db_overrides, normalize_litellm_api_base, save_db_settings
from app.database import async_session
from app.agents.models import Agent, AgentProject
from app.projects.models import Project, ProjectMember

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# ── Schemas ──

class LLMSettingsResponse(BaseModel):
    provider: str
    model: str
    api_key: str
    api_base: str | None
    max_context_size: int
    timeout: int
    ingest_temperature: float
    chat_temperature: float
    stream: bool


class LLMSettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    max_context_size: int | None = None
    timeout: int | None = None
    ingest_temperature: float | None = None
    chat_temperature: float | None = None
    stream: bool | None = None


class EmbeddingSettingsResponse(BaseModel):
    enabled: bool
    provider: str
    model: str
    api_key: str
    api_base: str | None
    dimensions: int | None


class EmbeddingSettingsUpdate(BaseModel):
    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    dimensions: int | None = None


class TestResult(BaseModel):
    success: bool
    message: str
    detail: str | None = None


class ConfigImportSummary(BaseModel):
    users: int
    projects: int
    project_members: int
    agents: int
    agent_projects: int
    system_settings: int


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


async def _export_config_bundle(db) -> dict:
    users = list((await db.execute(select(User).order_by(User.id.asc()))).scalars().all())
    projects = list((await db.execute(select(Project).order_by(Project.created_at.asc()))).scalars().all())
    project_members = list((await db.execute(select(ProjectMember).order_by(ProjectMember.id.asc()))).scalars().all())
    agents = list((await db.execute(select(Agent).order_by(Agent.created_at.asc()))).scalars().all())
    agent_projects = list((await db.execute(select(AgentProject).order_by(AgentProject.id.asc()))).scalars().all())
    system_settings = (await db.execute(
        text("SELECT section, data FROM system_settings ORDER BY section ASC")
    )).fetchall()

    return {
        "version": 1,
        "exported_at": datetime.utcnow().isoformat(),
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "password_hash": u.password_hash,
                "role": u.role,
                "created_at": _dt(u.created_at),
            }
            for u in users
        ],
        "system_settings": [
            {"section": row[0], "data": json.loads(row[1])}
            for row in system_settings
        ],
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "slug": p.slug,
                "description": p.description,
                "created_by": p.created_by,
                "created_at": _dt(p.created_at),
                "ticket_project_id": p.ticket_project_id,
                "project_type": p.project_type,
                "feedback_enabled": p.feedback_enabled,
                "git_repo_url": p.git_repo_url,
                "git_branch": p.git_branch,
                "git_username": p.git_username,
                "git_auth_token": p.git_auth_token,
                "git_author_name": p.git_author_name,
                "git_author_email": p.git_author_email,
                "git_sync_enabled": p.git_sync_enabled,
                "git_sync_time": p.git_sync_time,
            }
            for p in projects
        ],
        "project_members": [
            {
                "project_id": m.project_id,
                "user_id": m.user_id,
                "role": m.role,
            }
            for m in project_members
        ],
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "system_prompt": a.system_prompt,
                "is_public": a.is_public,
                "require_api_key": a.require_api_key,
                "api_key_hash": a.api_key_hash,
                "max_tool_calls": a.max_tool_calls,
                "debug_result_limit": a.debug_result_limit,
                "tool_labels": a.tool_labels,
                "created_by": a.created_by,
                "created_at": _dt(a.created_at),
            }
            for a in agents
        ],
        "agent_projects": [
            {
                "agent_id": ap.agent_id,
                "project_id": ap.project_id,
            }
            for ap in agent_projects
        ],
    }


def _validate_config_bundle(bundle: dict) -> None:
    if bundle.get("version") != 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported config bundle version")
    required = {"users", "system_settings", "projects", "project_members", "agents", "agent_projects"}
    missing = required - set(bundle.keys())
    if missing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Missing sections: {', '.join(sorted(missing))}")


async def _restore_config_bundle(db, bundle: dict) -> dict[str, int]:
    _validate_config_bundle(bundle)

    await db.execute(delete(AgentProject))
    await db.execute(delete(ProjectMember))
    await db.execute(delete(Agent))
    await db.execute(delete(Project))
    await db.execute(delete(UserApiToken))
    await db.execute(delete(User))
    await db.execute(text("DELETE FROM system_settings"))

    for row in bundle["users"]:
        db.add(User(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=row["role"],
            created_at=_parse_dt(row.get("created_at")),
        ))
    await db.flush()

    for row in bundle["system_settings"]:
        await db.execute(
            text("INSERT INTO system_settings(section, data) VALUES (:section, :data)"),
            {"section": row["section"], "data": json.dumps(row["data"], ensure_ascii=False)},
        )

    deferred_ticket_links: list[tuple[str, str | None]] = []
    for row in bundle["projects"]:
        deferred_ticket_links.append((row["id"], row.get("ticket_project_id")))
        db.add(Project(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            description=row.get("description", ""),
            _disk_path="",
            created_by=row["created_by"],
            created_at=_parse_dt(row.get("created_at")),
            ticket_project_id=None,
            project_type=row.get("project_type", "knowledge_base"),
            feedback_enabled=row.get("feedback_enabled", True),
            git_repo_url=row.get("git_repo_url", ""),
            git_branch=row.get("git_branch", "main"),
            git_username=row.get("git_username", ""),
            git_auth_token=row.get("git_auth_token", ""),
            git_author_name=row.get("git_author_name", ""),
            git_author_email=row.get("git_author_email", ""),
            git_sync_enabled=row.get("git_sync_enabled", False),
            git_sync_time=row.get("git_sync_time", "02:00"),
        ))
    await db.flush()

    if deferred_ticket_links:
        project_map = {
            proj.id: proj
            for proj in list((await db.execute(select(Project))).scalars().all())
        }
        for project_id, ticket_project_id in deferred_ticket_links:
            project_map[project_id].ticket_project_id = ticket_project_id
        await db.flush()

    for row in bundle["project_members"]:
        db.add(ProjectMember(
            project_id=row["project_id"],
            user_id=row["user_id"],
            role=row["role"],
        ))

    for row in bundle["agents"]:
        db.add(Agent(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            system_prompt=row.get("system_prompt", ""),
            is_public=row.get("is_public", False),
            require_api_key=row.get("require_api_key", True),
            api_key_hash=row.get("api_key_hash"),
            max_tool_calls=row.get("max_tool_calls", 20),
            debug_result_limit=row.get("debug_result_limit", 2000),
            tool_labels=row.get("tool_labels", "{}"),
            created_by=row["created_by"],
            created_at=_parse_dt(row.get("created_at")),
        ))
    await db.flush()

    project_types = {
        row["id"]: row.get("project_type", "knowledge_base")
        for row in bundle["projects"]
    }
    for row in bundle["agent_projects"]:
        if project_types.get(row["project_id"]) == "case_library":
            continue
        db.add(AgentProject(
            agent_id=row["agent_id"],
            project_id=row["project_id"],
        ))

    await db.commit()
    _reload_config()
    load_db_overrides()

    return {
        "users": len(bundle["users"]),
        "projects": len(bundle["projects"]),
        "project_members": len(bundle["project_members"]),
        "agents": len(bundle["agents"]),
        "agent_projects": len(bundle["agent_projects"]),
        "system_settings": len(bundle["system_settings"]),
    }


# ── LLM endpoints ──

@router.get("/settings/llm", response_model=LLMSettingsResponse)
async def get_llm_settings(user: User = Depends(require_admin)):
    cfg = get_config().llm
    return LLMSettingsResponse(
        provider=cfg.provider,
        model=cfg.model,
        api_key=_mask_key(cfg.api_key),
        api_base=cfg.api_base,
        max_context_size=cfg.max_context_size,
        timeout=cfg.timeout,
        ingest_temperature=cfg.ingest_temperature,
        chat_temperature=cfg.chat_temperature,
        stream=cfg.stream,
    )


@router.post("/settings/llm/test", response_model=TestResult)
async def test_llm(body: LLMSettingsUpdate, user: User = Depends(require_admin)):
    """Test LLM connection with provided settings (does not save)."""
    import litellm
    cfg = get_config().llm

    provider = body.provider or cfg.provider
    model = body.model or cfg.model
    api_key = _resolve_key(body.api_key, cfg.api_key)
    api_base = body.api_base if body.api_base is not None else cfg.api_base
    timeout = body.timeout if body.timeout is not None else cfg.timeout

    if "/" in model:
        model_name = model
    elif api_base:
        model_name = f"openai/{model}"
    elif provider == "openai":
        model_name = model
    else:
        model_name = f"{provider}/{model}"

    kwargs: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi, reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0,
        "api_key": api_key or None,
        "timeout": timeout,
    }
    if api_base:
        kwargs["api_base"] = normalize_litellm_api_base(api_base)

    try:
        resp = await litellm.acompletion(**kwargs)
        content = resp.choices[0].message.content or ""
        return TestResult(success=True, message="LLM 连接成功", detail=f"响应: {content[:100]}")
    except Exception as exc:
        return TestResult(success=False, message="LLM 连接失败", detail=str(exc)[:500])


@router.put("/settings/llm", response_model=LLMSettingsResponse)
async def update_llm_settings(body: LLMSettingsUpdate, user: User = Depends(require_admin)):
    cfg = get_config().llm
    values = body.model_dump(exclude_none=True)
    if "api_key" in values and _is_masked(values["api_key"]):
        del values["api_key"]

    await save_db_settings("llm", values)
    cfg = get_config().llm
    return LLMSettingsResponse(
        provider=cfg.provider,
        model=cfg.model,
        api_key=_mask_key(cfg.api_key),
        api_base=cfg.api_base,
        max_context_size=cfg.max_context_size,
        timeout=cfg.timeout,
        ingest_temperature=cfg.ingest_temperature,
        chat_temperature=cfg.chat_temperature,
        stream=cfg.stream,
    )


# ── Embedding endpoints ──

@router.get("/settings/embedding", response_model=EmbeddingSettingsResponse)
async def get_embedding_settings(user: User = Depends(require_admin)):
    cfg = get_config().embedding
    return EmbeddingSettingsResponse(
        enabled=cfg.enabled,
        provider=cfg.provider,
        model=cfg.model,
        api_key=_mask_key(cfg.api_key),
        api_base=cfg.api_base,
        dimensions=cfg.dimensions,
    )


@router.post("/settings/embedding/test", response_model=TestResult)
async def test_embedding(body: EmbeddingSettingsUpdate, user: User = Depends(require_admin)):
    """Test embedding connection with provided settings (does not save)."""
    import litellm
    cfg = get_config().embedding

    provider = body.provider or cfg.provider
    model = body.model or cfg.model
    api_key = _resolve_key(body.api_key, cfg.api_key)
    api_base = body.api_base if body.api_base is not None else cfg.api_base
    dimensions = body.dimensions if body.dimensions is not None else cfg.dimensions

    if "/" in model:
        model_name = model
    elif api_base:
        model_name = f"openai/{model}"
    elif provider == "openai":
        model_name = model
    else:
        model_name = f"{provider}/{model}"

    kwargs: dict = {
        "model": model_name,
        "input": ["测试文本"],
        "api_key": api_key or None,
    }
    if dimensions:
        kwargs["dimensions"] = dimensions
    if api_base:
        kwargs["api_base"] = normalize_litellm_api_base(api_base)

    try:
        resp = await litellm.aembedding(**kwargs)
        dim = len(resp.data[0]["embedding"])
        return TestResult(success=True, message="Embedding 连接成功", detail=f"向量维度: {dim}")
    except Exception as exc:
        return TestResult(success=False, message="Embedding 连接失败", detail=str(exc)[:500])


@router.put("/settings/embedding", response_model=EmbeddingSettingsResponse)
async def update_embedding_settings(body: EmbeddingSettingsUpdate, user: User = Depends(require_admin)):
    cfg = get_config().embedding
    values = body.model_dump(exclude_none=True)
    if "api_key" in values and _is_masked(values["api_key"]):
        del values["api_key"]

    await save_db_settings("embedding", values)
    cfg = get_config().embedding
    return EmbeddingSettingsResponse(
        enabled=cfg.enabled,
        provider=cfg.provider,
        model=cfg.model,
        api_key=_mask_key(cfg.api_key),
        api_base=cfg.api_base,
        dimensions=cfg.dimensions,
    )


# ── Helpers ──

def _is_masked(key: str) -> bool:
    return "***" in key


def _mask_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "***" if key else ""
    return key[:4] + "***" + key[-4:]


def _resolve_key(new_key: str | None, current_key: str) -> str:
    """If new_key is masked or None, use the current key."""
    if new_key is None or _is_masked(new_key):
        return current_key
    return new_key


# ── Feedback model settings ──

class FeedbackModelSettingsResponse(BaseModel):
    evaluator_model: str | None
    evaluator_provider: str | None
    evaluator_api_key: str
    evaluator_api_base: str | None
    evaluator_temperature: float
    evaluator_max_tokens: int
    compiler_model: str | None
    compiler_provider: str | None
    compiler_api_key: str
    compiler_api_base: str | None
    compiler_temperature: float
    compiler_max_tokens: int
    enabled: bool
    max_revisions: int
    dedup_window_minutes: int


class FeedbackModelSettingsUpdate(BaseModel):
    evaluator_model: str | None = None
    evaluator_provider: str | None = None
    evaluator_api_key: str | None = None
    evaluator_api_base: str | None = None
    evaluator_temperature: float | None = None
    evaluator_max_tokens: int | None = None
    compiler_model: str | None = None
    compiler_provider: str | None = None
    compiler_api_key: str | None = None
    compiler_api_base: str | None = None
    compiler_temperature: float | None = None
    compiler_max_tokens: int | None = None
    enabled: bool | None = None
    max_revisions: int | None = None
    dedup_window_minutes: int | None = None


def _feedback_response() -> FeedbackModelSettingsResponse:
    cfg = get_config().feedback
    return FeedbackModelSettingsResponse(
        evaluator_model=cfg.evaluator.model,
        evaluator_provider=cfg.evaluator.provider,
        evaluator_api_key=_mask_key(cfg.evaluator.api_key or ""),
        evaluator_api_base=cfg.evaluator.api_base,
        evaluator_temperature=cfg.evaluator.temperature,
        evaluator_max_tokens=cfg.evaluator.max_tokens,
        compiler_model=cfg.compiler.model,
        compiler_provider=cfg.compiler.provider,
        compiler_api_key=_mask_key(cfg.compiler.api_key or ""),
        compiler_api_base=cfg.compiler.api_base,
        compiler_temperature=cfg.compiler.temperature,
        compiler_max_tokens=cfg.compiler.max_tokens,
        enabled=cfg.enabled,
        max_revisions=cfg.max_revisions,
        dedup_window_minutes=cfg.dedup_window_minutes,
    )


@router.get("/settings/feedback", response_model=FeedbackModelSettingsResponse)
async def get_feedback_settings(user: User = Depends(require_admin)):
    return _feedback_response()


@router.put("/settings/feedback", response_model=FeedbackModelSettingsResponse)
async def update_feedback_settings(body: FeedbackModelSettingsUpdate, user: User = Depends(require_admin)):
    cfg = get_config().feedback
    nested: dict = {"evaluator": {}, "compiler": {}}
    top: dict = {}

    field_map = {
        "evaluator_model": ("evaluator", "model"),
        "evaluator_provider": ("evaluator", "provider"),
        "evaluator_api_key": ("evaluator", "api_key"),
        "evaluator_api_base": ("evaluator", "api_base"),
        "evaluator_temperature": ("evaluator", "temperature"),
        "evaluator_max_tokens": ("evaluator", "max_tokens"),
        "compiler_model": ("compiler", "model"),
        "compiler_provider": ("compiler", "provider"),
        "compiler_api_key": ("compiler", "api_key"),
        "compiler_api_base": ("compiler", "api_base"),
        "compiler_temperature": ("compiler", "temperature"),
        "compiler_max_tokens": ("compiler", "max_tokens"),
    }

    raw = body.model_dump(exclude_none=True)
    for flat_key, (group, attr) in field_map.items():
        if flat_key in raw:
            val = raw[flat_key]
            if "api_key" in flat_key and _is_masked(str(val)):
                continue
            nested[group][attr] = val

    for k in ("enabled", "max_revisions", "dedup_window_minutes"):
        if k in raw:
            top[k] = raw[k]

    values = {**top}
    if nested["evaluator"]:
        values["evaluator"] = nested["evaluator"]
    if nested["compiler"]:
        values["compiler"] = nested["compiler"]

    if values:
        await save_db_settings("feedback", values)

    return _feedback_response()


@router.post("/settings/feedback/test", response_model=TestResult)
async def test_feedback_model(body: FeedbackModelSettingsUpdate, user: User = Depends(require_admin)):
    """Test feedback evaluator/compiler LLM connection."""
    import litellm
    main_cfg = get_config().llm
    fb_cfg = get_config().feedback

    eval_model = body.evaluator_model or fb_cfg.evaluator.model or main_cfg.model
    eval_provider = body.evaluator_provider or fb_cfg.evaluator.provider or main_cfg.provider
    eval_key = _resolve_key(body.evaluator_api_key, fb_cfg.evaluator.api_key or main_cfg.api_key)
    eval_base = body.evaluator_api_base if body.evaluator_api_base is not None else (fb_cfg.evaluator.api_base or main_cfg.api_base)

    if "/" in eval_model:
        model_name = eval_model
    elif eval_base:
        model_name = f"openai/{eval_model}"
    elif eval_provider == "openai":
        model_name = eval_model
    else:
        model_name = f"{eval_provider}/{eval_model}"

    kwargs: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi, reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0,
        "api_key": eval_key or None,
    }
    if eval_base:
        kwargs["api_base"] = normalize_litellm_api_base(eval_base)

    try:
        resp = await litellm.acompletion(**kwargs)
        content = resp.choices[0].message.content or ""
        return TestResult(success=True, message="Feedback 模型连接成功", detail=f"响应: {content[:100]}")
    except Exception as exc:
        return TestResult(success=False, message="Feedback 模型连接失败", detail=str(exc)[:500])


# ── Config backup ──

@router.get("/config/export")
async def export_config(user: User = Depends(require_admin)):
    async with async_session() as db:
        bundle = await _export_config_bundle(db)
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": 'attachment; filename="llm-wiki-config.json"'},
    )


@router.post("/config/import", response_model=ConfigImportSummary)
async def import_config(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
):
    try:
        payload = json.loads((await file.read()).decode("utf-8"))
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid JSON bundle")

    async with async_session() as db:
        summary = await _restore_config_bundle(db, payload)
    return ConfigImportSummary(**summary)


# ── User management ──

class UserListResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: str

    class Config:
        from_attributes = True


class UpdateRoleRequest(BaseModel):
    role: str


@router.get("/users", response_model=list[UserListResponse])
async def list_users(user: User = Depends(require_admin)):
    from sqlalchemy import select
    from app.database import async_session
    async with async_session() as db:
        stmt = select(User).order_by(User.created_at.desc())
        users = list((await db.execute(stmt)).scalars().all())
        return [
            UserListResponse(
                id=u.id,
                username=u.username,
                role=u.role,
                created_at=u.created_at.isoformat() if u.created_at else "",
            )
            for u in users
        ]


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    body: UpdateRoleRequest,
    user: User = Depends(require_admin),
):
    if body.role not in ("admin", "user"):
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Role must be 'admin' or 'user'")
    if user_id == user.id:
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot change your own role")

    from sqlalchemy import select
    from app.database import async_session
    async with async_session() as db:
        target = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not target:
            from fastapi import HTTPException, status
            raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
        target.role = body.role
        await db.commit()
        return {"id": target.id, "username": target.username, "role": target.role}


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    user: User = Depends(require_admin),
):
    if user_id == user.id:
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot delete yourself")

    from sqlalchemy import select
    from app.database import async_session
    async with async_session() as db:
        target = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not target:
            from fastapi import HTTPException, status
            raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
        await db.delete(target)
        await db.commit()
