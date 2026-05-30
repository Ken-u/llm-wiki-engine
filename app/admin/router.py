"""Admin-only endpoints for LLM / Embedding configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.deps import require_admin
from app.auth.models import User
from app.config import get_config, save_db_settings

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# ── Schemas ──

class LLMSettingsResponse(BaseModel):
    provider: str
    model: str
    api_key: str
    api_base: str | None
    max_context_size: int
    ingest_temperature: float
    chat_temperature: float
    stream: bool


class LLMSettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    max_context_size: int | None = None
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
    }
    if api_base:
        kwargs["api_base"] = api_base

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
        kwargs["api_base"] = api_base

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
        kwargs["api_base"] = eval_base

    try:
        resp = await litellm.acompletion(**kwargs)
        content = resp.choices[0].message.content or ""
        return TestResult(success=True, message="Feedback 模型连接成功", detail=f"响应: {content[:100]}")
    except Exception as exc:
        return TestResult(success=False, message="Feedback 模型连接失败", detail=str(exc)[:500])


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
