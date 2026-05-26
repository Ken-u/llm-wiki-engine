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
