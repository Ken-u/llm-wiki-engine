"""Unified LLM client via LiteLLM."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

from app.config import get_config, normalize_litellm_api_base

logger = logging.getLogger(__name__)

T = TypeVar("T")
ShouldCancel = Callable[[], Awaitable[bool]] | None
MAX_LLM_ATTEMPTS = 4
LLM_RETRY_DELAYS = (1.0, 2.0, 4.0)
_TRANSIENT_ERROR_MARKERS = (
    "internalservererror",
    "upstream error",
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "apiconnectionerror",
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)

_debug_log_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar("debug_llm_log_path", default=None)
_debug_log_counter: contextvars.ContextVar[int] = contextvars.ContextVar("debug_llm_log_counter", default=0)
_debug_log_group_active: contextvars.ContextVar[bool] = contextvars.ContextVar("debug_llm_log_group_active", default=False)


def _litellm():
    import litellm

    litellm.drop_params = True
    return litellm


def _model_name() -> str:
    cfg = get_config().llm
    provider = cfg.provider
    model = cfg.model
    if "/" in model:
        return model
    if cfg.api_base:
        return f"openai/{model}"
    if provider == "openai":
        return model
    return f"{provider}/{model}"


def _common_kwargs(temperature: float, max_tokens: int) -> dict:
    cfg = get_config().llm
    kwargs: dict = {
        "model": _model_name(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": cfg.api_key or None,
        "timeout": cfg.timeout,
    }
    if cfg.api_base:
        kwargs["api_base"] = normalize_litellm_api_base(cfg.api_base)
    return kwargs


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return repr(value)


def _logged_kwargs(kwargs: dict) -> dict:
    data = {k: v for k, v in kwargs.items() if k != "api_key"}
    return _jsonable(data)


def _debug_log_dir() -> Path | None:
    dir_value = getattr(get_config().llm, "debug_llm_log", "")
    if not dir_value:
        return None
    return Path(dir_value).expanduser()


def _next_debug_log_path() -> Path | None:
    log_dir = _debug_log_dir()
    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    grouped = _debug_log_group_active.get()
    if grouped:
        existing = _debug_log_path.get()
        if existing is not None:
            return existing
    now = datetime.now(timezone.utc)
    count = _debug_log_counter.get() + 1
    _debug_log_counter.set(count)
    path = log_dir / f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{count:04d}.json"
    if grouped:
        _debug_log_path.set(path)
    return path


def _write_debug_llm_log(entry: dict) -> None:
    path = _next_debug_log_path()
    if path is None:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "created_at": payload["timestamp"],
        }
    data["updated_at"] = payload["timestamp"]

    if payload.get("kind") == "tool_execution":
        data.setdefault("tool_executions", []).append({
            "timestamp": payload["timestamp"],
            **payload.get("data", {}),
        })
    elif payload.get("request") is not None:
        request = dict(payload.get("request") or {})
        tools = request.pop("tools", None)
        if tools is not None:
            data["tools"] = tools
        llm_entry = {
            "timestamp": payload["timestamp"],
            "kind": payload.get("kind"),
            "status": payload.get("status"),
            "request": request,
        }
        if tools is not None:
            llm_entry["tools_ref"] = "tools"
        if payload.get("response") is not None:
            llm_entry["response"] = payload["response"]
        if payload.get("error") is not None:
            llm_entry["error"] = payload["error"]
        data["llm"] = llm_entry
        data["llm_call_count"] = int(data.get("llm_call_count", 0)) + 1
    else:
        data.setdefault("records", []).append(payload)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_debug_llm_log_context() -> None:
    _debug_log_path.set(None)
    _debug_log_group_active.set(True)


def finish_debug_llm_log_context() -> None:
    _debug_log_path.set(None)
    _debug_log_group_active.set(False)


def _log_llm_success(kind: str, request: dict, response: dict) -> None:
    try:
        _write_debug_llm_log({
            "kind": kind,
            "status": "succeeded",
            "request": _jsonable(request),
            "response": _jsonable(response),
        })
    except Exception:
        logger.exception("Failed to write debug LLM log")


def _log_llm_error(kind: str, request: dict, exc: Exception) -> None:
    try:
        _write_debug_llm_log({
            "kind": kind,
            "status": "failed",
            "request": _jsonable(request),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        })
    except Exception:
        logger.exception("Failed to write debug LLM log")


def log_debug_llm_event(kind: str, data: dict) -> None:
    try:
        _write_debug_llm_log({
            "kind": kind,
            "status": "recorded",
            "data": _jsonable(data),
        })
    except Exception:
        logger.exception("Failed to write debug LLM log")


def _is_transient_llm_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = f"{name}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


async def _with_llm_retries(label: str, fn: Callable[[], Awaitable[T]]) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_LLM_ATTEMPTS or not _is_transient_llm_error(exc):
                raise
            delay = LLM_RETRY_DELAYS[min(attempt - 1, len(LLM_RETRY_DELAYS) - 1)]
            logger.warning(
                "Transient LLM error during %s (attempt %d/%d), retrying in %.1fs: %s",
                label,
                attempt,
                MAX_LLM_ATTEMPTS,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _wait_with_cancel(task: asyncio.Task[T], should_cancel: ShouldCancel) -> T:
    if should_cancel is None:
        return await task
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=0.2)
        if task in done:
            break
        if await should_cancel():
            task.cancel()
            raise asyncio.CancelledError("Client disconnected")
    return await task


async def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> str:
    litellm = _litellm()
    cfg = get_config().llm
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = _common_kwargs(temperature if temperature is not None else cfg.ingest_temperature, max_tokens)
    request = {"messages": messages, **_logged_kwargs(kwargs)}
    try:
        resp = await _with_llm_retries(
            "completion",
            lambda: litellm.acompletion(messages=messages, **kwargs),
        )
        content = resp.choices[0].message.content or ""
        _log_llm_success("completion", request, {"content": content, "raw": _jsonable(resp)})
        return content
    except Exception as exc:
        _log_llm_error("completion", request, exc)
        raise


async def stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
    should_cancel: ShouldCancel = None,
) -> AsyncGenerator[str, None]:
    litellm = _litellm()
    cfg = get_config().llm
    kwargs = _common_kwargs(temperature if temperature is not None else cfg.chat_temperature, max_tokens)
    request = {"messages": messages, "stream": True, **_logged_kwargs(kwargs)}
    parts: list[str] = []
    try:
        stream_task = asyncio.create_task(_with_llm_retries(
            "stream",
            lambda: litellm.acompletion(
                messages=messages,
                stream=True,
                **kwargs,
            ),
        ))
        resp = await _wait_with_cancel(stream_task, should_cancel)
        async for chunk in resp:
            if should_cancel is not None and await should_cancel():
                close = getattr(resp, "aclose", None)
                if callable(close):
                    await close()
                raise asyncio.CancelledError("Client disconnected")
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
                yield delta.content
        _log_llm_success("stream", request, {"content": "".join(parts)})
    except Exception as exc:
        _log_llm_error("stream", request, exc)
        raise


async def stream_collect(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> str:
    """Call LLM and return the full text.

    Uses streaming or non-streaming mode based on config.llm.ingest_stream.
    """
    cfg = get_config().llm
    if not cfg.stream:
        return await complete(system, user, temperature=temperature, max_tokens=max_tokens)

    async def collect_stream() -> str:
        parts: list[str] = []
        async for token in stream(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            parts.append(token)
        return "".join(parts)

    return await _with_llm_retries("stream_collect", collect_stream)


@dataclass
class ToolCallRequest:
    """A single tool call parsed from the LLM response."""
    id: str
    name: str
    arguments: str


@dataclass
class TokenUsage:
    """Token usage reported by the LLM provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Structured LLM response that may contain text, tool calls, or both."""
    content: str | None
    tool_calls: list[ToolCallRequest]
    finish_reason: str | None
    usage: TokenUsage | None = None


async def complete_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
    should_cancel: ShouldCancel = None,
) -> LLMResponse:
    """Single LLM call with tool definitions. Returns structured response."""
    litellm = _litellm()
    cfg = get_config().llm
    kwargs = _common_kwargs(
        temperature if temperature is not None else cfg.chat_temperature,
        max_tokens,
    )
    kwargs["tools"] = tools
    request = {"messages": messages, **_logged_kwargs(kwargs)}

    try:
        completion_task = asyncio.create_task(_with_llm_retries(
            "tool_completion",
            lambda: litellm.acompletion(messages=messages, **kwargs),
        ))
        resp = await _wait_with_cancel(completion_task, should_cancel)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        usage = None
        if getattr(resp, "usage", None):
            u = resp.usage
            usage = TokenUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        result = LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
        _log_llm_success("tool_completion", request, {
            "content": result.content,
            "tool_calls": [tc.__dict__ for tc in result.tool_calls],
            "finish_reason": result.finish_reason,
            "usage": result.usage.__dict__ if result.usage else None,
            "raw": _jsonable(resp),
        })
        return result
    except Exception as exc:
        _log_llm_error("tool_completion", request, exc)
        raise
