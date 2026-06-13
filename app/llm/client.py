"""Unified LLM client via LiteLLM."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

from app.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar("T")
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
        kwargs["api_base"] = cfg.api_base
    return kwargs


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


async def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> str:
    litellm = _litellm()
    cfg = get_config().llm
    resp = await _with_llm_retries(
        "completion",
        lambda: litellm.acompletion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **_common_kwargs(temperature if temperature is not None else cfg.ingest_temperature, max_tokens),
        ),
    )
    return resp.choices[0].message.content or ""


async def stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    litellm = _litellm()
    cfg = get_config().llm
    resp = await _with_llm_retries(
        "stream",
        lambda: litellm.acompletion(
            messages=messages,
            stream=True,
            **_common_kwargs(temperature if temperature is not None else cfg.chat_temperature, max_tokens),
        ),
    )
    async for chunk in resp:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


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
) -> LLMResponse:
    """Single LLM call with tool definitions. Returns structured response."""
    litellm = _litellm()
    cfg = get_config().llm
    kwargs = _common_kwargs(
        temperature if temperature is not None else cfg.chat_temperature,
        max_tokens,
    )
    kwargs["tools"] = tools

    resp = await _with_llm_retries(
        "tool_completion",
        lambda: litellm.acompletion(messages=messages, **kwargs),
    )
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

    return LLMResponse(
        content=msg.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        usage=usage,
    )
