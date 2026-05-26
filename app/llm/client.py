"""Unified LLM client via LiteLLM."""

from __future__ import annotations

from typing import AsyncGenerator

import litellm

from app.config import get_config

litellm.drop_params = True


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
    }
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    return kwargs


async def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> str:
    cfg = get_config().llm
    resp = await litellm.acompletion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **_common_kwargs(temperature if temperature is not None else cfg.ingest_temperature, max_tokens),
    )
    return resp.choices[0].message.content or ""


async def stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    cfg = get_config().llm
    resp = await litellm.acompletion(
        messages=messages,
        stream=True,
        **_common_kwargs(temperature if temperature is not None else cfg.chat_temperature, max_tokens),
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
