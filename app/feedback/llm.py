"""Feedback pipeline LLM client — independent from app/llm/client.py.

Uses litellm directly with its own model configuration resolved from
FeedbackModelConfig. Supports tool calling for structured output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import litellm

from app.config import get_config, FeedbackModelConfig

logger = logging.getLogger(__name__)

litellm.drop_params = True


@dataclass
class ToolCallResult:
    """A single tool call extracted from the LLM response."""
    id: str
    name: str
    arguments: dict


@dataclass
class FeedbackLLMResponse:
    """Structured response from a feedback LLM call."""
    content: str | None
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    finish_reason: str | None = None


def _resolve_model(cfg: FeedbackModelConfig) -> str:
    """Build litellm model string. Falls back to main llm config for None fields."""
    main = get_config().llm
    model = cfg.model or main.model
    provider = cfg.provider or main.provider
    api_base = cfg.api_base or main.api_base

    if "/" in model:
        return model
    if api_base:
        return f"openai/{model}"
    if provider == "openai":
        return model
    return f"{provider}/{model}"


def _build_kwargs(cfg: FeedbackModelConfig) -> dict:
    """Build litellm call kwargs from feedback model config, falling back to main llm."""
    main = get_config().llm
    kwargs: dict = {
        "model": _resolve_model(cfg),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": cfg.api_key or main.api_key or None,
        "timeout": cfg.timeout,
    }
    api_base = cfg.api_base or main.api_base
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


class SchemaValidationError(RuntimeError):
    """Raised when a tool schema is corrupted before sending to the provider."""


def _validate_tools_schema(tools: list[dict]) -> None:
    """Ensure every tool has a non-empty parameters schema.

    Catches the bug where LiteLLM/provider adapters strip out the
    parameters object, leaving ``{}``.
    """
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        props = params.get("properties")
        if not props:
            raise SchemaValidationError(
                f"Tool '{fn.get('name', '?')}' has empty parameters schema. "
                f"Full tool dict: {json.dumps(tool, ensure_ascii=False)}"
            )


async def complete_with_tools(
    messages: list[dict],
    tools: list[dict],
    cfg: FeedbackModelConfig,
    *,
    tool_choice: dict | str | None = None,
) -> FeedbackLLMResponse:
    """Call LLM with tool definitions using feedback-specific model config.

    Returns structured response. Tool call arguments are automatically
    parsed from JSON strings into dicts.

    ``tool_choice`` can be "auto", "required", or a specific function
    selector like ``{"type": "function", "function": {"name": "xxx"}}``.
    """
    _validate_tools_schema(tools)

    kwargs = _build_kwargs(cfg)
    kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Feedback LLM request: model=%s, tool_choice=%s, tools_schema=%s",
            kwargs.get("model"),
            tool_choice,
            json.dumps(
                [{
                    "name": t["function"]["name"],
                    "parameters": t["function"].get("parameters", {}),
                } for t in tools],
                ensure_ascii=False,
            ),
        )

    resp = await litellm.acompletion(messages=messages, **kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[ToolCallResult] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCallResult(
                id=tc.id or "",
                name=tc.function.name,
                arguments=args,
            ))

    logger.info(
        "Feedback LLM response: finish_reason=%s, tool_calls=%d, has_content=%s",
        choice.finish_reason,
        len(tool_calls),
        bool(msg.content),
    )

    return FeedbackLLMResponse(
        content=msg.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
    )
