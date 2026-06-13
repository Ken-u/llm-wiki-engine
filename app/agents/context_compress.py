"""Agent context compression via LLM summarization.

Triggered when prompt_tokens reach context_compress_threshold (default 85%)
of max_context_size. Summarizes older history and tool results into system
prompt sections, keeping recent turns intact.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.chat.context import CHARS_PER_TOKEN, truncate_to_budget
from app.llm import client as llm_client

logger = logging.getLogger(__name__)

HISTORY_SUMMARY_HEADER = "## 此前对话摘要"
TOOL_SUMMARY_HEADER = "## 此前工具调用摘要"
KEEP_RECENT_TURNS = 2  # user/assistant pairs
KEEP_RECENT_TOOL_GROUPS = 1

_SUMMARY_SYSTEM = "\n".join([
    "你是一个对话摘要助手。将以下对话历史压缩为简洁的中文摘要。",
    "要求：",
    "- 保留关键事实、结论、案例 ID（如 [[558753]]）、未决问题",
    "- 不要编造原文中没有的信息",
    "- 使用要点列表，控制在 500 字以内",
])

_TOOL_SUMMARY_SYSTEM = "\n".join([
    "你是一个工具调用结果摘要助手。将以下工具调用及其返回结果压缩为简洁的中文摘要。",
    "要求：",
    "- 保留工具名称、关键参数、重要返回值",
    "- 不要编造原文中没有的信息",
    "- 使用要点列表，控制在 300 字以内",
])


@dataclass
class CompressResult:
    messages: list[dict]
    persisted_history: list[dict]
    compressed: bool
    summary_snippet: str | None = None


@dataclass
class _MessageLayout:
    system_idx: int
    history_end: int  # exclusive index in messages; history is [1:history_end)
    current_user_idx: int
    tool_tail_start: int  # inclusive; assistant+tool pairs from here


def should_compress(prompt_tokens: int, max_context_size: int, threshold: float) -> bool:
    return prompt_tokens >= int(max_context_size * threshold)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return max(1, total_chars // CHARS_PER_TOKEN)


def resolve_prompt_tokens(
    messages: list[dict],
    *,
    api_prompt_tokens: int | None,
) -> tuple[int, str]:
    if api_prompt_tokens and api_prompt_tokens > 0:
        return api_prompt_tokens, "api"
    return estimate_prompt_tokens(messages), "estimated"


def _parse_layout(messages: list[dict]) -> _MessageLayout | None:
    if not messages or messages[0].get("role") != "system":
        return None

    # Last user message before tool tail is the current turn user message
    last_user_idx = None
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None

    tool_tail_start = last_user_idx + 1
    if tool_tail_start < len(messages) and messages[tool_tail_start].get("role") != "assistant":
        tool_tail_start = len(messages)

    return _MessageLayout(
        system_idx=0,
        history_end=last_user_idx,
        current_user_idx=last_user_idx,
        tool_tail_start=tool_tail_start,
    )


def _format_turns_for_summary(turns: list[dict]) -> str:
    parts: list[str] = []
    for msg in turns:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role in ("user", "assistant") and content:
            label = "用户" if role == "user" else "助手"
            parts.append(f"{label}: {content}")
    return "\n\n".join(parts)


def _format_tool_tail_for_summary(messages: list[dict], start: int, end: int) -> str:
    parts: list[str] = []
    i = start
    while i < end:
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg["tool_calls"]
            ]
            parts.append(f"工具调用: {', '.join(names)}")
            j = i + 1
            while j < end and messages[j].get("role") == "tool":
                tool_msg = messages[j]
                content = tool_msg.get("content") or ""
                if len(content) > 2000:
                    content = content[:2000] + "..."
                parts.append(f"工具结果: {content}")
                j += 1
            i = j
        else:
            i += 1
    return "\n\n".join(parts)


def _inject_summary_block(system_content: str, header: str, summary: str) -> str:
    pattern = re.compile(
        rf"(\n\n{re.escape(header)}\n)(.*?)(?=\n\n## |\Z)",
        re.DOTALL,
    )
    block = f"\n\n{header}\n{summary.strip()}"
    if pattern.search(system_content):
        return pattern.sub(block, system_content, count=1)
    return system_content.rstrip() + block


async def _summarize_text(system: str, user: str) -> str | None:
    try:
        return await llm_client.complete(
            system, user, temperature=0.2, max_tokens=2048,
        )
    except Exception:
        logger.warning("Context summary LLM call failed", exc_info=True)
        return None


def _truncate_oldest_history(
    history: list[dict],
    *,
    max_context_size: int,
    target: float,
) -> list[dict]:
    """Fallback: drop oldest history messages until under target budget."""
    keep = KEEP_RECENT_TURNS * 2
    if len(history) <= keep:
        return history

    target_tokens = int(max_context_size * target)
    trimmed = list(history)
    while len(trimmed) > keep and estimate_prompt_tokens([{"role": "user", "content": _format_turns_for_summary(trimmed)}]) > target_tokens // 4:
        trimmed = trimmed[2:]
    return trimmed


async def compress_agent_context(
    messages: list[dict],
    persisted_history: list[dict],
    *,
    prompt_tokens: int,
    max_context_size: int,
    threshold: float,
    target: float,
) -> CompressResult:
    if not should_compress(prompt_tokens, max_context_size, threshold):
        return CompressResult(messages, persisted_history, compressed=False)

    layout = _parse_layout(messages)
    if layout is None:
        return CompressResult(messages, persisted_history, compressed=False)

    new_messages = [dict(messages[0])]
    system_content = new_messages[0].get("content") or ""

    history_msgs = messages[1:layout.history_end]
    keep_count = KEEP_RECENT_TURNS * 2
    compressible_history = history_msgs[:-keep_count] if len(history_msgs) > keep_count else []
    kept_history = history_msgs[-keep_count:] if len(history_msgs) > keep_count else history_msgs

    summary_snippet: str | None = None
    new_persisted = list(persisted_history)

    if compressible_history:
        compressible_persisted = (
            persisted_history[:-keep_count]
            if len(persisted_history) > keep_count
            else []
        )
        kept_persisted = (
            persisted_history[-keep_count:]
            if len(persisted_history) > keep_count
            else list(persisted_history)
        )

        turns_text = _format_turns_for_summary(compressible_history)
        summary = await _summarize_text(_SUMMARY_SYSTEM, turns_text)

        if summary:
            system_content = _inject_summary_block(system_content, HISTORY_SUMMARY_HEADER, summary)
            summary_snippet = summary[:200]
            new_persisted = kept_persisted
        else:
            new_persisted = _truncate_oldest_history(persisted_history, max_context_size=max_context_size, target=target)
            kept_history = new_persisted[-keep_count:] if len(new_persisted) > keep_count else new_persisted
            system_content = _inject_summary_block(
                system_content,
                HISTORY_SUMMARY_HEADER,
                "[较早对话已截断以节省上下文]",
            )
            summary_snippet = "[truncated]"

    new_messages[0]["content"] = system_content
    new_messages.extend(kept_history)
    new_messages.append(dict(messages[layout.current_user_idx]))

    tool_tail = messages[layout.tool_tail_start:]
    target_tokens = int(max_context_size * target)

    if tool_tail and estimate_prompt_tokens(new_messages + tool_tail) > target_tokens:
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(tool_tail):
            if tool_tail[i].get("role") == "assistant" and tool_tail[i].get("tool_calls"):
                j = i + 1
                while j < len(tool_tail) and tool_tail[j].get("role") == "tool":
                    j += 1
                groups.append((i, j))
                i = j
            else:
                i += 1

        if len(groups) > KEEP_RECENT_TOOL_GROUPS:
            compress_end = groups[-KEEP_RECENT_TOOL_GROUPS][0]
            compress_slice = tool_tail[:compress_end]
            kept_slice = [dict(m) for m in tool_tail[compress_end:]]

            tool_text = _format_tool_tail_for_summary(compress_slice, 0, len(compress_slice))
            tool_summary = await _summarize_text(_TOOL_SUMMARY_SYSTEM, tool_text)

            if tool_summary:
                system_content = _inject_summary_block(
                    new_messages[0]["content"] or "",
                    TOOL_SUMMARY_HEADER,
                    tool_summary,
                )
                new_messages[0]["content"] = system_content
                if summary_snippet is None:
                    summary_snippet = tool_summary[:200]
            else:
                for msg in kept_slice:
                    if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                        content = msg["content"]
                        if len(content) > 500:
                            msg["content"] = truncate_to_budget(content, 125)
                system_content = _inject_summary_block(
                    new_messages[0]["content"] or "",
                    TOOL_SUMMARY_HEADER,
                    "[较早工具调用结果已截断以节省上下文]",
                )
                new_messages[0]["content"] = system_content

            new_messages.extend(kept_slice)
        else:
            new_messages.extend(tool_tail)
    else:
        new_messages.extend(tool_tail)

    logger.info(
        "Agent context compressed: prompt_tokens=%d threshold=%d target=%d summary=%s",
        prompt_tokens,
        int(max_context_size * threshold),
        target_tokens,
        (summary_snippet or "")[:80],
    )

    return CompressResult(
        messages=new_messages,
        persisted_history=new_persisted,
        compressed=True,
        summary_snippet=summary_snippet,
    )
