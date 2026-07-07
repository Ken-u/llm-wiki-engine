"""Agent orchestrator: tool-calling loop between LLM and agent tools.

Handles:
1. System prompt construction with tool use policy
2. Registering tool schemas with LLM
3. Executing tool calls and feeding results back
4. Per-call budget tracking — rejects excess calls via tool result
5. Streaming final text response via SSE events
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import AsyncGenerator

ShouldCancel = Callable[[], Awaitable[bool]] | None

from app.agents.context_compress import (
    compress_agent_context,
    resolve_prompt_tokens,
    should_compress,
)
from app.agents.tools import ToolContext, execute_tool, get_tool_definitions
from app.config import get_config
from app.llm.client import (
    complete_with_tools,
    finish_debug_llm_log_context,
    LLMResponse,
    log_debug_llm_event,
    reset_debug_llm_log_context,
    stream as llm_stream,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 20
COMPRESS_CONTEXT_TOOL = "compress_context"

REJECT_MSG = "已达到最大工具调用次数，请立即根据目前已获取的数据输出结果。"


@dataclass
class ToolTrace:
    """Record of a single tool call for debugging."""
    name: str
    arguments: dict
    result: dict


def _build_system_prompt(custom_prompt: str, has_ticket: bool, override_prompt: str = "") -> str:
    """Build system prompt with tool use policy."""
    if override_prompt.strip():
        return override_prompt
    ticket_policy = ""
    if has_ticket:
        ticket_policy = "\n".join([
            "",
            "案例库使用策略：",
            "- 如果用户问题涉及历史案例、具体故障经验、类似 issue、处理先例，可调用 search_ticket_cases 搜索案例库。",
            "- 搜索案例后，优先基于候选摘要（problem_summary、root_cause、resolution）直接回答。",
            "- 只有需要具体排查步骤、日志、完整上下文时才调用 read_ticket_case。",
            "- 不要一次读取多个案例全文。",
            "- 使用案例库时，回答中引用案例必须使用 [[case_id]] 格式，其中 case_id 为工具返回的纯数字案例 ID。",
            "- 正确示例：[[558753]]；错误示例：[[case558753]]、[[case_558753]]、[[CASE-558753]]、[[#558753]]。",
            "- 调用 read_ticket_case 时，case_id 参数必须为纯数字，不得添加 case、case_、CASE-、# 等前缀。",
            "- 案例内容只能通过 search_ticket_cases 和 read_ticket_case 获取。",
            "- 禁止对案例库文件使用 read_raw、grep_raw、read_wiki_page。",
            "- read_ticket_case 返回的字段中不包含本地文件路径，不要尝试读取案例 raw 源文件。",
            "- read_ticket_case 的 section 参数用于指定章节（如'处理过程'、'最终处理方案'），不是 session。",
            "- 搜索结果的摘要足够回答时，不要调用 read_ticket_case。",
        ])

    policy = "\n".join([
        "你是一个知识问答助手。你可以使用工具来检索知识库和案例库。",
        "",
        "工具使用策略：",
        "- 优先使用 search_wiki 搜索与问题相关的知识页面。",
        "- 搜索结果中的 snippet 不足以支撑结论时，使用 read_wiki_page 读取页面全文。",
        "- 索引搜索不到时，可使用 grep_raw 直接在原文件中搜索关键词。",
        "- 不要猜测答案，始终基于工具返回的知识库内容回答。",
        f"{ticket_policy}",
        "- 凡是参考任意知识库或案例库内容得出的事实、结论、步骤或建议，都必须在对应句子附近使用 [[...]] 标注引用。",
        "- 知识库引用使用 [[wiki/path-or-page-name]] 或工具返回的页面 path/title；案例库引用使用 [[case_id]]，例如 [[558753]]。",
        "- 如果知识库中没有找到相关信息，请如实说明。",
    ])

    if custom_prompt:
        return f"{custom_prompt}\n\n{policy}"
    return policy


DEFAULT_DEBUG_RESULT_LIMIT = 2000


def _truncate_for_debug(result: dict, limit: int = DEFAULT_DEBUG_RESULT_LIMIT) -> dict:
    """Truncate long string values in tool results for SSE debug events."""
    out = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + f"\n... ({len(v)} chars total)"
        elif isinstance(v, list) and len(v) > 30:
            out[k] = v[:30]
            out[f"_{k}_truncated"] = f"{len(v)} items total, showing first 30"
        else:
            out[k] = v
    return out


async def _abort_if_cancelled(should_cancel: ShouldCancel) -> bool:
    if should_cancel is not None and await should_cancel():
        logger.info("Client disconnected, aborting agent turn")
        return True
    return False


def _done_event(
    traces: list[ToolTrace],
    *,
    compressed_history: list[dict] | None = None,
    context_compressed: bool = False,
) -> str:
    payload: dict = {
        "done": True,
        "tool_traces": [{"name": t.name, "arguments": t.arguments} for t in traces],
        "context_compressed": context_compressed,
    }
    if compressed_history is not None:
        payload["compressed_history"] = compressed_history
    return json.dumps(payload)


async def run_agent_turn(
    message: str,
    history: list[dict],
    system_prompt: str,
    ctx: ToolContext,
    *,
    system_prompt_override: str = "",
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    debug_result_limit: int = DEFAULT_DEBUG_RESULT_LIMIT,
    should_cancel: ShouldCancel = None,
) -> AsyncGenerator[str, None]:
    """Run one agent turn: tool-calling loop then stream final answer.

    Yields SSE-compatible JSON strings:
      {"token": "..."}          — streamed text chunk
      {"tool_call": {...}}      — tool call event (may include "rejected": true)
      {"done": true, ...}       — completion marker
    """
    reset_debug_llm_log_context()
    full_system = _build_system_prompt(system_prompt, ctx.ticket_project is not None, system_prompt_override)
    tool_defs = get_tool_definitions(ctx)
    cfg = get_config().llm

    messages: list[dict] = [{"role": "system", "content": full_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    persisted_history = list(history)
    context_compressed = False

    traces: list[ToolTrace] = []
    used_ticket = False
    tool_call_count = 0

    # Upper bound on LLM rounds: max_tool_calls + 2 to allow rejection + final text
    for _round in range(max_tool_calls + 2):
        if await _abort_if_cancelled(should_cancel):
            finish_debug_llm_log_context()
            return

        resp: LLMResponse = await complete_with_tools(
            messages,
            tool_defs,
            max_tokens=4096,
            should_cancel=should_cancel,
        )

        if await _abort_if_cancelled(should_cancel):
            finish_debug_llm_log_context()
            return

        prompt_tokens, usage_source = resolve_prompt_tokens(
            messages,
            api_prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
        )
        if usage_source == "estimated":
            logger.debug("Using estimated prompt_tokens=%d for context compression", prompt_tokens)

        needs_compress = should_compress(
            prompt_tokens,
            cfg.max_context_size,
            cfg.context_compress_threshold,
        )
        if needs_compress:
            yield json.dumps({
                "tool_call": {
                    "name": COMPRESS_CONTEXT_TOOL,
                    "arguments": {"prompt_tokens": prompt_tokens, "usage_source": usage_source},
                },
            })

        compress_result = await compress_agent_context(
            messages,
            persisted_history,
            prompt_tokens=prompt_tokens,
            max_context_size=cfg.max_context_size,
            threshold=cfg.context_compress_threshold,
            target=cfg.context_compress_target,
        )

        if await _abort_if_cancelled(should_cancel):
            finish_debug_llm_log_context()
            return

        if needs_compress:
            yield json.dumps({
                "tool_result": {
                    "name": COMPRESS_CONTEXT_TOOL,
                    "result": {
                        "compressed": compress_result.compressed,
                        "prompt_tokens": prompt_tokens,
                        "usage_source": usage_source,
                    },
                },
            })

        if compress_result.compressed:
            messages = compress_result.messages
            persisted_history = compress_result.persisted_history
            context_compressed = True
            logger.info(
                "Context compressed after %s prompt_tokens=%d",
                usage_source,
                prompt_tokens,
            )

        if not resp.tool_calls:
            if resp.content:
                resp_content = resp.content
                if used_ticket and "参考了案例库" not in resp.content:
                    resp_content += "\n\n> 以上结论参考了案例库。"
                yield json.dumps({"token": resp_content})
            yield _done_event(
                traces,
                compressed_history=persisted_history if context_compressed else None,
                context_compressed=context_compressed,
            )
            finish_debug_llm_log_context()
            return

        # Build assistant message
        assistant_msg: dict = {"role": "assistant", "content": resp.content, "tool_calls": []}
        for tc in resp.tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            })
        messages.append(assistant_msg)

        for tc in resp.tool_calls:
            if await _abort_if_cancelled(should_cancel):
                finish_debug_llm_log_context()
                return

            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_call_count >= max_tool_calls:
                yield json.dumps({"tool_call": {"name": tc.name, "arguments": args, "rejected": True}})
                reject_result = {"rejected": True, "message": REJECT_MSG}
                log_debug_llm_event("tool_execution", {
                    "name": tc.name,
                    "arguments": args,
                    "result": reject_result,
                    "rejected": True,
                })
                yield json.dumps({"tool_result": {"name": tc.name, "result": reject_result}})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": REJECT_MSG,
                })
                traces.append(ToolTrace(name=tc.name, arguments=args, result=reject_result))
            else:
                yield json.dumps({"tool_call": {"name": tc.name, "arguments": args}})
                result = await execute_tool(tc.name, args, ctx)
                log_debug_llm_event("tool_execution", {
                    "name": tc.name,
                    "arguments": args,
                    "result": result,
                    "rejected": False,
                })
                yield json.dumps({"tool_result": {"name": tc.name, "result": _truncate_for_debug(result, debug_result_limit)}})
                traces.append(ToolTrace(name=tc.name, arguments=args, result=result))

                if tc.name in ("search_ticket_cases", "read_ticket_case"):
                    if "error" not in result:
                        used_ticket = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                tool_call_count += 1

    # Exhausted all rounds — hard fallback, stream without tools
    async for token in llm_stream(
        messages,
        temperature=cfg.chat_temperature,
        max_tokens=4096,
        should_cancel=should_cancel,
    ):
        yield json.dumps({"token": token})

    yield _done_event(
        traces,
        compressed_history=persisted_history if context_compressed else None,
        context_compressed=context_compressed,
    )
    finish_debug_llm_log_context()
